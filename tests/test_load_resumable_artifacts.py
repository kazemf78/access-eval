"""
Unit tests for the final+partial merge helpers added to fix "resume discards
an in-progress partial's already-computed rows whenever a final also
happens to exist" (answer_generation.py's `_load_resumable_answers`,
evaluation.py's `_load_resumable_grades`). Pure pandas/CSV I/O, no LLM calls,
no mocking needed.
"""

import pandas as pd
import pytest

from access_eval.pipeline.answer_generation import _load_resumable_answers
from access_eval.pipeline.evaluation import _load_resumable_grades


class TestLoadResumableAnswers:
    def test_only_final_exists(self, tmp_path):
        final = tmp_path / "final.csv"
        partial = tmp_path / "partial.csv"
        pd.DataFrame([
            {"model": "gpt-4.1", "idx": 0, "original_question": "Q1", "paraphrased_question": "Q1",
             "answer": "A", "repeat_id": 0, "gen_config_hash": "cfg_a"},
        ]).to_csv(final, index=False)

        rows = _load_resumable_answers(final, partial)
        assert len(rows) == 1

    def test_only_partial_exists(self, tmp_path):
        final = tmp_path / "final.csv"
        partial = tmp_path / "partial.csv"
        pd.DataFrame([
            {"model": "gpt-4.1", "idx": 0, "original_question": "Q1", "paraphrased_question": "Q1",
             "answer": "A", "repeat_id": 0, "gen_config_hash": "cfg_a"},
        ]).to_csv(partial, index=False)

        rows = _load_resumable_answers(final, partial)
        assert len(rows) == 1

    def test_neither_exists(self, tmp_path):
        assert _load_resumable_answers(tmp_path / "final.csv", tmp_path / "partial.csv") == []

    def test_final_and_further_along_partial_are_merged_not_final_only(self, tmp_path):
        """The scenario from the review: run A completes with repeat_id=0
        (final); run B generates repeat_id=1,2 into partial before crashing.
        Both must survive the next resume, not just final's repeat_id=0."""
        final = tmp_path / "final.csv"
        partial = tmp_path / "partial.csv"
        pd.DataFrame([
            {"model": "gpt-4.1", "idx": 0, "original_question": "Q1", "paraphrased_question": "Q1",
             "answer": "A", "repeat_id": 0, "gen_config_hash": "cfg_a"},
        ]).to_csv(final, index=False)
        pd.DataFrame([
            {"model": "gpt-4.1", "idx": 0, "original_question": "Q1", "paraphrased_question": "Q1",
             "answer": "A", "repeat_id": 0, "gen_config_hash": "cfg_a"},
            {"model": "gpt-4.1", "idx": 0, "original_question": "Q1", "paraphrased_question": "Q1",
             "answer": "B", "repeat_id": 1, "gen_config_hash": "cfg_a"},
            {"model": "gpt-4.1", "idx": 0, "original_question": "Q1", "paraphrased_question": "Q1",
             "answer": "C", "repeat_id": 2, "gen_config_hash": "cfg_a"},
        ]).to_csv(partial, index=False)

        rows = _load_resumable_answers(final, partial)
        assert {r["repeat_id"] for r in rows} == {0, 1, 2}
        assert len(rows) == 3  # the overlapping repeat_id=0 row deduped, not duplicated

    def test_final_wins_on_exact_identity_collision(self, tmp_path):
        """If final and partial both have a row for the exact same identity
        (formulation_hash, repeat_id, gen_config_hash) with DIFFERENT answer
        text (shouldn't normally happen, but), final's data wins -- it's the
        more authoritative/later-completed artifact."""
        final = tmp_path / "final.csv"
        partial = tmp_path / "partial.csv"
        pd.DataFrame([
            {"model": "gpt-4.1", "idx": 0, "original_question": "Q1", "paraphrased_question": "Q1",
             "answer": "FINAL_ANSWER", "repeat_id": 0, "gen_config_hash": "cfg_a"},
        ]).to_csv(final, index=False)
        pd.DataFrame([
            {"model": "gpt-4.1", "idx": 0, "original_question": "Q1", "paraphrased_question": "Q1",
             "answer": "STALE_PARTIAL_ANSWER", "repeat_id": 0, "gen_config_hash": "cfg_a"},
        ]).to_csv(partial, index=False)

        rows = _load_resumable_answers(final, partial)
        assert len(rows) == 1
        assert rows[0]["answer"] == "FINAL_ANSWER"


class TestLoadResumableGrades:
    def test_final_and_further_along_partial_are_merged(self, tmp_path):
        final = tmp_path / "final.csv"
        partial = tmp_path / "partial.csv"
        pd.DataFrame([
            {"answer_idx": 0, "idx": 0, "model": "gpt-4.1", "original_question": "Q1",
             "paraphrased_question": "Q1", "answer": "A", "repeat_id": 0,
             "gen_config_hash": "cfg_a", "eval_config_hash": "cfg_e", "verdict": "correct"},
        ]).to_csv(final, index=False)
        pd.DataFrame([
            {"answer_idx": 0, "idx": 0, "model": "gpt-4.1", "original_question": "Q1",
             "paraphrased_question": "Q1", "answer": "A", "repeat_id": 0,
             "gen_config_hash": "cfg_a", "eval_config_hash": "cfg_e", "verdict": "correct"},
            {"answer_idx": 1, "idx": 0, "model": "gpt-4.1", "original_question": "Q1",
             "paraphrased_question": "Q1", "answer": "B", "repeat_id": 1,
             "gen_config_hash": "cfg_a", "eval_config_hash": "cfg_e", "verdict": "incorrect"},
        ]).to_csv(partial, index=False)

        rows = _load_resumable_grades(final, partial)
        assert {r["repeat_id"] for r in rows} == {0, 1}
        assert len(rows) == 2

    def test_cross_model_rows_both_preserved_by_dedup_key(self, tmp_path):
        """Dedup must key on model/gen_config_hash too, not just
        formulation_hash/repeat_id -- otherwise merging final+partial could
        itself introduce the cross-model collision this session's other fix
        (EvalTarget.row_key) closes on the read side."""
        final = tmp_path / "final.csv"
        partial = tmp_path / "partial.csv"
        pd.DataFrame([
            {"answer_idx": 0, "idx": 0, "model": "gpt-4.1", "original_question": "Q1",
             "paraphrased_question": "Q1", "answer": "Paris", "repeat_id": 0,
             "gen_config_hash": "cfg_gpt", "eval_config_hash": "cfg_e", "verdict": "correct"},
        ]).to_csv(final, index=False)
        pd.DataFrame([
            {"answer_idx": 5, "idx": 0, "model": "qwen3-8b", "original_question": "Q1",
             "paraphrased_question": "Q1", "answer": "Paris", "repeat_id": 0,
             "gen_config_hash": "cfg_qwen", "eval_config_hash": "cfg_e", "verdict": "correct"},
        ]).to_csv(partial, index=False)

        rows = _load_resumable_grades(final, partial)
        assert {r["model"] for r in rows} == {"gpt-4.1", "qwen3-8b"}
        assert len(rows) == 2
