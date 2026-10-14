"""
Unit tests for src/access_eval/io/paraphrase_store.py.

Pure DataFrame/file logic -- no LLM calls, no mocking needed. Uses a real
RunPaths rooted at tmp_path (never the real run_artifacts/ tree).
"""

import pandas as pd
import pytest

from access_eval.core.paths import ProjectPaths
from access_eval.io.judge_cache import append_judge_verdicts, prompt_hash_for
from access_eval.io.paraphrase_store import (
    load_generated_paraphrases, accepted_paraphrases, dedup_by_overlap, cross_validated_paraphrases,
)
from access_eval.prompts import load_prompt


@pytest.fixture
def rp(tmp_path):
    return ProjectPaths(project_root=tmp_path).run_paths("SimpleQA", "unit_test")


def _write_paraphrases(rp, rows):
    rp.paraphrases_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(rp.paraphrases_file(), index=False)


class TestLoadGeneratedParaphrasesDedup:
    def test_no_file_returns_empty(self, rp):
        assert load_generated_paraphrases(rp).empty

    def test_literal_duplicate_rows_deduped_by_idx_and_para_hash(self, rp):
        """Defensive safety net (Fix C): a candidate appended twice for the
        same idx (e.g. across two top-up rounds that both happened to
        produce identical text) must not inflate downstream counts."""
        _write_paraphrases(rp, [
            {"idx": 0, "original_question": "Q1", "paraphrased_question": "P1"},
            {"idx": 0, "original_question": "Q1", "paraphrased_question": "P1"},  # literal dup
            {"idx": 0, "original_question": "Q1", "paraphrased_question": "P2"},  # genuinely distinct
        ])
        df = load_generated_paraphrases(rp)
        assert len(df) == 2
        assert set(df["paraphrased_question"]) == {"P1", "P2"}

    def test_same_text_different_idx_not_deduped(self, rp):
        """Dedup key is (idx, para_hash), not para_hash alone -- the same
        paraphrase text for two DIFFERENT source questions/idx is not a
        duplicate."""
        _write_paraphrases(rp, [
            {"idx": 0, "original_question": "Q1", "paraphrased_question": "Same text"},
            {"idx": 1, "original_question": "Q2", "paraphrased_question": "Same text"},
        ])
        df = load_generated_paraphrases(rp)
        assert len(df) == 2
        assert set(df["idx"]) == {0, 1}


def _judge(rp, judge_model, prompt_name, rows):
    """rows: list of (original, paraphrased, llm_equiv) -- writes verdicts
    into the shared judge cache, matching what _judge_pending_candidates
    would produce."""
    from access_eval.io.identity import paraphrase_content_hash
    p_hash = prompt_hash_for(load_prompt(f"paraphrase/{prompt_name}"))
    append_judge_verdicts(rp, [
        {
            "para_hash": paraphrase_content_hash(o, p),
            "original_question": o, "paraphrased_question": p,
            "judge_model": judge_model, "prompt_name": prompt_name, "prompt_hash": p_hash,
            "llm_raw": "YES" if equiv else "NO", "llm_equiv": equiv,
        }
        for o, p, equiv in rows
    ])


class TestAcceptedParaphrases:
    def test_unjudged_candidate_excluded(self, rp):
        _write_paraphrases(rp, [{"idx": 0, "original_question": "Q1", "paraphrased_question": "P1"}])
        assert accepted_paraphrases(rp, "judge-a", "v1").empty

    def test_only_accepted_verdicts_included(self, rp):
        _write_paraphrases(rp, [
            {"idx": 0, "original_question": "Q1", "paraphrased_question": "Accepted one"},
            {"idx": 0, "original_question": "Q1", "paraphrased_question": "Rejected one"},
        ])
        _judge(rp, "judge-a", "v1", [
            ("Q1", "Accepted one", True),
            ("Q1", "Rejected one", False),
        ])
        accepted = accepted_paraphrases(rp, "judge-a", "v1")
        assert list(accepted["paraphrased_question"]) == ["Accepted one"]

    def test_different_judge_model_is_a_separate_view(self, rp):
        _write_paraphrases(rp, [{"idx": 0, "original_question": "Q1", "paraphrased_question": "P1"}])
        _judge(rp, "judge-a", "v1", [("Q1", "P1", True)])
        assert len(accepted_paraphrases(rp, "judge-a", "v1")) == 1
        assert accepted_paraphrases(rp, "judge-b", "v1").empty

    def test_different_prompt_name_is_a_separate_view(self, rp):
        _write_paraphrases(rp, [{"idx": 0, "original_question": "Q1", "paraphrased_question": "P1"}])
        _judge(rp, "judge-a", "v1", [("Q1", "P1", True)])
        assert len(accepted_paraphrases(rp, "judge-a", "v1")) == 1
        assert accepted_paraphrases(rp, "judge-a", "v2").empty

    def test_stricter_rejudge_drops_from_accepted_view_without_touching_original(self, rp):
        _write_paraphrases(rp, [{"idx": 0, "original_question": "Q1", "paraphrased_question": "P1"}])
        _judge(rp, "lenient-judge", "v1", [("Q1", "P1", True)])
        _judge(rp, "strict-judge", "v1", [("Q1", "P1", False)])
        assert len(accepted_paraphrases(rp, "lenient-judge", "v1")) == 1
        assert accepted_paraphrases(rp, "strict-judge", "v1").empty

    def test_reflects_growing_judge_cache_without_a_rebuild_step(self, rp):
        """The view is computed at read time, not materialized -- appending
        a new verdict changes the next call's result with no separate
        rebuild/refresh step."""
        _write_paraphrases(rp, [
            {"idx": 0, "original_question": "Q1", "paraphrased_question": "P1"},
            {"idx": 0, "original_question": "Q1", "paraphrased_question": "P2"},
        ])
        _judge(rp, "judge-a", "v1", [("Q1", "P1", True)])
        assert len(accepted_paraphrases(rp, "judge-a", "v1")) == 1

        _judge(rp, "judge-a", "v1", [("Q1", "P2", True)])
        assert len(accepted_paraphrases(rp, "judge-a", "v1")) == 2

    def test_no_prompt_name_matches_any_prompt(self, rp):
        """prompt_name=None (default) -- accepted under judge-a regardless of
        which prompt produced the verdict."""
        _write_paraphrases(rp, [{"idx": 0, "original_question": "Q1", "paraphrased_question": "P1"}])
        _judge(rp, "judge-a", "v1", [("Q1", "P1", True)])
        assert len(accepted_paraphrases(rp, "judge-a")) == 1
        assert list(accepted_paraphrases(rp, "judge-a")["paraphrased_question"]) == ["P1"]

    def test_no_prompt_name_still_scoped_to_judge_model(self, rp):
        _write_paraphrases(rp, [{"idx": 0, "original_question": "Q1", "paraphrased_question": "P1"}])
        _judge(rp, "judge-a", "v1", [("Q1", "P1", True)])
        assert accepted_paraphrases(rp, "judge-b").empty

    def test_no_prompt_name_dedups_a_hash_judged_true_under_two_prompts(self, rp):
        """Same para_hash, same judge_model, two different prompts both say
        YES -- must appear once, not twice, in the None-prompt_name view."""
        _write_paraphrases(rp, [{"idx": 0, "original_question": "Q1", "paraphrased_question": "P1"}])
        _judge(rp, "judge-a", "v1", [("Q1", "P1", True)])
        _judge(rp, "judge-a", "v2", [("Q1", "P1", True)])
        assert len(accepted_paraphrases(rp, "judge-a")) == 1


class TestDedupByOverlap:
    def test_empty_df_is_noop(self, rp):
        empty = accepted_paraphrases(rp, "judge-a", "v1")  # empty, right columns
        assert dedup_by_overlap(empty).empty

    def test_near_duplicate_within_same_idx_is_dropped(self, rp):
        df = pd.DataFrame([
            {"idx": 0, "paraphrased_question": "What year did the war end for good"},
            {"idx": 0, "paraphrased_question": "What year did the war end for real"},  # 6/7 words shared
        ])
        kept = dedup_by_overlap(df, overlap_max=0.5)
        assert len(kept) == 1
        assert kept.iloc[0]["paraphrased_question"] == "What year did the war end for good"

    def test_distinct_text_within_same_idx_is_kept(self, rp):
        df = pd.DataFrame([
            {"idx": 0, "paraphrased_question": "What year did the war end"},
            {"idx": 0, "paraphrased_question": "Who was the president at the time"},
        ])
        assert len(dedup_by_overlap(df, overlap_max=0.5)) == 2

    def test_overlap_only_checked_within_same_idx(self, rp):
        """Two idx can legitimately share near-identical paraphrase text
        (e.g. two questions that happen to be worded alike) -- overlap is a
        per-idx concept, never cross-idx."""
        df = pd.DataFrame([
            {"idx": 0, "paraphrased_question": "Same wording here"},
            {"idx": 1, "paraphrased_question": "Same wording here"},
        ])
        assert len(dedup_by_overlap(df, overlap_max=0.5)) == 2


class TestCrossValidatedParaphrases:
    def _write_with_resp_model(self, rp, rows):
        """rows: list of (idx, original, paraphrased, resp_model) -- resp_model
        is whichever model WROTE the candidate text (separate from whichever
        judge_model later accepts/rejects it via _judge below)."""
        _write_paraphrases(rp, [
            {"idx": idx, "original_question": o, "paraphrased_question": p, "resp_model": rm}
            for idx, o, p, rm in rows
        ])

    def test_judge_validating_its_own_writing_does_not_count(self, rp):
        """Written by judge-a (resp_model="judge-a") and also judged by
        judge-a -- a judge validating its own writing isn't independent
        validation."""
        self._write_with_resp_model(rp, [(0, "Q1", "P1", "judge-a")])
        _judge(rp, "judge-a", "v1", [("Q1", "P1", True)])
        assert cross_validated_paraphrases(rp, "judge-a", "v1").empty

    def test_validated_by_a_model_that_did_not_write_it_counts(self, rp):
        """Written by writer-model, judged by judge-a -- two different
        models, so this counts even though writer-model itself never
        re-judged anything."""
        self._write_with_resp_model(rp, [(0, "Q1", "P1", "writer-model")])
        _judge(rp, "judge-a", "v1", [("Q1", "P1", True)])
        cross = cross_validated_paraphrases(rp, "judge-a", "v1")
        assert list(cross["paraphrased_question"]) == ["P1"]

    def test_rejected_by_judge_excluded(self, rp):
        self._write_with_resp_model(rp, [(0, "Q1", "P1", "writer-model")])
        _judge(rp, "judge-a", "v1", [("Q1", "P1", False)])
        assert cross_validated_paraphrases(rp, "judge-a", "v1").empty

    def test_missing_resp_model_column_returns_empty(self, rp):
        """A pool that predates the resp_model provenance column can't
        answer 'who wrote this' -- excluded rather than guessed."""
        _write_paraphrases(rp, [{"idx": 0, "original_question": "Q1", "paraphrased_question": "P1"}])
        _judge(rp, "judge-a", "v1", [("Q1", "P1", True)])
        assert cross_validated_paraphrases(rp, "judge-a", "v1").empty

    def test_union_across_multiple_judge_models(self, rp):
        """Validated by ANY of the given judge_models counts -- mirrors
        GSM8K's gpt-5 / gpt-5.4-mini backfill, where each paraphrase only
        needs corroboration from one of the two stronger judges."""
        self._write_with_resp_model(rp, [
            (0, "Q1", "P1", "writer-model"),  # validated only by judge-b
            (0, "Q1", "P2", "writer-model"),  # validated only by judge-c
            (0, "Q1", "P3", "writer-model"),  # validated by neither
        ])
        _judge(rp, "judge-b", "v1", [("Q1", "P1", True), ("Q1", "P2", False), ("Q1", "P3", False)])
        _judge(rp, "judge-c", "v1", [("Q1", "P1", False), ("Q1", "P2", True), ("Q1", "P3", False)])
        cross = cross_validated_paraphrases(rp, ["judge-b", "judge-c"], "v1")
        assert set(cross["paraphrased_question"]) == {"P1", "P2"}

    def test_union_reapplies_overlap_dedup_across_judges(self, rp):
        """Two rows independently validated by DIFFERENT judges were never
        compared to each other at any generation-time accept step -- the
        union must still re-check overlap between them."""
        self._write_with_resp_model(rp, [
            (0, "Q1", "What year did the war end for good", "writer-model"),
            (0, "Q1", "What year did the war end for real", "writer-model"),  # near-dup of the above
        ])
        _judge(rp, "judge-b", "v1", [("Q1", "What year did the war end for good", True)])
        _judge(rp, "judge-c", "v1", [("Q1", "What year did the war end for real", True)])
        cross = cross_validated_paraphrases(rp, ["judge-b", "judge-c"], "v1", overlap_max=0.5)
        assert len(cross) == 1

    def test_require_different_generation_model_false_includes_self_written_rows(self, rp):
        """Written AND judged by judge-a. Excluded by default; included with
        require_different_generation_model=False (e.g. to also reach
        paraphrase_topup.py-style rows regardless of who wrote them)."""
        self._write_with_resp_model(rp, [(0, "Q1", "P1", "judge-a")])
        _judge(rp, "judge-a", "v1", [("Q1", "P1", True)])
        assert cross_validated_paraphrases(rp, "judge-a", "v1").empty
        included = cross_validated_paraphrases(rp, "judge-a", "v1", require_different_generation_model=False)
        assert list(included["paraphrased_question"]) == ["P1"]

    def test_require_different_generation_model_false_still_excludes_rejected(self, rp):
        self._write_with_resp_model(rp, [(0, "Q1", "P1", "judge-a")])
        _judge(rp, "judge-a", "v1", [("Q1", "P1", False)])
        assert cross_validated_paraphrases(rp, "judge-a", "v1", require_different_generation_model=False).empty

    def test_require_different_generation_model_false_unions_self_and_other_written(self, rp):
        """Mixed pool: one row judge-a both wrote and judged, one row judge-b
        judged but writer-model wrote -- both counted with
        require_different_generation_model=False, only the second one with
        the default True."""
        self._write_with_resp_model(rp, [
            (0, "Q1", "P1", "judge-a"),       # judge-a wrote AND judged this one
            (0, "Q1", "P2", "writer-model"),  # judge-b judged; writer-model wrote it
        ])
        _judge(rp, "judge-a", "v1", [("Q1", "P1", True)])
        _judge(rp, "judge-b", "v1", [("Q1", "P2", True)])

        strict = cross_validated_paraphrases(rp, ["judge-a", "judge-b"], "v1")
        assert set(strict["paraphrased_question"]) == {"P2"}

        lenient = cross_validated_paraphrases(rp, ["judge-a", "judge-b"], "v1", require_different_generation_model=False)
        assert set(lenient["paraphrased_question"]) == {"P1", "P2"}
