"""
Reproduction-style tests for answer_generation.py's keyed resume (§3 of the
reconciliation refactor): calls the real `run_answer_generation()`, with only
the LLM call itself mocked (matching the project's established verification
convention). Artifacts are confined to `tmp_path` via a monkeypatched
`ProjectPaths()` -- never the real `run_artifacts/` tree.
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock

from access_eval.core.paths import ProjectPaths
from access_eval.io.identity import paraphrase_content_hash
from access_eval.io.judge_cache import append_judge_verdicts, prompt_hash_for
from access_eval.models.openai_api import OpenAIAPILLM
from access_eval.pipeline import answer_generation as ag
from access_eval.prompts import load_prompt

MODEL = "gpt-test-model"  # matches the "gpt-" API pattern in models/factory.py


def _mock_llm(answer_text="42"):
    llm = MagicMock(spec=OpenAIAPILLM)
    llm.batch.side_effect = lambda prompts, **kw: [answer_text] * len(prompts)
    return llm


@pytest.fixture
def rp(tmp_path, monkeypatch):
    """Real RunPaths, rooted at tmp_path -- never the real run_artifacts/ tree."""
    monkeypatch.setattr(ag, "ProjectPaths", lambda: ProjectPaths(project_root=tmp_path))
    paths = ProjectPaths(project_root=tmp_path)
    rp = paths.run_paths("SimpleQA", "unit_test")
    rp.ensure_dirs()
    para_df = pd.DataFrame([
        {"idx": 0, "original_question": "What is 1+1?", "paraphrased_question": "What is 1+1?"},
        {"idx": 1, "original_question": "What is 2+2?", "paraphrased_question": "What is 2+2?"},
    ])
    para_df.to_csv(rp.paraphrases_file(), index=False)
    return rp


def _run(monkeypatch, llm=None, **overrides):
    monkeypatch.setattr(ag, "get_llm_from_list", lambda *a, **k: llm or _mock_llm())
    kwargs = dict(
        dataset="SimpleQA", experiment_flag="unit_test",
        api_models=[MODEL], local_models=[], question_subset="original_only",
        resume=True, force=False, repeat_ids=(0,),
    )
    kwargs.update(overrides)
    ag.run_answer_generation(**kwargs)


class TestFreshTargetAlreadySatisfied:
    def test_second_call_same_target_computes_nothing_new(self, tmp_path, monkeypatch, rp):
        llm = _mock_llm()
        _run(monkeypatch, llm=llm)
        final_path = rp.answers_file(subset="original_only", model=MODEL, conf_suffix=rp.conf_suffix(temperature=0.0))
        assert len(pd.read_csv(final_path)) == 2  # 2 formulations x 1 repeat

        llm2 = _mock_llm()
        _run(monkeypatch, llm=llm2)  # same repeat_ids=(0,) again
        llm2.batch.assert_not_called()  # nothing new needed -> no LLM calls at all
        assert len(pd.read_csv(final_path)) == 2


class TestGapInRepeatIds:
    def test_raising_target_generates_exactly_the_deficit(self, tmp_path, monkeypatch, rp):
        _run(monkeypatch, repeat_ids=(0,))
        final_path = rp.answers_file(subset="original_only", model=MODEL, conf_suffix=rp.conf_suffix(temperature=0.0))
        after_first = pd.read_csv(final_path)
        assert len(after_first) == 2
        assert set(after_first["repeat_id"]) == {0}

        llm2 = _mock_llm()
        _run(monkeypatch, llm=llm2, repeat_ids=(0, 1, 2))
        after_second = pd.read_csv(final_path)
        # 2 formulations x 3 repeats = 6 rows total; only 4 new prompts sent
        # (repeat_id=0 already existed for both formulations).
        assert len(after_second) == 6
        assert llm2.batch.call_count == 1
        (sent_prompts,), _ = llm2.batch.call_args
        assert len(sent_prompts) == 4
        for fh in after_second["original_question"].unique():
            repeats = set(after_second[after_second["original_question"] == fh]["repeat_id"])
            assert repeats == {0, 1, 2}

    def test_noncontiguous_existing_repeats_fill_exact_gap(self, tmp_path, monkeypatch, rp):
        """Repeats {0,2} pre-exist; target is {0,1,2} -- only repeat_id=1 is missing."""
        final_path = rp.answers_file(subset="original_only", model=MODEL, conf_suffix=rp.conf_suffix(temperature=0.0))
        rp.answers_dir.mkdir(exist_ok=True)
        # Seed existing rows directly (bypassing generation) for repeats 0 and 2,
        # for idx=0 only.
        gen_config_hash = ag.compute_gen_config_hash(
            model=MODEL, revision=None, temperature=0.0, max_tokens=256,
            prompt_name="factual_qa",
            prompt_template_text=ag.load_prompt("answer/factual_qa"),
        )
        seeded = pd.DataFrame([
            {"model": MODEL, "idx": 0, "original_question": "What is 1+1?",
             "paraphrased_question": "What is 1+1?", "answer": "2",
             "repeat_id": 0, "gen_config_hash": gen_config_hash},
            {"model": MODEL, "idx": 0, "original_question": "What is 1+1?",
             "paraphrased_question": "What is 1+1?", "answer": "2",
             "repeat_id": 2, "gen_config_hash": gen_config_hash},
        ])
        seeded.to_csv(final_path, index=False)

        # Trim paraphrases file to just idx=0 so we only reason about one formulation.
        pd.DataFrame([{"idx": 0, "original_question": "What is 1+1?", "paraphrased_question": "What is 1+1?"}]) \
            .to_csv(rp.paraphrases_file(), index=False)

        llm = _mock_llm(answer_text="2")
        _run(monkeypatch, llm=llm, repeat_ids=(0, 1, 2))

        (sent_prompts,), _ = llm.batch.call_args
        assert len(sent_prompts) == 1  # exactly the one missing repeat

        final_df = pd.read_csv(final_path)
        assert len(final_df) == 3
        assert set(final_df["repeat_id"]) == {0, 1, 2}


class TestConfigChangeTreatedAsFullyMissing:
    def test_different_temperature_does_not_satisfy_target_and_preserves_old_config_rows(self, tmp_path, monkeypatch, rp):
        _run(monkeypatch, repeat_ids=(0,), temperature=0.0)
        final_path = rp.answers_file(subset="original_only", model=MODEL, conf_suffix=rp.conf_suffix(temperature=0.0))
        after_first = pd.read_csv(final_path)
        assert len(after_first) == 2
        cfg_a = after_first["gen_config_hash"].iloc[0]

        # A different temperature writes to a DIFFERENT conf_suffix file (by
        # design, conf_suffix is part of the filename) -- confirm the new
        # config is treated as fully missing rather than satisfied by cfg_a's
        # rows, and that cfg_a's own file is left untouched.
        llm2 = _mock_llm()
        _run(monkeypatch, llm=llm2, repeat_ids=(0,), temperature=0.7)
        (sent_prompts,), _ = llm2.batch.call_args
        assert len(sent_prompts) == 2  # both formulations, nothing reused from cfg_a

        final_path_b = rp.answers_file(subset="original_only", model=MODEL, conf_suffix=rp.conf_suffix(temperature=0.7))
        after_second = pd.read_csv(final_path_b)
        cfg_b = after_second["gen_config_hash"].iloc[0]
        assert cfg_b != cfg_a
        assert len(after_second) == 2

        # cfg_a's own file is completely untouched by the cfg_b run.
        assert len(pd.read_csv(final_path)) == 2


def _judge(rp, judge_model, prompt_name, rows):
    """rows: list of (original, paraphrased, llm_equiv)."""
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


class TestParaphraseJudgeFilter:
    """paraphrase_judge_model=None (the default) must keep the pre-existing
    behavior byte-for-byte (see TestFreshTargetAlreadySatisfied etc. above);
    these cover the opt-in selection path added on top of it."""

    def test_default_is_unfiltered_pool(self, tmp_path, monkeypatch, rp):
        llm = _mock_llm()
        _run(monkeypatch, llm=llm)
        (sent_prompts,), _ = llm.batch.call_args
        assert len(sent_prompts) == 2  # both rp fixture rows, no filtering applied

    def test_judge_filter_restricts_to_cross_validated_rows(self, tmp_path, monkeypatch, rp):
        pd.DataFrame([
            {"idx": 0, "original_question": "What is 1+1?", "paraphrased_question": "What is 1+1?",
             "gen_judge_model": "gen-judge", "resp_model": "gen-model"},
            {"idx": 1, "original_question": "What is 2+2?", "paraphrased_question": "What is 2+2?",
             "gen_judge_model": "gen-judge", "resp_model": "gen-model"},
        ]).to_csv(rp.paraphrases_file(), index=False)
        # Only idx=0's paraphrase has ANY verdict cached under strong-judge/v1 --
        # idx=1 was simply never judged, so it's excluded regardless of resp_model.
        _judge(rp, "strong-judge", "v1", [("What is 1+1?", "What is 1+1?", True)])

        llm = _mock_llm()
        _run(monkeypatch, llm=llm, paraphrase_judge_model="strong-judge", paraphrase_judge_prompt_name="v1")
        (sent_prompts,), _ = llm.batch.call_args
        assert len(sent_prompts) == 1

    def test_unvalidated_pool_is_empty_selection_not_an_error(self, tmp_path, monkeypatch, rp):
        """No verdicts at all under (judge_model, prompt_name) -> nothing
        selected (never falls back to judging on demand -- this is a
        cache-only lookup)."""
        pd.DataFrame([
            {"idx": 0, "original_question": "What is 1+1?", "paraphrased_question": "What is 1+1?",
             "gen_judge_model": "gen-judge", "resp_model": "gen-model"},
        ]).to_csv(rp.paraphrases_file(), index=False)

        llm = _mock_llm()
        _run(monkeypatch, llm=llm, paraphrase_judge_model="strong-judge", paraphrase_judge_prompt_name="v1")
        llm.batch.assert_not_called()

    def test_no_prompt_name_matches_any_prompt_for_that_judge(self, tmp_path, monkeypatch, rp):
        """paraphrase_judge_prompt_name is optional -- omitting it means
        'accepted by this judge_model under ANY prompt', not an error."""
        pd.DataFrame([
            {"idx": 0, "original_question": "What is 1+1?", "paraphrased_question": "What is 1+1?",
             "gen_judge_model": "gen-judge", "resp_model": "gen-model"},
        ]).to_csv(rp.paraphrases_file(), index=False)
        _judge(rp, "strong-judge", "v1", [("What is 1+1?", "What is 1+1?", True)])

        llm = _mock_llm()
        _run(monkeypatch, llm=llm, paraphrase_judge_model="strong-judge")  # no prompt_name given
        (sent_prompts,), _ = llm.batch.call_args
        assert len(sent_prompts) == 1

    def test_recorded_in_run_config_with_selected_pool_size(self, tmp_path, monkeypatch, rp):
        import json
        pd.DataFrame([
            {"idx": 0, "original_question": "What is 1+1?", "paraphrased_question": "What is 1+1?",
             "gen_judge_model": "gen-judge", "resp_model": "gen-model"},
            {"idx": 1, "original_question": "What is 2+2?", "paraphrased_question": "What is 2+2?",
             "gen_judge_model": "gen-judge", "resp_model": "gen-model"},
        ]).to_csv(rp.paraphrases_file(), index=False)
        _judge(rp, "strong-judge", "v1", [("What is 1+1?", "What is 1+1?", True)])

        _run(monkeypatch, paraphrase_judge_model="strong-judge", paraphrase_judge_prompt_name="v1")
        cfg = json.loads((rp.run_dir / "config.answer_generation.json").read_text())
        assert cfg["paraphrase_judge_model"] == "strong-judge"
        assert cfg["paraphrase_judge_prompt_name"] == "v1"
        assert cfg["paraphrase_pool_size"] == 1
        assert "created_at" in cfg  # save_pipeline_config's existing timestamp

    def test_resp_model_further_restricts_selected_pool(self, tmp_path, monkeypatch, rp):
        pd.DataFrame([
            {"idx": 0, "original_question": "What is 1+1?", "paraphrased_question": "What is 1+1?",
             "gen_judge_model": "gen-judge", "resp_model": "model-a"},
            {"idx": 1, "original_question": "What is 2+2?", "paraphrased_question": "What is 2+2?",
             "gen_judge_model": "gen-judge", "resp_model": "model-b"},
        ]).to_csv(rp.paraphrases_file(), index=False)
        _judge(rp, "strong-judge", "v1", [
            ("What is 1+1?", "What is 1+1?", True),
            ("What is 2+2?", "What is 2+2?", True),
        ])

        llm = _mock_llm()
        _run(
            monkeypatch, llm=llm, paraphrase_judge_model="strong-judge",
            paraphrase_judge_prompt_name="v1", paraphrase_resp_model="model-a",
        )
        (sent_prompts,), _ = llm.batch.call_args
        assert len(sent_prompts) == 1

    def test_require_different_generation_model_false_reaches_self_written_rows(self, tmp_path, monkeypatch, rp):
        """A paraphrase WRITTEN by strong-judge itself (resp_model ==
        "strong-judge", the same model we're asking to judge) is excluded by
        default -- a judge validating its own writing isn't independent
        validation. Opting in with paraphrase_require_different_generation_model
        =False reaches it anyway."""
        pd.DataFrame([
            {"idx": 0, "original_question": "What is 1+1?", "paraphrased_question": "What is 1+1?",
             "resp_model": "strong-judge"},
        ]).to_csv(rp.paraphrases_file(), index=False)
        _judge(rp, "strong-judge", "v1", [("What is 1+1?", "What is 1+1?", True)])

        llm = _mock_llm()
        _run(monkeypatch, llm=llm, paraphrase_judge_model="strong-judge", paraphrase_judge_prompt_name="v1")
        llm.batch.assert_not_called()  # default: excluded, nothing to answer

        llm2 = _mock_llm()
        _run(
            monkeypatch, llm=llm2, paraphrase_judge_model="strong-judge",
            paraphrase_judge_prompt_name="v1", paraphrase_require_different_generation_model=False,
        )
        (sent_prompts,), _ = llm2.batch.call_args
        assert len(sent_prompts) == 1
