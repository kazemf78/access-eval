"""
Golden-fixture regression test for evaluation.py's `_resolve_resume_state`,
covering its refactor into a thin wrapper around
`core.reconcile.reconcile_keyed` (see core/reconcile.py, evaluation.py).

Each fixture's expected values are hand-computed against the documented
contract:
  - row-identity key (formulation_hash, answer, model, gen_config_hash,
    repeat_id, eval_config_hash) decides materialization -- model/
    gen_config_hash are part of it because the combined answers file spans
    every answer-generating model/config, and two different ones can
    legitimately produce identical answer text for the same
    question/repeat.
  - judge-verdict-cache key (formulation_hash, answer, eval_config_hash)
    decides whether a judge call is needed -- text-only, deliberately
    ignoring model/gen_config_hash/repeat_id, so identical answer text only
    ever needs judging once.
  - `text_to_rows` is keyed by the 2-tuple (formulation_hash, answer),
    matching exactly what both call sites (`run_evaluation`'s judge loop,
    `_run_custom_evaluator`) build and look up.

`evaluations_so_far` fixtures use real (original_question, paraphrased_question)
pairs, not a hand-set `formulation_hash` -- `_resolve_resume_state` recomputes
it from those columns (matching a REAL grades CSV's shape, which has no
stored formulation_hash column at all), so a fixture using a literal
"formulation_hash" key would silently mask this exact recomputation step ever
being skipped -- as an earlier version of this file's fixtures did.

Not derived by calling the implementation under test -- a true regression
gate, not a tautology. Pure DataFrame logic -- no LLM calls, no mocking needed.
"""

import pandas as pd

from access_eval.io.identity import paraphrase_content_hash
from access_eval.pipeline.evaluation import _resolve_resume_state

Q1 = ("What is the capital of France?", "What is France's capital?")
Q2 = ("Who wrote Hamlet?", "Who is the author of Hamlet?")
F1 = paraphrase_content_hash(*Q1)
F2 = paraphrase_content_hash(*Q2)


def _answers_df(rows):
    """rows: list of dicts with formulation_hash/answer/model/gen_config_hash/
    repeat_id -- the TARGET side, which already has formulation_hash computed
    (matches what run_evaluation does before calling _resolve_resume_state)."""
    return pd.DataFrame(rows)


def _answer_row(formulation_hash, answer, model="gpt-4.1", gen_config_hash="cfg_gen_a", repeat_id=0):
    return {
        "formulation_hash": formulation_hash, "answer": answer, "model": model,
        "gen_config_hash": gen_config_hash, "repeat_id": repeat_id,
    }


def _prior_eval(orig_para, answer, model="gpt-4.1", gen_config_hash="cfg_gen_a",
                 repeat_id=0, eval_config_hash="cfg_a", **extra):
    """rows: the EXISTING side (a real grades CSV row) -- original_question/
    paraphrased_question, not formulation_hash."""
    o, p = orig_para
    return {
        "original_question": o, "paraphrased_question": p, "answer": answer,
        "model": model, "gen_config_hash": gen_config_hash, "repeat_id": repeat_id,
        "eval_config_hash": eval_config_hash, **extra,
    }


class TestFreshRun:
    def test_no_prior_evaluations_everything_missing_nothing_cached(self):
        answers_df = _answers_df([
            _answer_row(F1, "Paris"),
            _answer_row(F2, "Shakespeare"),
        ])
        missing_df, cache_resolved_rows, needs_judging_df, to_judge_df, text_to_rows = \
            _resolve_resume_state(answers_df, evaluations_so_far=[], eval_config_hash="cfg_a")

        assert len(missing_df) == 2
        assert cache_resolved_rows == []
        assert len(needs_judging_df) == 2
        assert len(to_judge_df) == 2
        assert set(text_to_rows.keys()) == {(F1, "Paris"), (F2, "Shakespeare")}
        assert text_to_rows[(F1, "Paris")] == [0]
        assert text_to_rows[(F2, "Shakespeare")] == [1]


class TestExactResumeMatch:
    def test_already_graded_row_is_not_missing(self):
        answers_df = _answers_df([_answer_row(F1, "Paris")])
        evaluations_so_far = [_prior_eval(Q1, "Paris", verdict="correct")]
        missing_df, cache_resolved_rows, needs_judging_df, to_judge_df, text_to_rows = \
            _resolve_resume_state(answers_df, evaluations_so_far, eval_config_hash="cfg_a")

        assert missing_df.empty
        assert cache_resolved_rows == []
        assert needs_judging_df.empty
        assert to_judge_df.empty
        assert text_to_rows == {}

    def test_different_eval_config_hash_is_not_a_match(self):
        """A row graded under an old prompt/config must not satisfy a new one."""
        answers_df = _answers_df([_answer_row(F1, "Paris")])
        evaluations_so_far = [_prior_eval(Q1, "Paris", eval_config_hash="cfg_old", verdict="correct")]
        missing_df, cache_resolved_rows, needs_judging_df, to_judge_df, text_to_rows = \
            _resolve_resume_state(answers_df, evaluations_so_far, eval_config_hash="cfg_new")

        assert len(missing_df) == 1
        # Also NOT cache-resolved -- the cache key includes eval_config_hash too.
        assert cache_resolved_rows == []
        assert len(to_judge_df) == 1


class TestCacheHitDifferentRepeatId:
    """The bug scenario this whole mechanism exists to prevent: repeat_0's
    answer text already graded; repeat_1 produces IDENTICAL text under the
    same model/config. Row-identity says repeat_1 is missing; the
    judge-verdict cache (text-only) says its TEXT was already judged --
    zero new judge calls, one new row materialized by copying."""

    def test_duplicate_repeat_text_resolved_from_cache_not_rejudged(self):
        answers_df = _answers_df([
            _answer_row(F1, "Paris", repeat_id=0),
            _answer_row(F1, "Paris", repeat_id=1),
        ])
        evaluations_so_far = [_prior_eval(Q1, "Paris", repeat_id=0, verdict="correct", score=1.0)]
        missing_df, cache_resolved_rows, needs_judging_df, to_judge_df, text_to_rows = \
            _resolve_resume_state(answers_df, evaluations_so_far, eval_config_hash="cfg_a")

        # repeat_0 already has its own row -> not missing.
        # repeat_1 doesn't have its own row yet -> missing.
        assert len(missing_df) == 1
        assert missing_df.iloc[0]["repeat_id"] == 1

        # But repeat_1's TEXT was already judged -> resolved from cache, not
        # sent to the judge.
        assert len(cache_resolved_rows) == 1
        resolved = cache_resolved_rows[0]
        assert resolved["answer_idx"] == 1          # answers_df's row label for repeat_1
        assert resolved["repeat_id"] == 1            # stamped with the NEW repeat_id
        assert resolved["model"] == "gpt-4.1"        # stamped from the TARGET row too
        assert resolved["gen_config_hash"] == "cfg_gen_a"
        assert resolved["verdict"] == "correct"      # copied from the cached verdict
        assert resolved["score"] == 1.0

        assert needs_judging_df.empty
        assert to_judge_df.empty
        assert text_to_rows == {}

    def test_three_repeats_same_text_one_already_graded_two_resolved_from_cache(self):
        answers_df = _answers_df([
            _answer_row(F1, "Paris", repeat_id=0),
            _answer_row(F1, "Paris", repeat_id=1),
            _answer_row(F1, "Paris", repeat_id=2),
        ])
        evaluations_so_far = [_prior_eval(Q1, "Paris", repeat_id=0, verdict="correct")]
        missing_df, cache_resolved_rows, needs_judging_df, to_judge_df, text_to_rows = \
            _resolve_resume_state(answers_df, evaluations_so_far, eval_config_hash="cfg_a")

        assert len(missing_df) == 2
        assert len(cache_resolved_rows) == 2
        assert {r["repeat_id"] for r in cache_resolved_rows} == {1, 2}
        assert to_judge_df.empty  # zero redundant judge calls


class TestCrossModelIdenticalAnswer:
    """The reviewer-flagged bug: TWO DIFFERENT answer-generating models
    produce byte-identical answer text for the same formulation/repeat.
    Row-identity (with model/gen_config_hash) must NOT let one model's
    already-materialized row make the other model's row look already
    covered -- but the judge-verdict cache (text-only) SHOULD still let the
    second model's row be resolved without a redundant judge call."""

    def test_second_model_gets_its_own_row_not_swallowed_by_first(self):
        answers_df = _answers_df([
            _answer_row(F1, "Paris", model="gpt-4.1", gen_config_hash="cfg_gen_gpt"),
            _answer_row(F1, "Paris", model="qwen3-8b", gen_config_hash="cfg_gen_qwen"),
        ])
        evaluations_so_far = [
            _prior_eval(Q1, "Paris", model="gpt-4.1", gen_config_hash="cfg_gen_gpt", verdict="correct"),
        ]
        missing_df, cache_resolved_rows, needs_judging_df, to_judge_df, text_to_rows = \
            _resolve_resume_state(answers_df, evaluations_so_far, eval_config_hash="cfg_a")

        # qwen3-8b's row is NOT already covered by gpt-4.1's materialized row.
        assert len(missing_df) == 1
        assert missing_df.iloc[0]["model"] == "qwen3-8b"

        # But its identical TEXT resolves from the judge-verdict cache --
        # zero new judge calls -- and the materialized row correctly carries
        # QWEN's identity, not GPT-4.1's (whose row the cache entry came from).
        assert len(cache_resolved_rows) == 1
        resolved = cache_resolved_rows[0]
        assert resolved["model"] == "qwen3-8b"
        assert resolved["gen_config_hash"] == "cfg_gen_qwen"
        assert resolved["verdict"] == "correct"
        assert to_judge_df.empty

    def test_same_model_different_gen_config_hash_also_gets_its_own_row(self):
        """Two runs of the SAME model under different generation configs
        (e.g. temperature 0.0 vs 0.7) both accumulated in the combined
        answers file -- same disambiguation requirement as cross-model."""
        answers_df = _answers_df([
            _answer_row(F1, "Paris", model="gpt-4.1", gen_config_hash="cfg_temp00"),
            _answer_row(F1, "Paris", model="gpt-4.1", gen_config_hash="cfg_temp07"),
        ])
        evaluations_so_far = [
            _prior_eval(Q1, "Paris", model="gpt-4.1", gen_config_hash="cfg_temp00", verdict="correct"),
        ]
        missing_df, cache_resolved_rows, needs_judging_df, to_judge_df, text_to_rows = \
            _resolve_resume_state(answers_df, evaluations_so_far, eval_config_hash="cfg_a")

        assert len(missing_df) == 1
        assert missing_df.iloc[0]["gen_config_hash"] == "cfg_temp07"
        assert len(cache_resolved_rows) == 1
        assert cache_resolved_rows[0]["gen_config_hash"] == "cfg_temp07"


class TestMixedLegacyAndNewConfig:
    def test_legacy_sentinel_rows_coexist_with_new_config_rows(self):
        """Rows normalized with the legacy sentinel (pre-dating eval_config_hash
        tracking) must not be mistaken for matches under a real config hash,
        and vice versa -- each is its own bucket."""
        answers_df = _answers_df([
            _answer_row(F1, "Paris"),      # will match a legacy-config prior eval
            _answer_row(F2, "Berlin"),     # genuinely new under cfg_new
        ])
        evaluations_so_far = [_prior_eval(Q1, "Paris", eval_config_hash="legacy", verdict="correct")]

        # Resuming under the LEGACY config: F1/repeat_0 matches, F2 is missing.
        missing_df, cache_resolved_rows, needs_judging_df, to_judge_df, text_to_rows = \
            _resolve_resume_state(answers_df, evaluations_so_far, eval_config_hash="legacy")
        assert len(missing_df) == 1
        assert missing_df.iloc[0]["formulation_hash"] == F2
        assert cache_resolved_rows == []  # different text (Berlin), not cache-eligible

        # Resuming under a NEW config: neither row matches the legacy-config prior eval.
        missing_df2, cache_resolved_rows2, needs_judging_df2, to_judge_df2, text_to_rows2 = \
            _resolve_resume_state(answers_df, evaluations_so_far, eval_config_hash="cfg_new")
        assert len(missing_df2) == 2
        assert cache_resolved_rows2 == []
        assert len(to_judge_df2) == 2


class TestGenuinelyNewTextNeedsJudging:
    def test_never_before_seen_text_goes_to_to_judge_df(self):
        answers_df = _answers_df([_answer_row(F1, "London")])
        evaluations_so_far = [_prior_eval(Q1, "Paris", verdict="correct")]
        missing_df, cache_resolved_rows, needs_judging_df, to_judge_df, text_to_rows = \
            _resolve_resume_state(answers_df, evaluations_so_far, eval_config_hash="cfg_a")

        assert len(missing_df) == 1
        assert cache_resolved_rows == []
        assert len(needs_judging_df) == 1
        assert len(to_judge_df) == 1
        assert text_to_rows[(F1, "London")] == [0]
