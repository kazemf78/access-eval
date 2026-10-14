"""
Unit tests for src/access_eval/core/reconcile.py.

Pure DataFrame logic, no LLM/GPU calls anywhere on this path -- no mocking
needed.
"""

import pandas as pd
import pytest

from access_eval.core.reconcile import (
    reconcile_keyed,
    reconcile_quota,
    EvalTarget,
    AnswerTarget,
)


# ---------------------------------------------------------------------------
# reconcile_keyed
# ---------------------------------------------------------------------------

class TestReconcileKeyedSingleKey:
    def test_reduces_to_existence_check(self):
        """No compute_key_fn -> resolved_from_cache always empty, matching
        answer_generation.py's current single-key `not in done_keys` logic."""
        target = pd.DataFrame({
            "formulation_hash": ["f1", "f1", "f2"],
            "repeat_id": [0, 1, 0],
            "gen_config_hash": ["cfg_a", "cfg_a", "cfg_a"],
        })
        existing = pd.DataFrame({
            "formulation_hash": ["f1"],
            "repeat_id": [0],
            "gen_config_hash": ["cfg_a"],
        })
        result = reconcile_keyed(target, existing, AnswerTarget.row_key)

        assert result.resolved_from_cache == {}
        assert len(result.missing) == 2
        assert set(zip(result.missing["formulation_hash"], result.missing["repeat_id"])) == {
            ("f1", 1), ("f2", 0),
        }
        # to_compute defaults to compute_key_fn == row_key_fn -- no dedup collapse
        assert len(result.to_compute) == 2

    def test_gap_in_repeat_ids_identifies_exact_missing_repeat(self):
        """Repeats {0,2,3} exist, target is {0,1,2,3} -- only repeat_id=1 is missing."""
        target = AnswerTarget(
            model="gpt-test", gen_config_hash="cfg_a", repeat_ids=(0, 1, 2, 3),
        ).expand(["f1"])
        existing = pd.DataFrame({
            "formulation_hash": ["f1", "f1", "f1"],
            "repeat_id": [0, 2, 3],
            "gen_config_hash": ["cfg_a", "cfg_a", "cfg_a"],
        })
        result = reconcile_keyed(target, existing, AnswerTarget.row_key)
        assert list(result.missing["repeat_id"]) == [1]

    def test_config_change_treated_as_fully_missing(self):
        """Existing rows under a different gen_config_hash never satisfy the target."""
        target = AnswerTarget(
            model="gpt-test", gen_config_hash="cfg_temp0", repeat_ids=(0, 1),
        ).expand(["f1"])
        existing = pd.DataFrame({
            "formulation_hash": ["f1", "f1"],
            "repeat_id": [0, 1],
            "gen_config_hash": ["cfg_temp07", "cfg_temp07"],
        })
        result = reconcile_keyed(target, existing, AnswerTarget.row_key)
        assert len(result.missing) == 2  # neither existing row matches this config


class TestReconcileKeyedTwoKey:
    def test_cache_hit_different_repeat_id(self):
        """Reproduces the evaluation.py bug scenario: repeat_0's answer text
        already judged; repeat_1 produces IDENTICAL text. Row-identity key
        (with model/gen_config_hash/repeat_id) says repeat_1 is missing;
        compute key (text-only) says it can be resolved from cache -- zero
        new judge calls."""
        target = pd.DataFrame({
            "formulation_hash": ["f1", "f1"],
            "answer": ["Paris", "Paris"],
            "model": ["gpt-4.1", "gpt-4.1"],
            "gen_config_hash": ["cfg_gen_a", "cfg_gen_a"],
            "repeat_id": [0, 1],
            "eval_config_hash": ["cfg_a", "cfg_a"],
        })
        existing = pd.DataFrame({
            "formulation_hash": ["f1"],
            "answer": ["Paris"],
            "model": ["gpt-4.1"],
            "gen_config_hash": ["cfg_gen_a"],
            "repeat_id": [0],
            "eval_config_hash": ["cfg_a"],
            "verdict": ["correct"],
        })
        result = reconcile_keyed(target, existing, EvalTarget.row_key, EvalTarget.compute_key)

        assert len(result.missing) == 1  # repeat_1's row doesn't exist yet
        assert list(result.missing["repeat_id"]) == [1]
        assert len(result.resolved_from_cache) == 1  # but its TEXT is cached
        cached_row = list(result.resolved_from_cache.values())[0]
        assert cached_row["verdict"] == "correct"
        assert result.to_compute.empty  # zero new judge calls needed

    def test_cross_model_identical_text_gets_its_own_row_not_treated_as_already_done(self):
        """A DIFFERENT model producing byte-identical answer text for the
        same formulation/repeat must still get its own materialized row --
        the text-only compute_key correctly reuses the judge verdict, but
        row_key (which includes model/gen_config_hash) must NOT let one
        model's already-materialized row make another model's row look
        already covered."""
        target = pd.DataFrame({
            "formulation_hash": ["f1", "f1"],
            "answer": ["Paris", "Paris"],
            "model": ["gpt-4.1", "qwen3-8b"],
            "gen_config_hash": ["cfg_gen_a", "cfg_gen_b"],
            "repeat_id": [0, 0],
            "eval_config_hash": ["cfg_a", "cfg_a"],
        })
        existing = pd.DataFrame({
            "formulation_hash": ["f1"],
            "answer": ["Paris"],
            "model": ["gpt-4.1"],
            "gen_config_hash": ["cfg_gen_a"],
            "repeat_id": [0],
            "eval_config_hash": ["cfg_a"],
            "verdict": ["correct"],
        })
        result = reconcile_keyed(target, existing, EvalTarget.row_key, EvalTarget.compute_key)

        # qwen3-8b's row is NOT already covered by gpt-4.1's row.
        assert len(result.missing) == 1
        assert result.missing.iloc[0]["model"] == "qwen3-8b"
        # But its identical TEXT resolves from the cache -- no new judge call.
        assert len(result.resolved_from_cache) == 1
        assert result.to_compute.empty

    def test_genuinely_new_text_needs_computing(self):
        target = pd.DataFrame({
            "formulation_hash": ["f1"],
            "answer": ["London"],
            "model": ["gpt-4.1"],
            "gen_config_hash": ["cfg_gen_a"],
            "repeat_id": [0],
            "eval_config_hash": ["cfg_a"],
        })
        existing = pd.DataFrame({
            "formulation_hash": ["f1"],
            "answer": ["Paris"],
            "model": ["gpt-4.1"],
            "gen_config_hash": ["cfg_gen_a"],
            "repeat_id": [0],
            "eval_config_hash": ["cfg_a"],
            "verdict": ["correct"],
        })
        result = reconcile_keyed(target, existing, EvalTarget.row_key, EvalTarget.compute_key)
        assert result.resolved_from_cache == {}
        assert len(result.to_compute) == 1

    def test_dedup_fans_out_to_multiple_targets(self):
        """Two target rows share the same compute key (different repeat_ids,
        identical text, neither previously graded) -- to_compute has exactly
        one row, compute_key_to_targets maps it back to both."""
        target = pd.DataFrame({
            "formulation_hash": ["f1", "f1"],
            "answer": ["Paris", "Paris"],
            "model": ["gpt-4.1", "gpt-4.1"],
            "gen_config_hash": ["cfg_gen_a", "cfg_gen_a"],
            "repeat_id": [0, 1],
            "eval_config_hash": ["cfg_a", "cfg_a"],
        })
        existing = pd.DataFrame(columns=["formulation_hash", "answer", "repeat_id", "eval_config_hash"])
        result = reconcile_keyed(target, existing, EvalTarget.row_key, EvalTarget.compute_key)

        assert len(result.to_compute) == 1
        [(key, labels)] = result.compute_key_to_targets.items()
        assert key == ("f1", "Paris", "cfg_a")
        assert len(labels) == 2


class TestReconcileKeyedEdgeCases:
    def test_empty_target(self):
        target = pd.DataFrame(columns=["formulation_hash", "repeat_id", "gen_config_hash"])
        existing = pd.DataFrame({
            "formulation_hash": ["f1"], "repeat_id": [0], "gen_config_hash": ["cfg_a"],
        })
        result = reconcile_keyed(target, existing, AnswerTarget.row_key)
        assert result.missing.empty
        assert result.to_compute.empty
        assert result.resolved_from_cache == {}
        assert result.compute_key_to_targets == {}

    def test_empty_existing(self):
        target = AnswerTarget(
            model="gpt-test", gen_config_hash="cfg_a", repeat_ids=(0, 1),
        ).expand(["f1", "f2"])
        existing = pd.DataFrame(columns=["formulation_hash", "repeat_id", "gen_config_hash"])
        result = reconcile_keyed(target, existing, AnswerTarget.row_key)
        assert len(result.missing) == 4
        assert len(result.to_compute) == 4


# ---------------------------------------------------------------------------
# reconcile_quota
# ---------------------------------------------------------------------------

class TestReconcileQuota:
    def test_basic_deficit(self):
        existing = pd.DataFrame({"idx": [1, 1, 1, 2, 2]})
        deficit = reconcile_quota(existing, group_cols=["idx"], target_count=5)
        assert deficit.loc[1] == 2
        assert deficit.loc[2] == 3

    def test_never_negative(self):
        existing = pd.DataFrame({"idx": [1, 1, 1, 1, 1, 1]})
        deficit = reconcile_quota(existing, group_cols=["idx"], target_count=3)
        assert deficit.loc[1] == 0

    def test_valid_mask_excludes_rows(self):
        """A rejected candidate doesn't count toward the accepted quota."""
        existing = pd.DataFrame({
            "idx": [1, 1, 1],
            "llm_equiv": [True, True, False],
        })
        deficit = reconcile_quota(
            existing, group_cols=["idx"], target_count=3,
            valid_mask=existing["llm_equiv"],
        )
        assert deficit.loc[1] == 1  # only 2 of 3 rows are valid

    def test_stricter_judge_reopens_a_previously_satisfied_group(self):
        """Same idea as above, phrased as the paraphrase-refiltering scenario:
        a judge that rejects some previously-accepted candidates drops the
        accepted count and reopens the deficit."""
        existing = pd.DataFrame({
            "idx": [7, 7, 7, 7],
            "llm_equiv": [True, True, True, False],  # judge rejected the 4th
        })
        deficit = reconcile_quota(
            existing, group_cols=["idx"], target_count=4,
            valid_mask=existing["llm_equiv"],
        )
        assert deficit.loc[7] == 1

    def test_group_absent_entirely_needs_reindex_by_caller(self):
        """Documents the caller-responsibility noted in the docstring: a group
        with zero existing rows isn't in the returned Series at all -- the
        caller must reindex against its own full expected-group list and
        fillna(target_count) to get that group's deficit."""
        existing = pd.DataFrame({"idx": [1, 1]})
        deficit = reconcile_quota(existing, group_cols=["idx"], target_count=5)
        assert 2 not in deficit.index

        full_scope_idx = [1, 2]  # caller's full expected-group list
        reconciled = deficit.reindex(full_scope_idx).fillna(5).astype(int)
        assert reconciled.loc[1] == 3  # 2 existing, target 5
        assert reconciled.loc[2] == 5  # absent entirely -> full target

    def test_empty_existing_returns_empty_series(self):
        existing = pd.DataFrame(columns=["idx"])
        deficit = reconcile_quota(existing, group_cols=["idx"], target_count=5)
        assert deficit.empty
