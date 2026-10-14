# src/access_eval/core/reconcile.py
"""
Generic "target state vs. existing artifacts -> missing work" reconciliation
primitives, shared across pipeline stages instead of each stage hand-rolling
its own resume/dedup logic.

Two independent shapes cover every resume pattern currently in the pipeline:

  - reconcile_keyed:  "is this exact identity already materialized?" -- for
    targets that are exactly enumerable in advance (e.g. answer generation:
    every (formulation, model, gen_config, repeat_id) combination is known
    before generating anything). Supports an optional SECOND, coarser key
    for cache reuse -- e.g. evaluation's judge-verdict cache, where a row's
    own identity needs repeat_id but whether a judge CALL is needed does not
    (identical answer text only needs judging once, regardless of repeat).

  - reconcile_quota:  "how many more do I need?" -- for targets that are a
    COUNT, not enumerable in advance (e.g. paraphrase generation: a
    paraphrase's identity is a content hash unknowable before it's
    generated, so "N accepted candidates for this idx" is a count problem).

Both are pure, stage-agnostic functions operating on DataFrames + caller-
supplied key functions -- this module never imports from `pipeline/` and
never reads/writes files itself. Callers are responsible for passing
already-normalized input (via `io.compat.normalize_generation_df`/
`normalize_grading_df`) -- which normalizer applies is stage-specific
knowledge this module deliberately doesn't have.

The `EvalTarget`/`AnswerTarget` dataclasses below are thin, stage-specific
key/target-set builders on top of the two primitives -- so call sites don't
hand-roll lambdas inconsistently. They know about specific column names;
`reconcile_keyed`/`reconcile_quota` themselves do not.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple

import pandas as pd


# =============================================================================
# reconcile_keyed
# =============================================================================

@dataclass(frozen=True)
class KeyedReconcileResult:
    """
    missing:                target rows without their own materialized row
                             in `existing` yet (preserves target's row order).
    resolved_from_cache:     {target_row_label: existing_row_dict} for missing
                             rows whose `compute_key_fn` value already has a
                             match in `existing` under some OTHER row-identity
                             (e.g. a different repeat_id) -- these can be
                             materialized by copying, no new computation.
    to_compute:              the rest of `missing`, deduplicated by
                             `compute_key_fn` (keep-first, order-preserving) --
                             what actually needs computing.
    compute_key_to_targets:  {compute_key: [target_row_label, ...]} over the
                             (non-deduplicated) rows needing computation, so
                             one computed result can be fanned out to every
                             target row sharing that compute key.
    """
    missing: pd.DataFrame
    resolved_from_cache: Dict[Any, dict]
    to_compute: pd.DataFrame
    compute_key_to_targets: Dict[Hashable, List[Any]]


def reconcile_keyed(
    target: pd.DataFrame,
    existing: pd.DataFrame,
    row_key_fn: Callable[[pd.Series], Hashable],
    compute_key_fn: Optional[Callable[[pd.Series], Hashable]] = None,
) -> KeyedReconcileResult:
    """
    target:          rows representing the desired end state.
    existing:        rows already materialized (may contain rows outside the
                      target's scope entirely, e.g. other configs -- these
                      simply never match any target row's key and are ignored).
    row_key_fn:       row -> hashable key deciding "does this exact row exist
                      in `existing` yet." Applied to both `target` and
                      `existing` rows, so both must have the columns it reads.
    compute_key_fn:   row -> hashable key deciding "has the underlying
                      computation for this row already happened, even under a
                      different row identity." Defaults to `row_key_fn` when
                      omitted, which collapses `resolved_from_cache` to always
                      be empty -- a plain existence check with no cache reuse.
    """
    empty_result = KeyedReconcileResult(
        missing=target.iloc[0:0],
        resolved_from_cache={},
        to_compute=target.iloc[0:0],
        compute_key_to_targets={},
    )
    if target.empty:
        return empty_result
    if compute_key_fn is None:
        compute_key_fn = row_key_fn

    if existing.empty:
        missing = target
        existing_compute_index: Dict[Hashable, dict] = {}
    else:
        existing_row_keys = {row_key_fn(row) for _, row in existing.iterrows()}
        missing_mask = ~target.apply(lambda r: row_key_fn(r) in existing_row_keys, axis=1)
        missing = target[missing_mask]

        existing_compute_index = {}
        for _, row in existing.iterrows():
            key = compute_key_fn(row)
            if key not in existing_compute_index:
                existing_compute_index[key] = row.to_dict()

    if missing.empty:
        return KeyedReconcileResult(
            missing=missing, resolved_from_cache={}, to_compute=missing,
            compute_key_to_targets={},
        )

    resolved_from_cache: Dict[Any, dict] = {}
    needs_compute_labels = []
    for label, row in missing.iterrows():
        key = compute_key_fn(row)
        if key in existing_compute_index:
            resolved_from_cache[label] = existing_compute_index[key]
        else:
            needs_compute_labels.append(label)

    needs_compute = missing.loc[needs_compute_labels]

    compute_key_to_targets: Dict[Hashable, List[Any]] = {}
    for label, row in needs_compute.iterrows():
        compute_key_to_targets.setdefault(compute_key_fn(row), []).append(label)

    to_compute = needs_compute[
        ~needs_compute.apply(compute_key_fn, axis=1).duplicated(keep="first")
    ]

    return KeyedReconcileResult(
        missing=missing,
        resolved_from_cache=resolved_from_cache,
        to_compute=to_compute,
        compute_key_to_targets=compute_key_to_targets,
    )


# =============================================================================
# reconcile_quota
# =============================================================================

def reconcile_quota(
    existing: pd.DataFrame,
    group_cols: List[str],
    target_count: int,
    valid_mask: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Per-group deficit: how many more rows each group in `group_cols` needs to
    reach `target_count`. A group with zero valid rows (including groups
    absent from `existing` entirely, if the caller separately enumerates all
    expected groups -- see note below) gets the full `target_count`. Never
    returns a negative deficit -- this primitive only ever says "need more,"
    never "have too many" (matches the pipeline's append-only philosophy:
    nothing gets deleted because a quota was exceeded).

    Note: groups that don't appear in `existing` AT ALL are not represented in
    the returned Series (there's nothing to group by) -- callers that need a
    deficit for idx values with zero candidates yet should reindex the result
    against their own full expected-group list and fillna(target_count).
    """
    scoped = existing if valid_mask is None else existing[valid_mask]
    if scoped.empty:
        return pd.Series(dtype=int)
    counts = scoped.groupby(group_cols).size()
    deficit = (target_count - counts).clip(lower=0).astype(int)
    return deficit


# =============================================================================
# Stage-specific target-set builders
# =============================================================================

@dataclass(frozen=True)
class EvalTarget:
    """Key shapes for evaluation.py's row-identity vs. judge-verdict-cache
    split (see pipeline/evaluation.py's _resolve_resume_state).

    row_key includes `model`/`gen_config_hash` -- the evaluated answers span
    every answer-generating model and config, and two different ones can
    legitimately produce identical answer text for the same question/repeat;
    without these in the key, one model's/config's materialized row would
    wrongly look like it already covers a different model's/config's row.
    compute_key deliberately omits them -- identical answer TEXT only needs
    judging once, regardless of which model/config produced it."""

    @staticmethod
    def row_key(row: pd.Series) -> Hashable:
        # `answer` stays in the key (as it was pre-fix) alongside the two
        # new disambiguating fields, not replaced by them: (formulation,
        # model, gen_config_hash, repeat_id) *should* uniquely determine a
        # source answers_df row and hence its text, but keeping `answer`
        # here too is defense-in-depth against ever silently treating two
        # rows with genuinely different answer text as "the same row
        # already materialized" if that invariant is ever violated (e.g. a
        # forced regeneration under an unchanged identity tuple).
        return (
            row["formulation_hash"], row["answer"], row["model"], row["gen_config_hash"],
            row["repeat_id"], row["eval_config_hash"],
        )

    @staticmethod
    def compute_key(row: pd.Series) -> Hashable:
        return (row["formulation_hash"], row["answer"], row["eval_config_hash"])


@dataclass(frozen=True)
class AnswerTarget:
    """Target-row builder for answer_generation.py's resume: the target is
    exactly enumerable (every (formulation_hash, repeat_id) pair under this
    model/gen_config), so this feeds reconcile_keyed, not reconcile_quota."""

    model: str
    gen_config_hash: str
    repeat_ids: Tuple[int, ...]

    def expand(self, formulation_hashes: List[str]) -> pd.DataFrame:
        """One row per (formulation_hash, repeat_id) pair."""
        rows = [
            {
                "formulation_hash": fh,
                "model": self.model,
                "gen_config_hash": self.gen_config_hash,
                "repeat_id": rid,
            }
            for fh in formulation_hashes
            for rid in self.repeat_ids
        ]
        return pd.DataFrame(rows, columns=["formulation_hash", "model", "gen_config_hash", "repeat_id"])

    @staticmethod
    def row_key(row: pd.Series) -> Hashable:
        return (row["formulation_hash"], row["repeat_id"], row["gen_config_hash"])
