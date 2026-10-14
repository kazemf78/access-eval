"""
Backward-compatible read-time adapters. Old artifacts (run_artifacts/,
variability_experiment_outputs/, formulation_comfort_outputs/) predate the
repeat_id/gen_config_hash/eval_config_hash columns the new content-keyed
resume logic needs. These functions fill legacy defaults IN MEMORY ONLY --
nothing on disk is ever rewritten, so old files keep loading unmodified
anywhere else (including already-published analysis notebooks).

Legacy rows get a fixed, non-hex sentinel ("legacy") rather than a
best-effort reconstructed hash: reverse-engineering what config produced an
old row is unreliable, and a wrong guess risks *silently* reusing it under
the wrong config -- worse than the positional-resume bug this replaces. A
fresh run's real hash can never collide with "legacy", so at worst it
regenerates once rather than risking silent contamination.
"""
import pandas as pd

from access_eval.io.identity import HASH_PREFIX

LEGACY_CONFIG_SENTINEL = "legacy"

assert not LEGACY_CONFIG_SENTINEL.startswith(HASH_PREFIX), (
    "legacy sentinel must never collide with a real cfg_<hash> value"
)


def normalize_generation_df(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing repeat_id/gen_config_hash with legacy defaults. Returns a
    copy; never mutates `df` or writes to disk."""
    df = df.copy()
    if "repeat_id" not in df.columns:
        df["repeat_id"] = 0
    else:
        df["repeat_id"] = df["repeat_id"].fillna(0).astype(int)

    if "gen_config_hash" not in df.columns:
        df["gen_config_hash"] = LEGACY_CONFIG_SENTINEL
    else:
        df["gen_config_hash"] = df["gen_config_hash"].fillna(LEGACY_CONFIG_SENTINEL)
    return df


def normalize_grading_df(df: pd.DataFrame) -> pd.DataFrame:
    """Same idea as normalize_generation_df, for the grading side's
    eval_config_hash column -- plus repeat_id, needed for the same reason as
    on the generation side: grading rows need a per-repeat identity too (see
    pipeline/evaluation.py's row-identity-vs-judge-cache-key split).

    Also backfills gen_config_hash -- the ANSWER's own generation config,
    copied onto its grade row at materialization time (a genuinely new
    column; legacy grades rows never had it, hence the same sentinel here as
    on the generation side) -- needed because EvalTarget.row_key includes it
    (alongside `model`) to disambiguate rows across the several
    answer-generating models/configs a combined answers file can span."""
    df = df.copy()
    if "repeat_id" not in df.columns:
        df["repeat_id"] = 0
    else:
        df["repeat_id"] = df["repeat_id"].fillna(0).astype(int)

    if "eval_config_hash" not in df.columns:
        df["eval_config_hash"] = LEGACY_CONFIG_SENTINEL
    else:
        df["eval_config_hash"] = df["eval_config_hash"].fillna(LEGACY_CONFIG_SENTINEL)

    if "gen_config_hash" not in df.columns:
        df["gen_config_hash"] = LEGACY_CONFIG_SENTINEL
    else:
        df["gen_config_hash"] = df["gen_config_hash"].fillna(LEGACY_CONFIG_SENTINEL)
    return df
