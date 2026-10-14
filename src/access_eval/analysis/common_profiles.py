# src/access_eval/analysis/common_profiles.py
"""
N-way generalization of profile_comparison.build_paired_profiles()'s Filter
1-3 pipeline -- the reusable "which questions/paraphrases are comparable,
and what did each model/condition say on them" substrate for 2 OR MORE
models/conditions at once, so a future metric is a cheap pandas op on an
already-filtered, cached table instead of another full pass over raw grades.

Covers all three shapes the project's interventions actually take, via one
mechanism (agg["model"] tagging -- same convention profile_ops.
load_qa_condition_pair and build_prompt_cot_comparison.py already use):
  - two arbitrary models                  ["gpt-4.1", "gpt-4.1-mini"]
  - a model vs an intervention condition  ["Qwen/Qwen3-8B__regular", "Qwen/Qwen3-8B__GEPA"]
  - a whole model-family SET              ["Qwen/Qwen3-0.6B", ..., "Qwen/Qwen3-32B"]

Deliberately does NOT compute build_paired_profiles()'s Layer I-III pairwise
metrics (tv_dist, kappa, jaccard, ...) -- those don't have one canonical
N-way generalization, and re-deriving them here would duplicate that
function's logic for the one case (N==2) where it already has a clearly
right answer. For a 2-model set, prefer build_paired_profiles() itself if
those Layer I-III metrics are what you actually need; reach for this module
when you want the filtered substrate for a SET (or a metric of your own
design) instead.

Pure additive extension: does not modify profile_comparison.py.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from access_eval.analysis.profile_comparison import VERDICTS


def _select_common_output(
    profiles: pd.DataFrame,
    funnel: pd.DataFrame,
    long_df: pd.DataFrame,
    return_diagnostics: bool,
    return_long: bool,
) -> pd.DataFrame | tuple[pd.DataFrame, ...]:
    out = (profiles,)
    if return_diagnostics:
        out += (funnel,)
    if return_long:
        out += (long_df,)
    return out if len(out) > 1 else profiles


def build_common_profiles(
    agg: pd.DataFrame,
    models: list[str],
    min_agreement_fraction: float = 0.5,
    return_diagnostics: bool = False,
    return_long: bool = False,
    cache_path: str | Path | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, ...]:
    """
    Per-question comparability substrate for `models` (2 or more), computed
    only on paraphrase texts EVERY model in the set answered.

    `agg` is preprocess_grades()'s or preprocess_math_answers()'s
    (unfiltered) output. Filters, N-way generalized from
    build_paired_profiles():
      Filter 1 - a graded row survives if n_agree > n_judges *
                 min_agreement_fraction AND the GPT-family judges
                 unanimously agree with each other (unchanged, per-row,
                 per-model -- doesn't care how many other models exist).
      Filter 2 - a question is comparable only if its ORIGINAL question's
                 row survived Filter 1 for EVERY model in `models`.
      Filter 3 - restrict to the paraphrase TEXTS that survived Filter 1
                 for EVERY model (N-way intersection, not just 2).

    Returns a DataFrame indexed by idx with n_para (the shared-across-every-
    model count) and, per model tag m in `models`: orig_label_{m},
    p{verdict}_{m}, correct_frac_{m} (= pcorrect_{m}). Column names use the
    literal tag string (not positional _a/_b/_c...), so they read directly
    even for a long model-family set.

    If return_diagnostics=True, also returns a per-idx funnel DataFrame:
    has_orig_{m}/n_para_total_{m}/n_para_filter1_{m} per model,
    passes_filter2, n_para_shared, included -- feed it to
    summarize_common_funnel().

    If return_long=True, also returns a long-format DataFrame (idx,
    paraphrased_question, model, verdict), one row per (shared paraphrase,
    model) for every included question -- the flexible substrate: pivot it
    however a specific downstream metric needs (wide per-model columns for
    an N-way agreement measure, or slice to exactly 2 models and feed that
    into profile_comparison's own Layer II/III math), without re-touching
    the raw grades or re-deriving Filter 1-3.

    cache_path: get one from RunPaths.common_profiles_cache_file(subset,
    models, min_agreement_fraction, collapse_not_attempted). Fingerprinted
    on (agg.attrs["source_fingerprint"], tuple(sorted(models)),
    min_agreement_fraction) -- `agg` MUST come from preprocess_grades()/
    preprocess_math_answers() (source_fingerprint lives in .attrs), and
    membership (not order) of `models` is what's hashed, matching
    RunPaths.common_profiles_cache_file()'s own sorted-tag hash.
    """
    if cache_path is not None:
        source_fp = agg.attrs.get("source_fingerprint")
        if source_fp is None:
            raise ValueError(
                "build_common_profiles(cache_path=...) requires `agg` to carry "
                "a source_fingerprint (agg.attrs) so the cache can be "
                "invalidated correctly -- build `agg` via preprocess_grades() "
                "or preprocess_math_answers(), which set this automatically."
            )
        fingerprint = (source_fp, tuple(sorted(models)), min_agreement_fraction)
        if Path(cache_path).exists():
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("fingerprint") == fingerprint:
                return _select_common_output(
                    cached["profiles"], cached["funnel"], cached["long"],
                    return_diagnostics, return_long,
                )

    sub = agg[agg["model"].isin(models)].copy()
    sub["is_orig"] = sub["original_question"] == sub["paraphrased_question"]
    sub["passes_filter1"] = (
        sub["n_agree"] > sub["n_judges"] * min_agreement_fraction
    ) & sub["gpt_agree"]

    funnel_rows, profile_rows, long_rows = [], [], []
    for idx, g in sub.groupby("idx"):
        g_orig = g[g["is_orig"]]
        g_para = g[~g["is_orig"]]

        has_orig, orig_label, texts_ok = {}, {}, {}
        n_para_total, n_para_filter1 = {}, {}
        for m in models:
            o = g_orig[(g_orig["model"] == m) & g_orig["passes_filter1"]]
            has_orig[m] = len(o) > 0
            if has_orig[m]:
                orig_label[m] = o["verdict"].iat[0]

            p_all = g_para[g_para["model"] == m]
            n_para_total[m] = p_all["paraphrased_question"].nunique()
            p_ok = p_all[p_all["passes_filter1"]]
            texts_ok[m] = set(p_ok["paraphrased_question"])
            n_para_filter1[m] = len(texts_ok[m])

        passes_filter2 = all(has_orig.values())
        shared_texts = set.intersection(*texts_ok.values()) if texts_ok else set()
        n_para_shared = len(shared_texts)
        included = passes_filter2 and n_para_shared > 0

        frow = {
            "idx": idx, "passes_filter2": passes_filter2,
            "n_para_shared": n_para_shared, "included": included,
        }
        for m in models:
            frow[f"has_orig_{m}"] = has_orig[m]
            frow[f"n_para_total_{m}"] = n_para_total[m]
            frow[f"n_para_filter1_{m}"] = n_para_filter1[m]
        funnel_rows.append(frow)

        if not included:
            continue

        shared = g_para[
            g_para["model"].isin(models)
            & g_para["paraphrased_question"].isin(shared_texts)
            & g_para["passes_filter1"]
        ]
        # One row per shared text, one column per model -- every model has
        # a value in every row by construction (shared_texts is the N-way
        # intersection of each model's own Filter-1 survivors).
        pivot = shared.pivot(index="paraphrased_question", columns="model", values="verdict")

        row = {"idx": idx, "n_para": n_para_shared}
        for m in models:
            labels_m = pivot[m]
            p = labels_m.value_counts(normalize=True)
            row[f"orig_label_{m}"] = orig_label[m]
            for v in VERDICTS:
                row[f"p{v}_{m}"] = p.get(v, 0.0)
            row[f"correct_frac_{m}"] = row[f"pcorrect_{m}"]
        profile_rows.append(row)

        for text, r in pivot.iterrows():
            for m in models:
                long_rows.append({
                    "idx": idx, "paraphrased_question": text,
                    "model": m, "verdict": r[m],
                })

    profiles = pd.DataFrame(profile_rows)
    profiles = profiles.set_index("idx") if len(profiles) else profiles.set_index(pd.Index([], name="idx"))
    funnel = pd.DataFrame(funnel_rows).set_index("idx")
    long_df = pd.DataFrame(long_rows, columns=["idx", "paraphrased_question", "model", "verdict"])

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(
                {"fingerprint": fingerprint, "profiles": profiles, "funnel": funnel, "long": long_df}, f
            )

    return _select_common_output(profiles, funnel, long_df, return_diagnostics, return_long)


def summarize_common_funnel(funnel: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """
    N-way analogue of profile_comparison.summarize_funnel() -- one row per
    filter stage: how many QUESTIONS survive. Reports each model's own
    Filter-1 survival individually (they can differ per model), then the
    two collective (all-models-at-once) stages.
    """
    n_total = len(funnel)

    def _pct(n):
        return n / n_total if n_total else float("nan")

    rows = [{"filter": "0_raw_dataset", "n_idx": n_total, "pct_of_total_idx": 1.0}]
    for m in models:
        n = int(funnel[f"has_orig_{m}"].sum())
        rows.append({"filter": f"1_filter1_agreement (model={m})", "n_idx": n, "pct_of_total_idx": _pct(n)})
    n2 = int(funnel["passes_filter2"].sum())
    rows.append({"filter": "2_filter2_orig_label_all_models", "n_idx": n2, "pct_of_total_idx": _pct(n2)})
    n3 = int(funnel["included"].sum())
    rows.append({"filter": "3_filter3_shared_paraphrases_all_models (final)", "n_idx": n3, "pct_of_total_idx": _pct(n3)})
    return pd.DataFrame(rows)
