# src/access_eval/analysis/profile_comparison.py
"""
Knowledge-accessibility profile comparison between two models on a shared
paraphrase set (formalizes the prototype in
analysis_v3_two_model_comparison.ipynb).

preprocess_grades() is the shared, model-pair-independent step: load the raw
grades CSV once and reduce it to one majority verdict per (model, idx,
paraphrased_question). Pure aggregation -- no filtering. Run it once, reuse
the result across every model pair you compare.

build_paired_profiles() is the model-pair-specific step and owns all three
filters end-to-end, in order:
  Filter 1 - per-model judge-agreement + GPT-unanimity filter on each graded
             row (original question AND every paraphrase)
  Filter 2 - the original question's label must have survived Filter 1 for
             BOTH models, otherwise the question can't be compared at all
  Filter 3 - restrict to the paraphrase TEXTS that survived Filter 1 for
             BOTH models (the shared/intersected subset), so every
             downstream comparison is apples-to-apples
Pass return_diagnostics=True to also get a per-question funnel DataFrame
showing exactly which of the three filters dropped each question (or shrank
its paraphrase count). Feed that funnel into summarize_funnel() for an
aggregated, stage-by-stage view.

build_paired_profiles()'s own Layer I-IV columns (tv_dist/js_dist, paired_agree/
kappa, jaccard/dice/hamming, transition_matrix()) are the primary way to compare
two models' profiles. aggregate_summary(), distribution_distance(), and
instance_level_report() are DEPRECATED -- kept only for
analysis_v3_two_model_comparison.ipynb, which still calls them -- and are
slated for removal or relocation out of this file. See each function's
docstring for why it's superseded.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

VERDICTS = ["correct", "incorrect", "not_attempted"]


def preprocess_grades(
    grades_path: str | Path,
    collapse_not_attempted: bool = False,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load the raw grades CSV and reduce it to one majority verdict per
    (model, idx, paraphrased_question), plus the raw judge-agreement stats
    (n_judges, n_agree, gpt_agree) needed to apply Filter 1 later. Pure
    aggregation -- no filtering happens here. Independent of any model pair —
    compute once, reuse for every build_paired_profiles() call.

    collapse_not_attempted: if True, remap "not_attempted" -> "incorrect" on
    the individual judge verdicts BEFORE computing the majority vote and
    agreement stats -- not as a relabel of the final verdict afterward. This
    matters: a judge saying "not_attempted" and a judge saying "incorrect"
    must count as AGREEING under this policy, which only holds if the remap
    happens before n_agree/gpt_agree are computed. Downstream functions
    (build_paired_profiles, aggregate_summary, VERDICTS-based tv_dist) need
    no changes -- they already degrade correctly to a 2-label verdict column
    (proven on the 2-label PopQA/TriviaQA datasets already running through
    this same code).

    cache_path: this is the expensive step (a groupby().apply() over every
    graded row) -- pass a path (get it from
    RunPaths.preprocessed_grades_cache_file(subset, collapse_not_attempted))
    to skip recomputing it on repeat runs. The cache is fingerprinted on
    (resolved source path, source mtime, source size, collapse_not_attempted)
    and is only reused if ALL four still match the current call -- a changed,
    moved, or re-graded source file invalidates it automatically. Written
    after a fresh computation whenever a path is given; no cache file has to
    exist ahead of time.

    The fingerprint is also stamped onto the returned DataFrame's `.attrs`
    (regardless of whether cache_path is set) -- build_paired_profiles() reads
    it from there to key its own cache off the exact same source-file
    identity, without re-hashing the (large) DataFrame.
    """
    grades_path = Path(grades_path)
    stat = grades_path.stat()
    fingerprint = (str(grades_path.resolve()), stat.st_mtime_ns, stat.st_size, collapse_not_attempted)

    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("fingerprint") == fingerprint:
            cached["df"].attrs["source_fingerprint"] = fingerprint
            return cached["df"]

    raw = pd.read_csv(grades_path)
    if collapse_not_attempted:
        raw = raw.assign(verdict=raw["verdict"].replace({"not_attempted": "incorrect"}))

    def summarize(g: pd.DataFrame) -> pd.Series:
        vc = g["verdict"].value_counts()
        gpt_verdicts = g.loc[g["evaluator"].str.contains("gpt", case=False), "verdict"]
        return pd.Series({
            "original_question": g["original_question"].iat[0],
            "verdict": vc.idxmax(),
            "n_judges": len(g),
            "n_agree": vc.max(),
            "gpt_agree": gpt_verdicts.nunique() <= 1,
        })

    result = (
        raw.groupby(["model", "idx", "paraphrased_question"])
        .apply(summarize)
        .reset_index()
    )
    result.attrs["source_fingerprint"] = fingerprint

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"fingerprint": fingerprint, "df": result}, f)

    return result


def preprocess_math_answers(
    answers_path: str | Path,
    dataset_path: str | Path,
    max_chars: int = 40_000,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Math-dataset counterpart to preprocess_grades() -- same output contract
    (one row per (model, idx, paraphrased_question) with a "verdict" column
    build_paired_profiles() can consume), but for datasets that skip the
    LLM-judge grading stage entirely (GSM8K, MATH-500). Correctness comes
    from math_verify (via analysis.math_grading), not judge majority vote,
    so there is no judge-agreement concept here: every row is graded once,
    deterministically, against the gold answer -- n_judges=1, n_agree=1,
    gpt_agree=True on every row, so build_paired_profiles()'s Filter 1
    (judge-agreement) passes everything unconditionally and every other
    Layer I-IV metric is computed exactly as it is for QA datasets.

    answers_path: RunPaths.answers_all_models_file(subset, conf_suffix) --
    raw per-model answer rows (model, idx, original_question,
    paraphrased_question, answer).
    dataset_path: the underlying dataset CSV (ProjectPaths.dataset_file(...)),
    used to look up gold answers by idx.

    cache_path: fingerprinted the same way as preprocess_grades() but keyed
    on BOTH source files (answers CSV and dataset CSV) plus max_chars, since
    grading depends on both. Pass RunPaths.preprocessed_grades_cache_file(...)
    (the name is QA-specific but the cache dir/mechanism is generic) to skip
    re-running math_verify on repeat calls -- grading ~200k rows of LaTeX
    answers is the expensive step here, analogous to the groupby().apply()
    in preprocess_grades().
    """
    from access_eval.analysis.math_grading import (
        add_gold_columns,
        add_normalized_values,
        filter_parseable,
    )

    answers_path, dataset_path = Path(answers_path), Path(dataset_path)
    a_stat, d_stat = answers_path.stat(), dataset_path.stat()
    fingerprint = (
        str(answers_path.resolve()), a_stat.st_mtime_ns, a_stat.st_size,
        str(dataset_path.resolve()), d_stat.st_mtime_ns, d_stat.st_size,
        max_chars,
    )

    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("fingerprint") == fingerprint:
            cached["df"].attrs["source_fingerprint"] = fingerprint
            return cached["df"]

    raw = pd.read_csv(answers_path)
    dataset_df = pd.read_csv(dataset_path)

    graded = add_gold_columns(raw, dataset_df, max_chars=max_chars)
    graded = add_normalized_values(graded)
    graded = filter_parseable(graded)

    result = graded[["model", "idx", "original_question", "paraphrased_question", "correct"]].copy()
    result["verdict"] = result["correct"].map({True: "correct", False: "incorrect"})
    result["n_judges"] = 1
    result["n_agree"] = 1
    result["gpt_agree"] = True
    result = result.drop(columns=["correct"])
    result.attrs["source_fingerprint"] = fingerprint

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"fingerprint": fingerprint, "df": result}, f)

    return result


def preprocess_ifbench_answers(
    answers_path: str | Path,
    dataset_path: str | Path,
    max_chars: int = 40_000,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    IFBench counterpart to preprocess_math_answers() -- same output contract
    (one row per (model, idx, paraphrased_question) with a "verdict" column
    build_paired_profiles() can consume), same reasoning (deterministic
    grading inside this stage, no judge-agreement concept, so n_judges=1,
    n_agree=1, gpt_agree=True unconditionally). Only difference: grading
    comes from analysis.ifbench_grading.add_gold_columns() (deterministic
    instruction-following constraint checking) instead of math_grading's
    math_verify wrapper, and there's no add_normalized_values() step (no
    numeric gold value to normalize against -- IFBench's "correct" is
    already a strict all-instructions-followed boolean, see add_gold_columns'
    own docstring).

    answers_path: RunPaths.answers_all_models_file(subset, conf_suffix).
    dataset_path: ProjectPaths.dataset_file(get_dataset_spec("IFBench")["path"]),
    i.e. the IFBench constraint-list CSV (instruction_id_list/kwargs per idx).

    cache_path: same fingerprinting scheme as preprocess_math_answers()
    (source answers CSV + dataset CSV + max_chars).
    """
    from access_eval.analysis.ifbench_grading import add_gold_columns, filter_parseable

    answers_path, dataset_path = Path(answers_path), Path(dataset_path)
    a_stat, d_stat = answers_path.stat(), dataset_path.stat()
    fingerprint = (
        str(answers_path.resolve()), a_stat.st_mtime_ns, a_stat.st_size,
        str(dataset_path.resolve()), d_stat.st_mtime_ns, d_stat.st_size,
        max_chars,
    )

    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("fingerprint") == fingerprint:
            cached["df"].attrs["source_fingerprint"] = fingerprint
            return cached["df"]

    raw = pd.read_csv(answers_path)
    dataset_df = pd.read_csv(dataset_path)

    graded = add_gold_columns(raw, dataset_df, max_chars=max_chars)
    graded = filter_parseable(graded)

    result = graded[["model", "idx", "original_question", "paraphrased_question", "correct"]].copy()
    result["verdict"] = result["correct"].map({True: "correct", False: "incorrect"})
    result["n_judges"] = 1
    result["n_agree"] = 1
    result["gpt_agree"] = True
    result = result.drop(columns=["correct"])
    result.attrs["source_fingerprint"] = fingerprint

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"fingerprint": fingerprint, "df": result}, f)

    return result


def _cohens_kappa(a: pd.Series, b: pd.Series) -> float:
    """Chance-corrected pointwise agreement between two label sequences of equal length."""
    po = (a.values == b.values).mean()
    labels = sorted(set(a) | set(b))
    pa = a.value_counts(normalize=True).reindex(labels, fill_value=0.0)
    pb = b.value_counts(normalize=True).reindex(labels, fill_value=0.0)
    pe = (pa * pb).sum()
    return 1.0 if pe >= 1.0 else (po - pe) / (1 - pe)


def _select_profiles_output(
    profiles: pd.DataFrame,
    funnel: pd.DataFrame,
    pairs: pd.DataFrame,
    return_diagnostics: bool,
    return_pairs: bool,
) -> pd.DataFrame | tuple[pd.DataFrame, ...]:
    """Slice build_paired_profiles()'s always-computed (profiles, funnel, pairs) down to what the caller asked for."""
    out = (profiles,)
    if return_diagnostics:
        out += (funnel,)
    if return_pairs:
        out += (pairs,)
    return out if len(out) > 1 else profiles


def build_paired_profiles(
    agg: pd.DataFrame,
    model_a: str,
    model_b: str,
    min_agreement_fraction: float = 0.5,
    return_diagnostics: bool = False,
    return_pairs: bool = False,
    cache_path: str | Path | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, ...]:
    """
    Per-question accessibility profile for model_a and model_b, computed only on
    paraphrase texts BOTH models answered.

    `agg` is the output of preprocess_grades() (unfiltered). This function
    applies all three filters itself, in order, since Filters 2 and 3 are
    inherently pair-specific and Filter 1 needs to be re-scoped to just
    model_a/model_b anyway:

      Filter 1 - a graded row (original question or paraphrase) survives if
                 `n_agree > n_judges * min_agreement_fraction` AND the
                 GPT-family judges (evaluator name containing "gpt")
                 unanimously agree with each other.
      Filter 2 - a question is only comparable if its ORIGINAL question's
                 row survived Filter 1 for BOTH model_a and model_b.
      Filter 3 - among comparable questions, restrict to the paraphrase
                 TEXTS that survived Filter 1 for BOTH models (the
                 shared/intersected subset).

    Returns a DataFrame indexed by idx with orig_label_a/b, n_para,
    p{verdict}_a/b, correct_frac_a/b (= pcorrect_a/b), plus three layers of
    pairwise comparison metrics, all computed per-question from the same
    shared-paraphrase pivot (model_a's and model_b's verdict on each shared
    text, lined up):

      Layer I  - marginal profile shift, ignores which text produced which
                 verdict: tv_dist (total variation), js_dist (Jensen-Shannon
                 distance, base 2, so both are bounded to [0, 1])
      Layer II - pointwise correspondence, same text/both models:
                 paired_agree, paired_disagree, kappa (chance-corrected
                 paired_agree)
      Layer III - accessible-region similarity, binary correct/not-correct
                 sets: jaccard, dice (= F1), hamming (mismatch rate on the
                 binary sets, distinct from paired_disagree's 3-label version)

    If return_diagnostics=True, also returns a DataFrame indexed by idx
    (including questions that got dropped) with one row per question
    showing how each filter acted on it -- feed it to summarize_funnel() for
    an aggregated view. Columns: n_para_total(_a/_b) [pre-Filter-1 pool],
    has_orig_a/b + passes_filter2 [Filter 2], n_para_filter1_a/b [per-model
    Filter-1 survivors], n_para_shared [Filter 3 result], included [made it
    into the returned profiles].

    If return_pairs=True, also returns a long-format DataFrame (idx,
    label_a, label_b), one row per shared paraphrase across every included
    question -- the same per-idx pivots Layer II/III consume, just pooled
    instead of collapsed. Feed it to transition_matrix() for the Layer IV
    corpus-level view neither per-idx layer can support on its own (a single
    question's ~5-20 paraphrases are too sparse for a readable 3x3 matrix).

    With any return_* flag set, the return value is a tuple in the order
    (profiles, [funnel], [pairs]); with neither, just profiles.

    cache_path: this per-idx loop is the expensive step for a given model
    pair -- pass a path (get it from
    RunPaths.paired_profiles_cache_file(subset, model_a, model_b,
    min_agreement_fraction, collapse_not_attempted)) to skip recomputing it.
    Fingerprinted on (agg.attrs["source_fingerprint"], model_a, model_b,
    min_agreement_fraction) -- `agg` MUST come from preprocess_grades() (it's
    the one that stamps source_fingerprint), otherwise this raises, since
    there'd be nothing reliable to invalidate the cache on. All three of
    profiles/funnel/pairs are cached together regardless of return_diagnostics/
    return_pairs (they're already computed internally either way) -- those two
    flags only control what THIS CALL returns, not what gets cached.
    """
    if cache_path is not None:
        source_fp = agg.attrs.get("source_fingerprint")
        if source_fp is None:
            raise ValueError(
                "build_paired_profiles(cache_path=...) requires `agg` to carry "
                "a source_fingerprint (agg.attrs) so the cache can be "
                "invalidated correctly -- build `agg` via preprocess_grades(), "
                "which sets this automatically."
            )
        fingerprint = (source_fp, model_a, model_b, min_agreement_fraction)
        if Path(cache_path).exists():
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("fingerprint") == fingerprint:
                return _select_profiles_output(
                    cached["profiles"], cached["funnel"], cached["pairs"],
                    return_diagnostics, return_pairs,
                )

    pair = agg[agg["model"].isin([model_a, model_b])].copy()
    pair["is_orig"] = pair["original_question"] == pair["paraphrased_question"]
    pair["passes_filter1"] = (
        pair["n_agree"] > pair["n_judges"] * min_agreement_fraction
    ) & pair["gpt_agree"]

    funnel_rows = []
    profile_rows = []
    pair_rows = []
    for idx, g in pair.groupby("idx"):
        g_orig = g[g["is_orig"]]
        g_para = g[~g["is_orig"]]

        n_para_total_a = g_para.loc[g_para["model"] == model_a, "paraphrased_question"].nunique()
        n_para_total_b = g_para.loc[g_para["model"] == model_b, "paraphrased_question"].nunique()

        g_para_ok = g_para[g_para["passes_filter1"]]
        texts_ok_a = set(g_para_ok.loc[g_para_ok["model"] == model_a, "paraphrased_question"])
        texts_ok_b = set(g_para_ok.loc[g_para_ok["model"] == model_b, "paraphrased_question"])

        orig_a = g_orig[(g_orig["model"] == model_a) & g_orig["passes_filter1"]]
        orig_b = g_orig[(g_orig["model"] == model_b) & g_orig["passes_filter1"]]
        has_orig_a, has_orig_b = len(orig_a) > 0, len(orig_b) > 0
        passes_filter2 = has_orig_a and has_orig_b

        shared_texts = texts_ok_a & texts_ok_b
        n_para_shared = len(shared_texts)
        included = passes_filter2 and n_para_shared > 0

        funnel_rows.append({
            "idx": idx,
            "n_para_total_a": n_para_total_a, "n_para_total_b": n_para_total_b,
            "has_orig_a": has_orig_a, "has_orig_b": has_orig_b,
            "passes_filter2": passes_filter2,
            "n_para_filter1_a": len(texts_ok_a), "n_para_filter1_b": len(texts_ok_b),
            "n_para_shared": n_para_shared,
            "included": included,
        })

        if not included:
            continue
        shared = g_para_ok[g_para_ok["paraphrased_question"].isin(shared_texts)]
        # Pivot so each shared text lines up with model_a's and model_b's verdict
        # on that EXACT text -- every pairwise metric below reuses this one pivot.
        pivot = shared.pivot(index="paraphrased_question", columns="model", values="verdict")
        a_labels, b_labels = pivot[model_a], pivot[model_b]
        pa, pb = a_labels.value_counts(normalize=True), b_labels.value_counts(normalize=True)

        row = {
            "idx": idx, "n_para": n_para_shared,
            "orig_label_a": orig_a["verdict"].iat[0], "orig_label_b": orig_b["verdict"].iat[0],
        }
        for tag, p in (("a", pa), ("b", pb)):
            for v in VERDICTS:
                row[f"p{v}_{tag}"] = p.get(v, 0.0)
            row[f"correct_frac_{tag}"] = row[f"pcorrect_{tag}"]

        # Layer I -- marginal profile shift (ignores which text produced which verdict)
        row["tv_dist"] = 0.5 * sum(abs(row[f"p{v}_a"] - row[f"p{v}_b"]) for v in VERDICTS)
        row["js_dist"] = jensenshannon([pa.get(v, 0.0) for v in VERDICTS], [pb.get(v, 0.0) for v in VERDICTS], base=2)
        row["correct_frac_abs_diff"] = abs(row["correct_frac_a"] - row["correct_frac_b"])  # TV on the binary correct/not-correct collapse

        # Layer II -- pointwise correspondence (same shared text, both models)
        row["paired_agree"] = (a_labels == b_labels).mean()
        row["paired_disagree"] = 1.0 - row["paired_agree"]
        row["kappa"] = _cohens_kappa(a_labels, b_labels)

        # Layer III -- accessible-region similarity (binary correct / not-correct sets)
        succ_a, succ_b = a_labels == "correct", b_labels == "correct"
        both, either = (succ_a & succ_b).sum(), (succ_a | succ_b).sum()
        row["jaccard"] = both / either if either else np.nan
        row["dice"] = 2 * both / (succ_a.sum() + succ_b.sum()) if (succ_a.sum() + succ_b.sum()) else np.nan
        row["hamming"] = (succ_a != succ_b).mean()

        profile_rows.append(row)
        pair_rows.extend({"idx": idx, "label_a": la, "label_b": lb} for la, lb in zip(a_labels, b_labels))

    profiles = pd.DataFrame(profile_rows)
    profiles = profiles.set_index("idx") if len(profiles) else profiles.set_index(pd.Index([], name="idx"))
    funnel = pd.DataFrame(funnel_rows).set_index("idx")
    pairs = pd.DataFrame(pair_rows, columns=["idx", "label_a", "label_b"])

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(
                {"fingerprint": fingerprint, "profiles": profiles, "funnel": funnel, "pairs": pairs}, f
            )

    return _select_profiles_output(profiles, funnel, pairs, return_diagnostics, return_pairs)


def build_single_profile(
    agg: pd.DataFrame,
    model: str,
    min_agreement_fraction: float = 0.5,
    return_diagnostics: bool = False,
    return_pairs: bool = False,
    cache_path: str | Path | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, ...]:
    """
    Single-model counterpart to build_paired_profiles() -- the per-question
    knowledge-accessibility profile for ONE model against conventional
    (single-prompt) benchmarking, instead of one model against another.
    Works identically for QA (preprocess_grades()) and math
    (preprocess_math_answers()) sources -- this function only cares that
    `agg` has the (model, idx, paraphrased_question, original_question,
    verdict, n_judges, n_agree, gpt_agree) schema, not where verdict came
    from.

    Filters 2 and 3 collapse to "just this model" (no second model to
    intersect against): a question is included if its original question
    survived Filter 1 and it has at least one Filter-1-surviving paraphrase.
    Filter 1 itself (judge-agreement + GPT-unanimity) is unchanged.

    There IS still a meaningful Layer I here, just not between two models:
    it's the shift between the original question's answer (a one-hot
    distribution at orig_label -- that's what conventional single-prompt
    benchmarking reports) and the marginal verdict distribution over
    shared paraphrases -- i.e. exactly what you'd miss by trusting the
    original phrasing alone. tv_dist/js_dist are computed the same way as
    build_paired_profiles, just against that implicit one-hot instead of a
    second model's p{v}. correct_frac_abs_diff reduces to
    |1 - correct_frac| when orig_label == "correct", or plain correct_frac
    when it isn't (the "accessibility gap" from trusting a single wrong
    answer). Layer II (paired_agree/kappa) and Layer III
    (jaccard/dice/hamming) aren't meaningful for a single model -- both
    require a second model's labels on the same shared text to compare
    against.

    Returns a DataFrame indexed by idx with orig_label, n_para, p{verdict}
    (marginal distribution over shared paraphrases), correct_frac (=
    pcorrect), tv_dist, js_dist, correct_frac_abs_diff, and two
    orig_label-independent per-question flags (latent is a SUPERSET of
    reliable -- reliable questions are also latent, so is a brittle
    question where 0 < correct_frac < 1, and so is a question the model
    got wrong on every paraphrase but happened to get right originally is
    NOT, since correct_frac == 0 there):
      reliable -- correct_frac >= 1.0 (correct on every shared paraphrase)
      latent   -- correct_frac > 0 (correct on at least one)

    If return_pairs=True, also returns a long-format DataFrame (idx,
    orig_label, para_label), one row per shared paraphrase across every
    included question -- feed it to self_transition_matrix() for the
    pooled Layer-IV-style view. Unlike the paired case, no separate
    stratification helper is needed: orig_label is already one axis of
    that matrix, so slicing by row directly answers "given the model said
    X on the original phrasing, what did paraphrases say" -- the
    partitioning falls out of the matrix itself.

    cache_path: fingerprinted on (agg.attrs["source_fingerprint"], model,
    min_agreement_fraction), same requirement as build_paired_profiles
    (agg must come from preprocess_grades()/preprocess_math_answers()).
    """
    if cache_path is not None:
        source_fp = agg.attrs.get("source_fingerprint")
        if source_fp is None:
            raise ValueError(
                "build_single_profile(cache_path=...) requires `agg` to carry "
                "a source_fingerprint (agg.attrs) so the cache can be "
                "invalidated correctly -- build `agg` via preprocess_grades() "
                "or preprocess_math_answers(), which set this automatically."
            )
        fingerprint = (source_fp, model, min_agreement_fraction)
        if Path(cache_path).exists():
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("fingerprint") == fingerprint:
                return _select_profiles_output(
                    cached["profiles"], cached["funnel"], cached["pairs"],
                    return_diagnostics, return_pairs,
                )

    sub = agg[agg["model"] == model].copy()
    sub["is_orig"] = sub["original_question"] == sub["paraphrased_question"]
    sub["passes_filter1"] = (
        sub["n_agree"] > sub["n_judges"] * min_agreement_fraction
    ) & sub["gpt_agree"]

    funnel_rows = []
    profile_rows = []
    pair_rows = []
    for idx, g in sub.groupby("idx"):
        g_orig = g[g["is_orig"]]
        g_para = g[~g["is_orig"]]

        n_para_total = g_para["paraphrased_question"].nunique()
        g_para_ok = g_para[g_para["passes_filter1"]]

        orig_ok = g_orig[g_orig["passes_filter1"]]
        has_orig = len(orig_ok) > 0
        n_para = g_para_ok["paraphrased_question"].nunique()
        included = has_orig and n_para > 0

        funnel_rows.append({
            "idx": idx,
            "n_para_total": n_para_total, "has_orig": has_orig,
            "n_para_filter1": n_para, "included": included,
        })

        if not included:
            continue

        labels = g_para_ok["verdict"]
        p = labels.value_counts(normalize=True)
        orig_label = orig_ok["verdict"].iat[0]

        row = {"idx": idx, "n_para": n_para, "orig_label": orig_label}
        for v in VERDICTS:
            row[f"p{v}"] = p.get(v, 0.0)
        row["correct_frac"] = row["pcorrect"]

        # Layer I -- shift between the original answer (one-hot at orig_label,
        # what single-prompt benchmarking reports) and the marginal
        # paraphrase distribution (what you'd see if you asked it many ways)
        orig_onehot = {v: (1.0 if v == orig_label else 0.0) for v in VERDICTS}
        row["tv_dist"] = 0.5 * sum(abs(orig_onehot[v] - row[f"p{v}"]) for v in VERDICTS)
        row["js_dist"] = jensenshannon(
            [orig_onehot[v] for v in VERDICTS], [row[f"p{v}"] for v in VERDICTS], base=2
        )
        row["correct_frac_abs_diff"] = abs(orig_onehot["correct"] - row["correct_frac"])

        # orig_label-independent: latent is a superset of reliable
        row["reliable"] = row["correct_frac"] >= 1.0
        row["latent"] = row["correct_frac"] > 0

        profile_rows.append(row)
        pair_rows.extend({"idx": idx, "orig_label": orig_label, "para_label": lbl} for lbl in labels)

    profiles = pd.DataFrame(profile_rows)
    profiles = profiles.set_index("idx") if len(profiles) else profiles.set_index(pd.Index([], name="idx"))
    funnel = pd.DataFrame(funnel_rows).set_index("idx")
    pairs = pd.DataFrame(pair_rows, columns=["idx", "orig_label", "para_label"])

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(
                {"fingerprint": fingerprint, "profiles": profiles, "funnel": funnel, "pairs": pairs}, f
            )

    return _select_profiles_output(profiles, funnel, pairs, return_diagnostics, return_pairs)


def transition_matrix(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Layer IV -- pooled, corpus-level label-transition structure. Unlike Layer
    II's per-idx paired_agree/kappa, which collapse each shared paraphrase's
    (label_a, label_b) pair down to matched/mismatched, this keeps the full
    pair and pools it across every shared paraphrase in every included
    question -- a single question's ~5-20 paraphrases are too sparse on
    their own for a readable 3x3 matrix. `pairs` is the extra return value
    from build_paired_profiles(..., return_pairs=True) (same underlying
    pivot Layer II/III already computed, just concatenated instead of
    collapsed). Cell (i, j) = P(model_a said i AND model_b said j) over all
    pooled shared paraphrases.
    """
    return (
        pd.crosstab(pairs["label_a"], pairs["label_b"], normalize=True)
        .reindex(index=VERDICTS, columns=VERDICTS, fill_value=0.0)
    )


def self_transition_matrix(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Single-model counterpart to transition_matrix() -- `pairs` here is
    build_single_profile(..., return_pairs=True)'s (idx, orig_label,
    para_label) table. Cell (i, j) = P(orig_label said i AND a pooled
    shared paraphrase said j) over every included question's shared
    paraphrases. orig_label is already one axis of this matrix, so reading
    off row i (and renormalizing it to sum to 1, i.e. conditioning on
    orig_label == i) directly answers "given the model said i originally,
    what fraction of paraphrases said j" -- no separate stratification
    function needed, unlike transition_matrix()'s pairwise case where
    same/diff_orig_label had to be manufactured by comparing two models.
    """
    return (
        pd.crosstab(pairs["orig_label"], pairs["para_label"], normalize=True)
        .reindex(index=VERDICTS, columns=VERDICTS, fill_value=0.0)
    )


def stratified_distribution(
    profiles: pd.DataFrame,
    metrics: tuple[str, ...] = ("correct_frac_abs_diff", "tv_dist", "paired_disagree", "jaccard"),
    thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5),
) -> pd.DataFrame:
    """
    Continuous-first replacement for the deprecated instance_level_report():
    split questions into same_orig_label / diff_orig_label (both-correct and
    both-wrong both count as "same" -- that's the case where original-question
    accuracy alone would call the two models equivalent, whether or not it's
    right), then report each metric's full distribution per group -- mean,
    median, p25/p75/p90 -- instead of collapsing straight to one
    threshold-derived number. `frac_gt_{t}` columns are included per
    `thresholds` as a secondary, readable summary only; the distribution
    columns are the primary result.
    """
    same = profiles["orig_label_a"] == profiles["orig_label_b"]
    rows = []
    for group, mask in [("same_orig_label", same), ("diff_orig_label", ~same)]:
        for metric in metrics:
            d = profiles.loc[mask, metric]
            row = {
                "group": group, "metric": metric, "n": len(d),
                "mean": d.mean(), "median": d.median(),
                "p25": d.quantile(0.25), "p75": d.quantile(0.75), "p90": d.quantile(0.90),
            }
            row.update({f"frac_gt_{t}": (d > t).mean() for t in thresholds})
            rows.append(row)
    return pd.DataFrame(rows)


def stratified_single_distribution(
    profiles: pd.DataFrame,
    metrics: tuple[str, ...] = ("correct_frac_abs_diff", "tv_dist", "js_dist"),
    thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5),
) -> pd.DataFrame:
    """
    Single-model counterpart to stratified_distribution() -- there's no
    second model's orig_label to compare against, so the natural partition
    is orig_label itself: "raw" (every included question, pooled) plus one
    group per distinct orig_label value actually PRESENT in `profiles`
    (dynamic, not hardcoded) -- typically correct/incorrect for math or a
    collapsed QA run, plus not_attempted too for an uncollapsed QA run
    where the model actually abstained on some original questions. Default
    metrics are the three Layer I columns build_single_profile() produces
    (correct_frac_abs_diff, tv_dist, js_dist) -- paired_disagree/jaccard
    aren't columns here since Layer II/III don't exist for a single model.
    Same mean/median/p25/p75/p90-then-frac_gt_{t} ordering convention as
    stratified_distribution().
    """
    groups = [("raw", pd.Series(True, index=profiles.index))]
    for label in sorted(profiles["orig_label"].unique()):
        groups.append((f"orig_label=={label}", profiles["orig_label"] == label))

    rows = []
    for group, mask in groups:
        for metric in metrics:
            d = profiles.loc[mask, metric]
            row = {
                "group": group, "metric": metric, "n": len(d),
                "mean": d.mean(), "median": d.median(),
                "p25": d.quantile(0.25), "p75": d.quantile(0.75), "p90": d.quantile(0.90),
            }
            row.update({f"frac_gt_{t}": (d > t).mean() for t in thresholds})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_funnel(funnel: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the per-idx funnel from build_paired_profiles(return_diagnostics=True)
    into one row per filter stage: how many QUESTIONS survive, and how many
    PARAPHRASES remain, at each step. `n_para` at the "orig label (both)" stage
    is an upper bound (min of each model's surviving count) since the exact
    shared count isn't known until Filter 3 actually intersects the texts.
    """
    n_total = len(funnel)
    all_idx = pd.Series(True, index=funnel.index)
    both_a_b = funnel[["n_para_filter1_a", "n_para_filter1_b"]].min(axis=1)

    stages = [
        ("0_raw_dataset (model_a pool)", all_idx, funnel["n_para_total_a"]),
        ("0_raw_dataset (model_b pool)", all_idx, funnel["n_para_total_b"]),
        ("1_filter1_agreement (model_a orig)", funnel["has_orig_a"], funnel["n_para_filter1_a"]),
        ("1_filter1_agreement (model_b orig)", funnel["has_orig_b"], funnel["n_para_filter1_b"]),
        ("2_filter2_orig_label_both_models", funnel["passes_filter2"], both_a_b),
        ("3_filter3_shared_paraphrases (final)", funnel["included"], funnel["n_para_shared"]),
    ]

    rows = []
    prev_n = n_total
    for name, mask, para_col in stages:
        n = int(mask.sum())
        rows.append({
            "filter": name,
            "n_idx": n,
            "pct_of_total_idx": n / n_total if n_total else np.nan,
            "pct_of_prev_stage": n / prev_n if prev_n else np.nan,
            "n_para_remaining": para_col[mask].sum(),
            "mean_para_per_idx": para_col[mask].mean() if n else np.nan,
        })
        if name.startswith("2_") or name.startswith("3_"):
            prev_n = n
    return pd.DataFrame(rows)


def summarize_row_level_funnel(funnel: pd.DataFrame) -> pd.DataFrame:
    """
    Row-level companion to summarize_funnel(). That function answers "how many
    QUESTIONS survive each filter" (and reports paraphrase counts *within*
    that idx-conditioned view). This answers a different question: "what
    fraction of all graded ROWS -- the original question, or an individual
    paraphrase text -- survive each filter," computed WITHOUT pre-conditioning
    on whether the question's original label survived. That distinction
    matters because Filter 1 acts on individual rows independently of
    whether the rest of the question later gets dropped by Filter 2/3.

    Every idx has exactly one original-question row and n_para_total_{a,b}
    paraphrase rows per model (verify this holds for your dataset -- it does
    for TruthfulQA_v00: 511/511 original rows, 4337/4337 paraphrase rows,
    graded for both models). Original-question rows are only touched by
    Filters 1 and 2 (Filter 3 only intersects paraphrase texts).
    """
    n_idx = len(funnel)
    rows = []

    def add(population, filter_name, n, n_total, prev_n):
        rows.append({
            "population": population,
            "filter": filter_name,
            "n_rows": n,
            "pct_of_total": n / n_total if n_total else np.nan,
            "pct_of_prev_stage": n / prev_n if prev_n else np.nan,
        })

    for tag in ("a", "b"):
        pop = f"original_question_row (model_{tag})"
        n1 = int(funnel[f"has_orig_{tag}"].sum())
        n2 = int(funnel["passes_filter2"].sum())
        add(pop, "0_raw (always graded)", n_idx, n_idx, n_idx)
        add(pop, "1_filter1_agreement", n1, n_idx, n_idx)
        add(pop, "2_filter2_both_models_agree (final)", n2, n_idx, n1)

    for tag in ("a", "b"):
        pop = f"paraphrase_rows (model_{tag})"
        n0 = int(funnel[f"n_para_total_{tag}"].sum())
        n1 = int(funnel[f"n_para_filter1_{tag}"].sum())
        n2 = int(funnel.loc[funnel["passes_filter2"], f"n_para_filter1_{tag}"].sum())
        add(pop, "0_raw", n0, n0, n0)
        add(pop, "1_filter1_agreement", n1, n0, n0)
        add(pop, "2_filter2_restrict_to_comparable_idx", n2, n0, n1)

    pop = "paraphrase_rows (shared, final)"
    n0 = int(funnel["n_para_total_a"].sum())  # = n_para_total_b, same raw pool
    n2_upper_bound = int(
        funnel.loc[funnel["passes_filter2"], ["n_para_filter1_a", "n_para_filter1_b"]]
        .min(axis=1).sum()
    )
    n3 = int(funnel.loc[funnel["included"], "n_para_shared"].sum())
    add(pop, "2_filter2_restrict_to_comparable_idx (upper bound, min(a,b))", n2_upper_bound, n0, n0)
    add(pop, "3_filter3_shared_intersection (final)", n3, n0, n2_upper_bound)

    return pd.DataFrame(rows)


def aggregate_summary(profiles: pd.DataFrame, mastered_threshold: float = 0.8) -> pd.DataFrame:
    """
    DEPRECATED: collapses each model independently, never actually comparing
    paired rows -- superseded by build_paired_profiles()'s Layer I-IV metrics.
    Kept only because analysis_v3_two_model_comparison.ipynb still calls it;
    slated for removal or relocation out of this file.

    Level 2.1 — collapse each model to one row (orig_acc, para_acc, state fractions), then compare.
    """
    rows = {}
    for tag in ("a", "b"):
        correct = profiles[f"orig_label_{tag}"] == "correct"
        cf = profiles[f"correct_frac_{tag}"]
        rows[tag] = {
            "orig_acc":     correct.mean(),
            "para_acc":     cf.mean(),
            "pct_reliable": (correct & (cf >= mastered_threshold)).mean(),
            "pct_brittle":  (correct & (cf <  mastered_threshold)).mean(),
            "pct_latent":   (~correct & (cf > 0)).mean(),
            "pct_absent":   (~correct & (cf == 0)).mean(),
        }
    return pd.DataFrame(rows).T


def distribution_distance(profiles: pd.DataFrame, metric: str = "correct_frac") -> dict:
    """
    DEPRECATED: treats metric_a/metric_b as two unpaired populations (proven
    order-invariant -- shuffling which idx a value belongs to changes nothing),
    so it never uses the pairing build_paired_profiles() provides. Superseded
    by the Layer I-IV metrics. Kept only because
    analysis_v3_two_model_comparison.ipynb still calls it; slated for removal
    or relocation out of this file.

    Level 2.2 — compare the two models' per-question `metric` distributions as populations.
    """
    a, b = profiles[f"{metric}_a"], profiles[f"{metric}_b"]
    ks = ks_2samp(a, b)
    return {"ks_stat": ks.statistic, "ks_pvalue": ks.pvalue, "wasserstein": wasserstein_distance(a, b)}


def instance_level_report(
    profiles: pd.DataFrame,
    metric: str = "correct_frac",
    threshold: float = 0.4,
) -> pd.DataFrame:
    """
    DEPRECATED: does use pairing, but only for one hardcoded metric with an
    arbitrary threshold -- superseded by build_paired_profiles()'s Layer I-IV
    metrics, which cover this same ground per-idx without a threshold. Kept
    only because analysis_v3_two_model_comparison.ipynb still calls it;
    slated for removal or relocation out of this file.

    Level 2.3 — paired per-question diff on `metric`, split by whether orig labels
    agreed, then aggregated:

      same_orig_label + high frac_mismatched -> accuracy AGREES, accessibility DOESN'T
      diff_orig_label + high frac_matched     -> accuracy DISAGREES, accessibility DOES
    """
    delta = (profiles[f"{metric}_a"] - profiles[f"{metric}_b"]).abs()
    same_label = profiles["orig_label_a"] == profiles["orig_label_b"]

    rows = []
    for name, mask in [("same_orig_label", same_label), ("diff_orig_label", ~same_label)]:
        d = delta[mask]
        rows.append({
            "group": name,
            "n": len(d),
            "mean_abs_delta": d.mean(),
            "frac_matched": (d <= threshold).mean(),
            "frac_mismatched": (d > threshold).mean(),
        })
    return pd.DataFrame(rows)
