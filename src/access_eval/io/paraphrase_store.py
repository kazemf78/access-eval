# src/access_eval/io/paraphrase_store.py
"""
Neutral artifact-access layer for the paraphrase pool + judge-verdict cache --
sits alongside the other io/ modules (`identity.py`, `judge_cache.py`,
`compat.py`) rather than inside either workflow script that touches
paraphrases. `paraphrase.py` and `refilter_paraphrases.py` both import FROM
here; neither imports from the other for this. This is what lets
`answer_generation.py` (roadmap) later depend on the same "currently accepted
paraphrases" view without an awkward workflow-script-depends-on-another-
workflow-script chain.
"""

import re
from collections import Counter
from typing import List, Optional, Union

import pandas as pd

from access_eval.io.identity import paraphrase_content_hash
from access_eval.io.judge_cache import load_judge_cache, prompt_hash_for
from access_eval.prompts import load_prompt

_GENERATED_COLUMNS = ["idx", "original_question", "paraphrased_question", "para_hash"]
_ACCEPTED_COLUMNS = _GENERATED_COLUMNS

# ---------------------------------------------------------------------------
# Lexical-overlap primitive -- canonical home. paraphrase.py imports
# `lexical_overlap_ratio` from here (as `_lexical_overlap_ratio`, its
# pre-existing internal name, for both its own generation-time dedup
# (_accepted_candidates) and ParaphraseGenerator._passes_overlap) rather than
# each module keeping its own copy. Living here (not in pipeline/paraphrase.py)
# is what lets dedup_by_overlap/cross_validated_paraphrases below use it
# without an io/-imports-from-pipeline/ layering violation.
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _words(s: str) -> List[str]:
    return _WORD_RE.findall(s.lower())


def lexical_overlap_ratio(a: str, b: str) -> float:
    """Returns ratio = (#common words) / min(len(a_words), len(b_words))."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    ca, cb = Counter(wa), Counter(wb)
    common = sum((ca & cb).values())
    return common / max(1, min(len(wa), len(wb)))


def load_generated_paraphrases(rp) -> pd.DataFrame:
    """
    Every candidate ever generated for this run -- judged or not, accepted or
    rejected, all mixed together with no distinction. A thin read of
    `paraphrases_file()` with `para_hash` attached (recomputed fresh from
    text each time, never a stored column -- see io/identity.py).

    Deduplicated by (idx, para_hash), keep-first: `paraphrases_file()` is
    append-only and callers (`paraphrase.py`'s resume in particular) don't
    always know the exact on-disk pool for an idx at generation time, so a
    literal duplicate candidate can genuinely end up appended twice across
    rounds/calls. Without this, a duplicate would join against the SAME
    judge-cache verdict and be counted twice by `accepted_paraphrases()` /
    `reconcile_quota()`, silently satisfying a quota with fewer truly
    distinct formulations than it looks like. This is a safety net --
    generation-time dedup against the on-disk pool (see `paraphrase.py`'s
    `_generate_round`) is the primary defense; this is what still holds if
    that's ever bypassed (e.g. rows appended by some other path).
    """
    path = rp.paraphrases_file()
    if not path.exists():
        return pd.DataFrame(columns=_GENERATED_COLUMNS)
    df = pd.read_csv(path)
    if df.empty:
        df["para_hash"] = []
        return df
    df["para_hash"] = df.apply(
        lambda r: paraphrase_content_hash(r["original_question"], r["paraphrased_question"]),
        axis=1,
    )
    return df.drop_duplicates(subset=["idx", "para_hash"], keep="first").reset_index(drop=True)


def accepted_paraphrases(rp, judge_model: str, prompt_name: Optional[str] = None) -> pd.DataFrame:
    """
    Every generated candidate currently judged `llm_equiv == True` under
    `judge_model` -- a live join of `load_generated_paraphrases()` against
    the judge-verdict cache (`io.judge_cache`), not a materialized snapshot
    file. Reflects whatever the cache currently holds: judging more
    candidates, or refiltering with a stricter judge that overturns
    previously-accepted rows, changes this view's result on the next call
    with no separate rebuild step.

    A candidate with NO verdict yet under this (judge_model, prompt_name) is
    excluded -- not-yet-judged is not the same as accepted. Callers that need
    "judge whatever's still ungraded first" should do so before calling this
    (see `paraphrase.py`'s resume, `refilter_paraphrases.run_refilter()`).

    `prompt_name` must be a file-backed prompt under `prompts/paraphrase/`
    (resolved via `prompts.load_prompt`, matching `io.judge_cache`'s
    `prompt_hash_for`) -- this excludes `refilter_paraphrases.py`'s CLI-only
    inline prompt convenience names ("default"/"detailed"/"with_examples",
    not backed by a file), which are for ad-hoc/manual refiltering runs, not
    for the automated pipeline calling this function.

    `prompt_name=None` (default) drops the prompt_hash filter entirely --
    "accepted by judge_model under ANY prompt it's ever been judged with",
    rather than pinning to one specific prompt version. Useful for a caller
    that only cares which judge signed off, not which prompt wording did it.
    A para_hash judged under two different prompts, with at least one YES, is
    included once (deduped by (idx, para_hash), keep-first) -- not once per
    matching prompt.
    """
    paras = load_generated_paraphrases(rp)
    if paras.empty:
        return paras

    cache_df = load_judge_cache(rp)
    if cache_df.empty:
        return paras.iloc[0:0]

    if prompt_name is None:
        judge_mask = cache_df["judge_model"] == judge_model
    else:
        prompt_template_text = load_prompt(f"paraphrase/{prompt_name}")
        p_hash = prompt_hash_for(prompt_template_text)
        judge_mask = (cache_df["judge_model"] == judge_model) & (cache_df["prompt_hash"] == p_hash)

    verdicts = cache_df[judge_mask][["para_hash", "llm_equiv"]]
    if verdicts.empty:
        return paras.iloc[0:0]

    merged = paras.merge(verdicts, on="para_hash", how="inner")
    accepted = merged[merged["llm_equiv"] == True].drop(columns=["llm_equiv"])
    return accepted.drop_duplicates(subset=["idx", "para_hash"], keep="first").reset_index(drop=True)


def dedup_by_overlap(df: pd.DataFrame, overlap_max: float = 0.8) -> pd.DataFrame:
    """
    Greedy per-idx lexical-overlap dedup over an already-selected set of
    paraphrase rows -- the same accept-if-under-threshold-against-
    everything-already-kept algorithm `pipeline.paraphrase._accepted_candidates`
    applies at generation time, re-run here over an arbitrary DataFrame (e.g.
    a union of several judge_model views) instead of a fresh candidate list.

    On a `df` drawn entirely from ONE paraphrases_file() pool this is
    provably a no-op: every row in that pool was already checked pairwise
    against every row accepted before it for the same idx at accept time, so
    any subset of it -- including one obtained by filtering on judge
    verdicts -- is automatically still pairwise below `overlap_max`. It
    stops being a guaranteed no-op only once rows validated by DIFFERENT
    judge_model calls are unioned together (see cross_validated_paraphrases
    below) -- no generation-time process ever compared those rows to each
    other -- which is exactly the case this function exists to catch.

    Order is stable input order (whatever the caller already sorted by) --
    not resampled or shuffled -- so results are deterministic for a given
    input ordering. Requires `idx` and `paraphrased_question` columns;
    other columns pass through unchanged on kept rows.
    """
    if df.empty:
        return df
    kept_rows = []
    for _, group in df.groupby("idx", sort=False):
        kept_texts: List[str] = []
        for _, row in group.iterrows():
            text = row["paraphrased_question"]
            if all(lexical_overlap_ratio(text, ref) < overlap_max for ref in kept_texts):
                kept_rows.append(row)
                kept_texts.append(text)
    if not kept_rows:
        return df.iloc[0:0]
    return pd.DataFrame(kept_rows, columns=df.columns).reset_index(drop=True)


def cross_validated_paraphrases(
    rp, judge_models: Union[str, List[str]], prompt_name: Optional[str] = None, overlap_max: float = 0.8,
    require_different_generation_model: bool = True,
) -> pd.DataFrame:
    """
    Paraphrases accepted by `judge_models` -- a single judge_model string or
    a list of them (kept if validated by ANY of them, e.g. GSM8K's gpt-5 /
    gpt-5.4-mini stronger-judge backfill) -- AND, by default, not written by
    that same judge_model: a judge validating its own writing isn't
    independent validation.

    Composed entirely from the live views paraphrase.py itself already uses
    for its own quota resume -- no separate materialized file and no
    refilter_paraphrases.py CLI run required:

      1. accepted_paraphrases(rp, jm, prompt_name), per judge_model -- rows
         with llm_equiv==True under (jm, prompt_name) in the shared judge
         cache (io/judge_cache.py). prompt_name=None (default) means "under
         ANY prompt this judge_model has ever been asked with" -- see
         accepted_paraphrases's docstring; pass a specific prompt_name to
         pin to one prompt version instead.
      2. If require_different_generation_model (default True): restricted to
         resp_model != jm -- resp_model is the model that actually WROTE the
         candidate text (see io/paraphrase.py's _provenance_fields), a
         different column from gen_judge_model (whichever judge first
         accepted the row into the pool, which this function does NOT
         compare against -- a row re-judged and reconfirmed by the very same
         judge that already accepted it is not what this flag is about).
         Concretely: a paraphrase_topup.py-produced row typically has
         resp_model=gpt-4.1-mini (the writer) and gen_judge_model=
         gpt-5.4-mini (the judge that accepted it) -- resp_model != "gpt-
         5.4-mini" holds, so it's INCLUDED in
         cross_validated_paraphrases(rp, "gpt-5.4-mini"): gpt-5.4-mini judged
         it, but never wrote it.
      3. Set require_different_generation_model=False to drop that check
         entirely -- "everything this judge currently accepts," full stop,
         even paraphrases it wrote itself. Named "cross_validated" for the
         True (default) case; with False this is really just
         accepted_paraphrases() unioned across judge_models, so name your
         caller's variable accordingly.

    The union (when multiple judge_models are given) is then re-verified
    with dedup_by_overlap -- see its docstring for why that's provably a
    no-op within one judge_model's accepted_paraphrases() view, but not once
    rows from different judge_models get unioned together.

    require_different_generation_model=True on a pool with no `resp_model`
    column (predates that provenance column) can't answer "who wrote this"
    and contributes nothing for that judge_model, rather than guessing.
    """
    if isinstance(judge_models, str):
        judge_models = [judge_models]
    frames = []
    for jm in judge_models:
        accepted = accepted_paraphrases(rp, jm, prompt_name)
        if accepted.empty:
            continue
        if require_different_generation_model:
            if "resp_model" not in accepted.columns:
                continue
            accepted = accepted[accepted["resp_model"] != jm]
        frames.append(accepted)
    if not frames:
        return pd.DataFrame(columns=_ACCEPTED_COLUMNS)
    union = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["idx", "para_hash"], keep="first"
    )
    return dedup_by_overlap(union, overlap_max=overlap_max)
