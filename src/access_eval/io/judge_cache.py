"""
Judge-verdict cache for paraphrase (equivalence) judging: one row per verdict, cache-only-grows, key =
(para_hash, judge_model, prompt_hash). `prompt_hash` is over the resolved
prompt TEMPLATE TEXT, not its name, so a silent prompt-file edit is a new
bucket, not aliased under the old name.
"""
from typing import Dict, List

import pandas as pd

from access_eval.core.paths import RunPaths
from access_eval.io.identity import compute_content_hash

CACHE_COLUMNS = [
    "para_hash", "original_question", "paraphrased_question",
    "judge_model", "prompt_name", "prompt_hash", "llm_raw", "llm_equiv",
]


def prompt_hash_for(prompt_template_text: str) -> str:
    return compute_content_hash(prompt_template_text)


def load_judge_cache(rp: RunPaths) -> pd.DataFrame:
    """Load the judge-verdict cache for this dataset/experiment, or an empty
    frame with the right columns if none exists yet."""
    path = rp.paraphrase_judge_cache_file()
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=CACHE_COLUMNS)


def append_judge_verdicts(rp: RunPaths, new_rows: List[Dict]) -> pd.DataFrame:
    """
    Append newly-judged rows (each a dict with CACHE_COLUMNS keys) to the cache
    on disk and return the full, updated cache. Rows already present (by key)
    are not duplicated, but existing rows are never dropped -- append-only-grows,
    matching the gencache/judgecache convention.
    """
    if not new_rows:
        return load_judge_cache(rp)

    existing = load_judge_cache(rp)
    incoming = pd.DataFrame(new_rows, columns=CACHE_COLUMNS)
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.drop_duplicates(subset=["para_hash", "judge_model", "prompt_hash"], keep="first")

    path = rp.paraphrase_judge_cache_file()
    combined.to_csv(path, index=False)
    return combined
