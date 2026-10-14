"""
Benchmark tables and the common-substrate model selection shared by the paper scripts.

Each benchmark is one table (paper/data/graded/<benchmark>.parquet) with one row per (model, inquiry, formulation):
model, idx, original_question, paraphrased_question (== original_question for the original wording), verdict
("correct" / ...), n_judges, n_agree, gpt_agree. Grading behind each table:
  SimpleQA       4 LLM judges (a row is used when > 1/2 agree and the GPT judges agree with each other)
  TruthfulQA     3 GPT judges (gpt-3.5-turbo, gpt-4.1-mini, gpt-5.4-mini), binary majority ("maj3_binary")
  GSM8K/MATH-500 math_verify + fallback answer extractor
  MMLU           custom generative protocol, letter extraction
  BFCL           live_multiple, BFCL harness scoring
The answers and judge outputs behind these tables are in access-eval-data.

Model selection (paper figure policy): thinking-mode models excluded; every model family (OpenAI, Qwen, Meta, Google,
Mistral, Phi) present. Seed = the one-model-per-family combination with the largest common substrate (same inquiries
AND the same >= MIN_PARA paraphrases for every model); then greedily add the model that keeps the most common
inquiries, while >= MIN_Q remain.
"""
import itertools
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data" / "graded"
MIN_PARA = 5
MIN_PARA_OVERRIDE = {"TruthfulQA": 4}
TQA_RULE = "maj3_binary"
MIN_Q = 100
EXCLUDE_THINKING = True
LABELS = {"SimpleQA": "SimpleQA", "TruthfulQA": "TruthfulQA (5.4-mini+1 GPT)", "gsm8k": "GSM8K",
          "math_500": "MATH-500", "mmlu": "MMLU (custom-generative)", "bfcl": "BFCL (live_multiple)"}


def load(ds, paths=None):
    """The benchmark's graded-formulation table (`paths` is accepted for compatibility and ignored)."""
    if ds == "TruthfulQA" and TQA_RULE != "maj3_binary":
        raise ValueError("only the paper's TruthfulQA labels (maj3_binary) are shipped; rebuild other rules from raw")
    return pd.read_parquet(DATA / f"{ds}.parquet")


def build_filter1_index(agg: pd.DataFrame, min_agreement_fraction: float = 0.5):
    """Per (model, idx): has_orig (bool) and texts_ok (frozenset of paraphrase texts passing Filter 1: > 1/2 of the
    judges agree and the GPT judges agree with each other). Returns ({model: {idx: (has_orig, texts_ok)}}, all_idx)."""
    sub = agg.copy()
    sub["is_orig"] = sub["original_question"] == sub["paraphrased_question"]
    sub["passes_filter1"] = (sub["n_agree"] > sub["n_judges"] * min_agreement_fraction) & sub["gpt_agree"]
    orig_ok = sub[sub["is_orig"] & sub["passes_filter1"]]
    has_orig_by_model = {m: set(g["idx"]) for m, g in orig_ok.groupby("model")}
    para_ok = sub[~sub["is_orig"] & sub["passes_filter1"]]
    texts_by_model = {}
    for m, g in para_ok.groupby("model"):
        texts_by_model[m] = {idx: frozenset(gg["paraphrased_question"]) for idx, gg in g.groupby("idx")}
    all_idx = set(agg["idx"].unique())
    index = {}
    for m in agg["model"].unique():
        ho, tb = has_orig_by_model.get(m, set()), texts_by_model.get(m, {})
        index[m] = {idx: (idx in ho, tb.get(idx, frozenset())) for idx in all_idx}
    return index, all_idx


def family(m):
    m = m.lower()
    for key, fam in (("gpt", "OpenAI"), ("qwen", "Qwen"), ("llama", "Meta"), ("gemma", "Google"),
                     ("mistral", "Mistral"), ("phi", "Phi")):
        if key in m:
            return fam
    return "other"


def common_count(index, all_idx, models, k):
    """# inquiries where every model has a Filter-1 original and >= k identical Filter-1 paraphrases."""
    n = 0
    for idx in all_idx:
        shared = None
        for m in models:
            has_orig, texts = index[m][idx]
            shared = texts if shared is None else shared & texts
            if not has_orig or len(shared) < k:
                break
        else:
            n += 1
    return n


def select_models(agg, k):
    pool = sorted(m for m in agg["model"].unique() if not (EXCLUDE_THINKING and "[with_thinking]" in m))
    index, all_idx = build_filter1_index(agg[agg["model"].isin(pool)])
    fams = {}
    for m in pool:
        fams.setdefault(family(m), []).append(m)
    seed = max(itertools.product(*fams.values()), key=lambda c: common_count(index, all_idx, c, k))
    chosen, n = list(seed), common_count(index, all_idx, seed, k)
    rows = [{"N": len(chosen), "added_model": "+".join(seed), "n_included": n}]
    while True:
        cands = [m for m in pool if m not in chosen]
        if not cands:
            break
        best = max(cands, key=lambda m: common_count(index, all_idx, chosen + [m], k))
        nb = common_count(index, all_idx, chosen + [best], k)
        rows.append({"N": len(chosen) + 1, "added_model": best, "n_included": nb})
        if nb < MIN_Q:
            break
        chosen.append(best)
        n = nb
    return chosen, n, pd.DataFrame(rows), sorted(fams)


def short(m):
    return (m.split("/")[-1].replace("-Instruct", "").replace("[with_thinking]", " [think]")
            .replace("Meta-Llama-3-8B", "Llama-3-8B").replace("-v0.1", "").replace("_", "."))
