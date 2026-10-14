"""Formulation-level flip rate (share of formulations whose correctness differs) for the RQ3 scaling-step and
native-thinking model changes, on the RQ1 inquiry sets, for comparison with the greedy rerun baseline
(null_gsm8k.json formulation_flip_rate). Output: $PAPER_REPRO_DIR/rerun_null/formulation_flip_model_changes.csv.
Usage: python paper/rerun_null/formulation_flip_model_changes.py  (or: bash paper/reproduce.sh)
"""
import pickle, sys, numpy as np, pandas as pd
import rq1_selection_pass as SP, knowledge_state_bars as KB, rq3_interventions as Q, paper_paths as PP  # noqa: E401,F401
store = pickle.load(open(PP.rq1_store(), "rb"))
paths = None
rows = []
for ds in ["gsm8k", "math_500", "mmlu", "bfcl", "TruthfulQA"]:
    agg = KB.load(ds, paths)
    idx = set(store[ds]["idx"])
    agg = agg[agg.idx.isin(idx) & (agg.paraphrased_question != agg.original_question)]   # paraphrases only, as the rerun
    agg = agg.assign(y=(agg.verdict == "correct").astype(int))
    piv = agg.pivot_table(index=["idx", "paraphrased_question"], columns="model", values="y", aggfunc="first")
    pairs = []
    for fam, lad in Q.LADDERS.items():
        ms = [m for m in lad if m in store[ds]["A"]]
        pairs += [("scaling-step", a, b) for a, b in zip(ms, ms[1:])]
    for s in Q.SIZES:
        b, t = f"Qwen/Qwen3-{s}", f"Qwen/Qwen3-{s}[with_thinking]"
        if b in piv and t in piv: pairs.append(("thinking", b, t))
    for cell, a, b in pairs:
        sub = piv[[a, b]].dropna()
        g = sub.groupby(level=0)
        sub = sub[g[a].transform("size") >= 5]
        d = (sub[a] != sub[b])
        n_inq = sub.index.get_level_values(0).nunique()
        rows.append({"dataset": ds, "cell": cell, "pair": f"{a} -> {b}", "n_form": len(sub), "n_inq": n_inq,
                     "form_flip": d.mean(), "up": ((sub[b] > sub[a])).mean(), "down": ((sub[b] < sub[a])).mean()})
r = pd.DataFrame(rows)
dest = PP.out_dir(None, "rerun_null")
dest.mkdir(parents=True, exist_ok=True)
r.to_csv(dest / "formulation_flip_model_changes.csv", index=False)
print(r.groupby(["dataset", "cell"])[["form_flip", "up", "down"]].agg(["mean", "median", "min"]).round(3).to_string())
print(r[r.dataset == "gsm8k"].round(3).to_string())
