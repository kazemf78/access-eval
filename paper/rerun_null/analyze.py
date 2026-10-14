"""
RQ3 noise baseline: for each rerun model, compare the frozen answers (run 1, as used in RQ1/RQ3) with the rerun
(run 2) on the SAME GSM8K inquiries and formulations, graded with the paper's math policy (fallback grader).
Under "no real change", any movement of A(q) between the two runs is run-to-run nondeterminism.
Reports per model: formulation-level flip rate, share of inquiries whose A moves (up/down), canonical-label flip
rate, and the same "hidden share" statistic used in RQ3; plus the RQ3 GSM8K cell values for comparison.
Usage: python paper/rerun_null/analyze.py  (or: bash paper/reproduce.sh)
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import paper_paths as PP
import rq3_interventions as Q3
import knowledge_state_bars as KB

RUN = PP.DATA / "rerun_null"   # rerun answers graded with the paper's math policy (fallback grader) + the pool keys


def main():
    store = pickle.load(open(PP.rq1_store(), "rb"))["gsm8k"]
    pool = pd.read_parquet(RUN / "gsm8k_pool_keys.parquet")
    keys = set(zip(pool.idx, pool.paraphrased_question))
    run1 = KB.load("gsm8k")
    run2 = pd.read_parquet(RUN / "gsm8k_rerun_graded.parquet")
    out = {}
    for m in sorted(run2["model"].unique()):
        a, b = run1[run1.model == m], run2[run2.model == m]
        a = a[a.idx.isin(store["idx"])]
        b = b[b.idx.isin(store["idx"])]
        j = a.merge(b, on=["idx", "paraphrased_question"], suffixes=("_1", "_2"))
        j["c1"], j["c2"] = (j.verdict_1 == "correct").astype(float), (j.verdict_2 == "correct").astype(float)
        para = j[[(i, q) in keys for i, q in zip(j.idx, j.paraphrased_question)]]
        orig = j[j.paraphrased_question == j.original_question_1]
        A1, A2 = para.groupby("idx").c1.mean(), para.groupby("idx").c2.mean()
        Y1, Y2 = orig.set_index("idx").c1, orig.set_index("idx").c2
        idx = A1.index.intersection(Y1.index)
        dA, dY = (A2 - A1).loc[idx], (Y2 - Y1).loc[idx]
        moved = dA.abs() > 1e-9
        out[m] = {"n_inquiries": int(len(idx)), "n_formulations": int(len(para)),
                  "formulation_flip_rate": float((para.c1 != para.c2).mean()),
                  "acc_run1": float(Y1.loc[idx].mean()), "acc_run2": float(Y2.loc[idx].mean()),
                  "moved": float(moved.mean()), "up": float((dA > 1e-9).mean()), "down": float((dA < -1e-9).mean()),
                  "canon_flip": float((dY != 0).mean()),
                  "hidden_share": float(((dY == 0) & moved).sum() / max(moved.sum(), 1)),
                  "mean_abs_dA_moved": float(dA[moved].abs().mean()) if moved.any() else 0.0}
    rq3 = pd.read_csv(Q3.OUT / "rq3_pairs.csv")   # pairs identical in v6/v7 (v7 changed figures only)
    g = rq3[(rq3.dataset == "gsm8k") & (rq3.cell != "scaling-endpoint")]
    ref = g.groupby("cell")[["up", "down", "hidden_share"]].median()
    ref["moved"] = ref.up + ref.down
    res = {"null": out, "rq3_gsm8k_medians": ref.round(4).to_dict(orient="index")}
    dest = PP.out_dir(None, "rerun_null")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "null_gsm8k.json").write_text(json.dumps(res, indent=2))
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(pd.DataFrame(out).T)
        print(ref)


if __name__ == "__main__":
    main()
