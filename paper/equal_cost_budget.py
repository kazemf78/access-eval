"""
Equal-cost profile estimation on the paper's RQ1 data (rq1_figs/rq1_per_question.pkl: frozen inquiries, TruthfulQA
majority-of-3 labels, math fallback grader, non-thinking models).

Question: at the cost of ONE conventional run (N calls on N inquiries), evaluate n = N // K random inquiries with K
random formulations each. Are the profile quantities as recoverable as accuracy itself?
Reference (full data): mean_A = mean c/m; Reliable_K = mean C(c,K)/C(m,K); Latent_K = mean 1 - C(m-c,K)/C(m,K)
(the exact expectations over K-subsets of each inquiry's m formulations), acc = mean canonical outcome.
Cheap estimate: n inquiries (same subset for all models), per inquiry c_K ~ Hypergeometric(m, c, K):
mean c_K/K, share c_K == K, share c_K > 0; accuracy on the same n inquiries (the conventional-run comparison, one
call per inquiry, i.e. n calls -- deliberately given the SAME n so the comparison is at equal inquiry count).
Criterion: |estimate - reference| <= TOL in >= 95% of R replicates. Reports, per benchmark x K, the share of models
passing for each quantity. Writes $PAPER_REPRO_DIR/equal_cost_budget/summary.csv.
Usage: python paper/equal_cost_budget.py  (or: bash paper/reproduce.sh)
"""
import pickle
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

import rq1_render as Rr

ROOT = Path(__file__).resolve().parents[1]
import paper_paths as PP  # noqa: E402  (PAPER_REPRO_DIR-aware output locations; see REPRODUCE.md)
OUT = PP.out_dir(None, "equal_cost_budget")
KS = [2, 3, 4, 5]
R = 2000
TOL, TARGET = 0.05, 0.95
RNG = np.random.default_rng(0)


def main():
    OUT.mkdir(exist_ok=True)
    store = pickle.load(open(PP.rq1_store(), "rb"))
    rows = []
    for ds, s in store.items():
        m = np.asarray(s["n_para"]).astype(int)
        N = len(m)
        for K in KS:
            n = N // K
            idx = np.array([RNG.choice(N, n, replace=False) for _ in range(R)])      # same subsets for all models
            for mod in s["models"]:
                c = np.rint(s["A"][mod] * m).astype(int)
                y = s["Y"][mod]
                ref = {"acc": y.mean(), "mean_A": (c / m).mean(),
                       "reliable": np.mean([comb(ci, K) / comb(mi, K) for ci, mi in zip(c, m)]),
                       "latent": np.mean([1 - comb(mi - ci, K) / comb(mi, K) for ci, mi in zip(c, m)])}
                cK = RNG.hypergeometric(c[idx], m[idx] - c[idx], K)                  # [R, n]
                est = {"acc": y[idx].mean(1), "mean_A": (cK / K).mean(1), "reliable": (cK == K).mean(1),
                       "latent": (cK > 0).mean(1)}
                for q in ref:
                    rows.append({"dataset": Rr.TITLES[ds], "K": K, "N": N, "n": n, "model": Rr.name(mod), "quantity": q,
                                 "p_within": float((np.abs(est[q] - ref[q]) <= TOL).mean())})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "per_model.csv", index=False)
    summ = d.assign(ok=d.p_within >= TARGET).groupby(["dataset", "K", "n", "quantity"], sort=False).ok.mean().unstack()
    summ = summ[["acc", "mean_A", "latent", "reliable"]]
    summ.to_csv(OUT / "summary.csv")
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(summ)


if __name__ == "__main__":
    main()
