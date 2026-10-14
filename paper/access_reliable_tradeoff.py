"""
Access@k / Reliable@k trade-off at (nearly) equal mean accessibility -- numbers for the Section-3 proposition and its
RQ1 illustration. Per-inquiry unbiased estimators from c successes among m formulations (k <= m):
    Reliable@k = C(c,k)/C(m,k),   Access@k = 1 - C(m-c,k)/C(m,k).
  featured   BFCL GPT-5.4-Mini vs Qwen-3 1.7B (Fig. 3 crossing pair): Access@k / Reliable@k for k in KS, paired bootstrap
             95% CIs over inquiries, number of sign changes of R_a(tau) - R_b(tau).
  pairs      all model pairs with |d mu| <= MU_PP on the five non-SimpleQA benchmarks: opposite Access@5 / Reliable@5
             order, and the subset where both gaps >= BIG_PP (with bootstrap CIs).
Writes rq1_figs/figs/v4/access_reliable_tradeoff.json.
Usage: python paper/access_reliable_tradeoff.py  (or: bash paper/reproduce.sh)
"""
import itertools
import json
from math import comb

import numpy as np

import rq1_render as R

KS = [1, 2, 3, 5, 10]
K_PAIR = 5
MU_PP, BIG_PP = 0.02, 0.02
N_BOOT = 2000
RNG = np.random.default_rng(0)
DSS = ["TruthfulQA", "gsm8k", "math_500", "mmlu", "bfcl"]


def per_inquiry(A, m, k):
    c, m = np.rint(A * m).astype(int), m.astype(int)
    rel = np.array([comb(ci, k) / comb(mi, k) for ci, mi in zip(c, m)])
    acc = np.array([1 - comb(mi - ci, k) / comb(mi, k) for ci, mi in zip(c, m)])
    return acc, rel


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def gaps(Aa, Ab, m, k, idx):
    aa, ra = per_inquiry(Aa, m, k)
    ab, rb = per_inquiry(Ab, m, k)
    da, dr = aa - ab, ra - rb
    return (float(da.mean()), ci(da[idx].mean(1)), float(dr.mean()), ci(dr[idx].mean(1)))


def crossings(Aa, Ab):
    tau = np.linspace(0, 1, 501)[1:]
    d = np.sign((Aa[None] >= tau[:, None] - 1e-9).mean(1) - (Ab[None] >= tau[:, None] - 1e-9).mean(1))
    d = d[d != 0]
    return int((np.diff(d) != 0).sum())


def main():
    store, met = R.load()
    out = {"k_pair": K_PAIR, "mu_window_pp": 100 * MU_PP, "big_pp": 100 * BIG_PP, "n_boot": N_BOOT}
    s = store["bfcl"]
    nm = {R.name(x): x for x in s["models"]}
    a, b = nm["GPT-5.4-Mini"], nm["Qwen-3 1.7B"]
    m = np.asarray(s["n_para"], float)
    idx = RNG.integers(0, len(m), (N_BOOT, len(m)))
    feat = {"pair": ["GPT-5.4-Mini", "Qwen-3 1.7B"], "mu": [float(s["A"][a].mean()), float(s["A"][b].mean())],
            "crossings": crossings(s["A"][a], s["A"][b]), "k": {}}
    for k in KS:
        aa, ra = per_inquiry(s["A"][a], m, k)
        ab, rb = per_inquiry(s["A"][b], m, k)
        g = gaps(s["A"][a], s["A"][b], m, k, idx)
        feat["k"][k] = {"access": [float(aa.mean()), float(ab.mean())], "reliable": [float(ra.mean()), float(rb.mean())],
                        "d_access": g[0], "d_access_CI": g[1], "d_reliable": g[2], "d_reliable_CI": g[3]}
    out["featured_bfcl"] = feat

    per, big = {}, []
    for ds in DSS:
        s = store[ds]
        m, A = np.asarray(s["n_para"], float), s["A"]
        idx = RNG.integers(0, len(m), (N_BOOT, len(m)))
        n = o = 0
        for x, y in itertools.combinations(s["models"], 2):
            if abs(A[x].mean() - A[y].mean()) > MU_PP + 1e-9:
                continue
            n += 1
            g = gaps(A[x], A[y], m, K_PAIR, idx)
            if g[0] * g[2] >= 0:
                continue
            o += 1
            if min(abs(g[0]), abs(g[2])) >= BIG_PP:
                hiR, loR = (x, y) if g[2] > 0 else (y, x)   # orient: first model = higher Reliable@k
                sgn = 1 if g[2] > 0 else -1
                big.append({"dataset": R.TITLES[ds], "higher_reliable": R.name(hiR), "higher_access": R.name(loR),
                            "d_mu": float(A[hiR].mean() - A[loR].mean()),
                            "d_reliable": sgn * g[2], "d_reliable_CI": sorted([sgn * v for v in g[3]]),
                            "d_access": sgn * g[0], "d_access_CI": sorted([sgn * v for v in g[1]]),
                            "both_CI_exclude_0": bool(min(sgn * v for v in g[3]) > 0 and max(sgn * v for v in g[1]) < 0),
                            "crossings": crossings(A[hiR], A[loR])})
        per[R.TITLES[ds]] = {"pairs": n, "opposite_order": o}
    out["per_benchmark"] = per
    out["pooled"] = {"pairs": sum(v["pairs"] for v in per.values()),
                     "opposite_order": sum(v["opposite_order"] for v in per.values()), "large": len(big),
                     "large_both_significant": sum(r["both_CI_exclude_0"] for r in big)}
    out["large_cases"] = big
    path = R.OUT / "access_reliable_tradeoff.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("featured_bfcl", "per_benchmark", "pooled")}, indent=1))
    for r in big:
        print(r["dataset"], r["higher_reliable"], "vs", r["higher_access"], "dmu %.3f" % r["d_mu"],
              "dRel %.3f %s" % (r["d_reliable"], np.round(r["d_reliable_CI"], 3)),
              "dAcc %.3f %s" % (r["d_access"], np.round(r["d_access_CI"], 3)), "sig", r["both_CI_exclude_0"],
              "cross", r["crossings"])
    print("wrote", path)


if __name__ == "__main__":
    main()
