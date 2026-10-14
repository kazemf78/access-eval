"""
Inquiry-level accessibility profile of one model on one formulation set (paper definitions, rq1_selection_pass.py).

Per inquiry q: y_q = correctness of the ORIGINAL wording (a separate observation), m_q = number of graded PARAPHRASES
(the original is not counted), c_q = correct paraphrases, A_q = c_q / m_q. Inquiries with m_q < min_para are
dropped (paper: 5).

  acc          P(y = 1)                         single-prompt (original-wording) accuracy
  mean_A (mu)  E[A]                             expected accuracy over formulations
  R            P(A = 1)                         reliable share (correct on every tested paraphrase)
  L            P(A > 0)                         latent share (correct on at least one)
  B            L - R                            brittle share
  reversal     P(y=1, A<1/2) + P(y=0, A>1/2)    single prompt contradicts the paraphrase majority (the Phase I
                                                plan's "mismatch rate"; A = 1/2 is no reversal)
  access_at_k  E[1 - C(m-c,k)/C(m,k)]           k random paraphrases, at least one correct (exact over m_q)
  reliable_at_k E[C(c,k)/C(m,k)]                k random paraphrases, all correct
"""
from math import comb

import numpy as np
import pandas as pd

KS = (2, 3, 4, 5)


def per_inquiry(graded: pd.DataFrame, min_para: int = 5) -> pd.DataFrame:
    """graded: one row per formulation with columns idx, is_original (bool/int), correct (bool; NaN = excluded).
    Returns idx, y, c, m for inquiries with a graded original and >= min_para graded paraphrases."""
    g = graded.dropna(subset=["correct"])
    orig = g[g["is_original"].astype(bool)].groupby("idx")["correct"].first().astype(int)
    para = g[~g["is_original"].astype(bool)].groupby("idx")["correct"].agg(["sum", "count"])
    t = pd.DataFrame({"y": orig}).join(para.rename(columns={"sum": "c", "count": "m"}), how="inner")
    t = t[t["m"] >= min_para].astype(int)
    return t.reset_index().rename(columns={"index": "idx"})


def access_reliable_at_k(c, m, k):
    """Exact per-inquiry Access@k and Reliable@k (expectations over k-subsets of the m paraphrases)."""
    rel = np.array([comb(ci, k) / comb(mi, k) for ci, mi in zip(c, m)])
    acc = np.array([1 - comb(mi - ci, k) / comb(mi, k) for ci, mi in zip(c, m)])
    return acc, rel


def profile_metrics(y, c, m, ks=KS) -> dict:
    y, c, m = (np.asarray(v, dtype=int) for v in (y, c, m))
    a = c / m
    R, L = (a >= 1).mean(), (a > 0).mean()
    so, su = ((y == 1) & (a < 0.5)).mean(), ((y == 0) & (a > 0.5)).mean()
    out = {"n_inquiries": int(len(y)), "n_paraphrases": int(m.sum()), "acc": y.mean(), "mean_A": a.mean(),
           "R": R, "L": L, "B": L - R, "reversal": so + su, "sev_over": so, "sev_under": su,
           "over_gap": y.mean() - R, "under_gap": L - y.mean()}
    for k in ks:
        if (m >= k).all():
            ak, rk = access_reliable_at_k(c, m, k)
            out[f"access_at_{k}"], out[f"reliable_at_{k}"] = ak.mean(), rk.mean()
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def bootstrap_ci(y, c, m, B=2000, seed=0, alpha=0.05) -> dict:
    """Paired inquiry-bootstrap percentile CIs of the profile quantities and the gaps the claims test."""
    y, c, m = (np.asarray(v, dtype=int) for v in (y, c, m))
    a = c / m
    idx = np.random.default_rng(seed).integers(0, len(y), size=(B, len(y)))
    Y, A = y[idx], a[idx]
    acc, R, L = Y.mean(1), (A >= 1).mean(1), (A > 0).mean(1)
    rev = ((Y == 1) & (A < 0.5)).mean(1) + ((Y == 0) & (A > 0.5)).mean(1)
    draws = {"acc": acc, "mean_A": A.mean(1), "R": R, "L": L, "B": L - R, "reversal": rev,
             "under_gap": L - acc, "over_gap": acc - R}
    out = {}
    for k, v in draws.items():
        out[f"{k}_ci_lo"], out[f"{k}_ci_hi"] = (float(x) for x in np.quantile(v, [alpha / 2, 1 - alpha / 2]))
    return out
