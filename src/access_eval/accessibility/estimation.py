"""
Budget-limited estimation of the accessibility profile, against the full evaluation of the same model as reference.

Costs, in target-model calls (N inquiries, m_q paraphrases of inquiry q -- ACTUAL counts, never approximations):
  conventional   C_conv = N                 one call per inquiry, original wording (a standard benchmark run)
  full           C_full = sum_q (m_q + 1)   every paraphrase plus the original: the reference evaluation
  budget         B = beta * C_conv          every design below spends at most B calls

Designs (each over `trials` random repetitions; error = estimate - full-run reference):
  conventional   min(B, N) random inquiries, original wording          -> acc
  subsample      random inquiries with ALL their formulations, added    -> acc, mean_A, R, L, B, reversal
                 while the running cost sum(m_q + 1) stays <= B          (the Phase I plan)
  kform (K)      min(floor(B/K), N) random inquiries x K paraphrases    -> mean_A, Access@K, Reliable@K, and acc on
                 (successes ~ Hypergeometric(m_q, c_q, K))                 the same inquiries (paper Appendix J)
                 references: E[c/m], E[C(c,K)/C(m,K)], E[1-C(m-c,K)/C(m,K)] over all N inquiries -- the K-level
                 quantities, not R and L.
At beta = 1, kform reproduces Appendix J (k_formulation_estimates is the exact estimator of
paper/equal_cost_budget.py; verified on the paper's data).
Criterion: |error| <= tol in >= target of the trials ("meets"); also mean |error| and the 2.5-97.5% error range.
"""
import numpy as np

from access_eval.accessibility.metrics import access_reliable_at_k


def references(y, c, m, K=None):
    y, c, m = (np.asarray(v, dtype=int) for v in (y, c, m))
    a = c / m
    ref = {"acc": y.mean(), "mean_A": a.mean(), "R": (a >= 1).mean(), "L": (a > 0).mean()}
    ref["B"] = ref["L"] - ref["R"]
    ref["reversal"] = (((y == 1) & (a < 0.5)) | ((y == 0) & (a > 0.5))).mean()
    if K is not None:
        acc_k, rel_k = access_reliable_at_k(c, m, K)
        ref["latent"], ref["reliable"] = acc_k.mean(), rel_k.mean()
    return ref


def k_formulation_estimates(y, c, m, K, idx, rng):
    """Appendix J estimator. idx: [trials, n] inquiry subsets (supplied by the caller so a verification can
    reproduce the paper's draw order). Returns per-trial estimates of acc (same n inquiries), mean_A, Reliable@K
    ('reliable') and Access@K ('latent')."""
    y, c, m = (np.asarray(v, dtype=int) for v in (y, c, m))
    cK = rng.hypergeometric(c[idx], m[idx] - c[idx], K)
    return {"acc": y[idx].mean(1), "mean_A": (cK / K).mean(1), "reliable": (cK == K).mean(1), "latent": (cK > 0).mean(1)}


def subsample_estimates(y, c, m, budget, trials, rng):
    y, c, m = (np.asarray(v, dtype=int) for v in (y, c, m))
    a, cost = c / m, m + 1
    names = ("acc", "mean_A", "R", "L", "B", "reversal")
    est = {k: np.empty(trials) for k in names}
    used = np.empty(trials)
    for t in range(trials):
        order = rng.permutation(len(y))
        keep = order[np.cumsum(cost[order]) <= budget]
        if len(keep) == 0:
            for k in names:
                est[k][t] = np.nan
            used[t] = 0
            continue
        yy, aa = y[keep], a[keep]
        R, L = (aa >= 1).mean(), (aa > 0).mean()
        est["acc"][t], est["mean_A"][t], est["R"][t], est["L"][t], est["B"][t] = yy.mean(), aa.mean(), R, L, L - R
        est["reversal"][t] = (((yy == 1) & (aa < 0.5)) | ((yy == 0) & (aa > 0.5))).mean()
        used[t] = cost[keep].sum()
    return est, used


def _summarize(est, ref, tol, target):
    out = {}
    for k, v in est.items():
        err = v - ref[k]
        ok = ~np.isnan(err)
        if not ok.any():
            out[k] = {"ref": float(ref[k]), "p_within": 0.0, "meets": False, "mean_abs_err": None,
                      "err_lo": None, "err_hi": None}
            continue
        e = err[ok]
        p = float((np.abs(err) <= tol).mean())   # an empty (NaN) trial counts as a miss
        out[k] = {"ref": float(ref[k]), "p_within": p, "meets": bool(p >= target),
                  "mean_abs_err": float(np.abs(e).mean()),
                  "err_lo": float(np.quantile(e, 0.025)), "err_hi": float(np.quantile(e, 0.975))}
    return out


def budget_study(y, c, m, betas=(0.25, 0.5, 1.0), ks=(2, 3), trials=2000, tol=0.05, target=0.95, seed=0):
    """All designs at every budget. Returns a nested dict (see module docstring) with the cost accounting."""
    y, c, m = (np.asarray(v, dtype=int) for v in (y, c, m))
    rng = np.random.default_rng(seed)
    N = len(y)
    out = {"N": N, "cost_conventional": N, "cost_full": int((m + 1).sum()), "tol": tol, "target": target,
           "trials": trials, "budgets": {}}
    base_ref = references(y, c, m)
    for beta in betas:
        B = int(np.floor(beta * N))
        res = {"beta": beta, "budget_calls": B, "designs": {}}
        # conventional: B inquiries, original wording
        n = min(B, N)
        idx = np.array([rng.choice(N, n, replace=False) for _ in range(trials)])
        res["designs"]["conventional"] = {"n_inquiries": n, "calls": n,
                                          "metrics": _summarize({"acc": y[idx].mean(1)}, base_ref, tol, target)}
        # subsample: whole inquiries until the budget is spent
        est, used = subsample_estimates(y, c, m, B, trials, rng)
        res["designs"]["subsample"] = {"calls_mean": float(used.mean()), "calls_max": float(used.max()),
                                       "metrics": _summarize(est, base_ref, tol, target)}
        for K in ks:
            if (m < K).any():
                continue
            n = min(B // K, N)   # above B = K*N every inquiry is already used (calls < B, reported)
            if n < 2:
                continue
            idx = np.array([rng.choice(N, n, replace=False) for _ in range(trials)])
            est = k_formulation_estimates(y, c, m, K, idx, rng)
            res["designs"][f"kform_K{K}"] = {"n_inquiries": n, "calls": n * K,
                                             "metrics": _summarize(est, references(y, c, m, K), tol, target)}
        out["budgets"][budget_key(beta)] = res
    return out


def budget_key(beta) -> str:
    """'beta_0p25', 'beta_1', 'beta_4' -- no '.', which result loaders read as nesting."""
    return f"beta_{beta:g}".replace(".", "p")


def flatten(study) -> dict:
    """Flat key/value view for result loaders: budget, design and metric joined with '__'."""
    flat = {k: v for k, v in study.items() if not isinstance(v, dict)}
    for bname, b in study["budgets"].items():
        flat[f"{bname}__budget_calls"] = b["budget_calls"]
        for dname, d in b["designs"].items():
            for ck in ("n_inquiries", "calls", "calls_mean", "calls_max"):
                if ck in d:
                    flat[f"{bname}__{dname}__{ck}"] = d[ck]
            for mname, s in d["metrics"].items():
                for sk, sv in s.items():
                    flat[f"{bname}__{dname}__{mname}__{sk}"] = sv
    return flat
