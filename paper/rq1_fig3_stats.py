"""
RQ1 Fig 3 (reliability curves) -- numbers for the paper text, with 95% bootstrap CIs over inquiries.

  featured   the plotted near-tied group per MAIN_FIG3 benchmark: acc / R / L ranges, paired-bootstrap CI of the R and
             L gaps between the most- and least-reliable model, and any L-vs-R order crossing inside the group.
  pairs      every model pair with |d acc| <= PAIR_PP (rq1_render.near_tied_pairs), per benchmark and pooled over all
             six: median d R (oriented more- minus less-reliable), share with |d R| > |d acc|, share with |d R| >= 5 pp,
             share whose paired CI excludes 0, share whose L order is opposite to their R order (curves cross). Also pooled without SimpleQA (92
             near-zero-R pairs dominate the six-benchmark pool). share_dR_gt_dacc CI is biased (|d acc| inflates
             under resampling) -- quote it as a count.
             CIs: resample inquiries within each benchmark (same draw for every model), pair set held fixed.
Writes rq1_figs/figs/<VERSION>/fig3_stats.json.
Usage: python paper/rq1_fig3_stats.py  (or: bash paper/reproduce.sh)
"""
import json

import numpy as np
import pandas as pd

import rq1_render as R
import rq1_selection_pass as SP

N_BOOT = 2000
RNG = np.random.default_rng(0)


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def pair_stats(d_acc, d_R, d_L):
    return {"median_dR": np.median(d_R, -1), "share_dR_gt_dacc": (d_R > d_acc).mean(-1),
            "share_dR_ge5pp": (d_R >= 0.05 - 1e-9).mean(-1), "share_cross": (d_L < -1e-9).mean(-1)}


def boot_pairs(store, pairs):
    """Per benchmark: (point arrays, bootstrap arrays [N_BOOT, n_pairs]) for d_acc, d_R, d_L (hi - lo, fixed pairs)."""
    out = {}
    for ds, p in pairs.groupby("dataset", sort=False):
        A, Y, n = store[ds]["A"], store[ds]["Y"], len(store[ds]["idx"])
        idx = RNG.integers(0, n, (N_BOOT, n))
        hi, lo = p.more_reliable.tolist(), p.less_reliable.tolist()
        acc = {m: Y[m][idx].mean(1) for m in set(hi + lo)}
        Rb = {m: (A[m][idx] >= 1).mean(1) for m in set(hi + lo)}
        Lb = {m: (A[m][idx] > 0).mean(1) for m in set(hi + lo)}
        b = [np.stack([np.abs(acc[h] - acc[l]) for h, l in zip(hi, lo)], 1),
             np.stack([Rb[h] - Rb[l] for h, l in zip(hi, lo)], 1), np.stack([Lb[h] - Lb[l] for h, l in zip(hi, lo)], 1)]
        out[ds] = ([p.d_acc.to_numpy(), p.d_R.to_numpy(), p.d_L.to_numpy()], b)
    return out


def summarize(point, boot, sig):
    ps, bs = pair_stats(*point), pair_stats(*boot)
    res = {"n_pairs": int(len(point[0])), "share_significant": float(sig.mean()),
           "median_dacc": float(np.median(point[0]))}
    for k in ps:
        res[k] = float(ps[k])
        res[k + "_CI"] = ci(bs[k])
    return res


def featured(store, met, ds):
    cand = SP.fig3_candidates(met[ds], store[ds]["A"], top=1).iloc[0]
    df = met[ds].loc[cand["models"]]
    A, n = store[ds]["A"], len(store[ds]["idx"])
    hi, lo = df["R"].idxmax(), df["R"].idxmin()
    idx = RNG.integers(0, n, (N_BOOT, n))
    dR = (A[hi][idx] >= 1).mean(1) - (A[lo][idx] >= 1).mean(1)
    dL = (A[hi][idx] > 0).mean(1) - (A[lo][idx] > 0).mean(1)
    crossings = []   # pairs inside the group whose L order is opposite to their R order
    for m1 in df.index:
        for m2 in df.index:
            if df.loc[m1, "R"] > df.loc[m2, "R"] and df.loc[m1, "L"] < df.loc[m2, "L"]:
                d_R = (A[m1][idx] >= 1).mean(1) - (A[m2][idx] >= 1).mean(1)
                d_L = (A[m1][idx] > 0).mean(1) - (A[m2][idx] > 0).mean(1)
                crossings.append({"higher_R": R.name(m1), "lower_R": R.name(m2),
                                  "dR": float(df.loc[m1, "R"] - df.loc[m2, "R"]), "dR_CI": ci(d_R),
                                  "dL": float(df.loc[m1, "L"] - df.loc[m2, "L"]), "dL_CI": ci(d_L)})
    return {"models": {R.name(m): {k: float(df.loc[m, k]) for k in ("acc", "L", "R", "mean_A")} for m in df.index},
            "acc_range_pp": float(100 * (df["acc"].max() - df["acc"].min())),
            "most_reliable": R.name(hi), "least_reliable": R.name(lo),
            "dR": float(df.loc[hi, "R"] - df.loc[lo, "R"]), "dR_CI": ci(dR),
            "dL": float(df.loc[hi, "L"] - df.loc[lo, "L"]), "dL_CI": ci(dL),
            "L_range": float(df["L"].max() - df["L"].min()), "crossings": crossings}


def main():
    store, met = R.load()
    pairs = pd.concat([R.near_tied_pairs(store, met, ds) for ds in R.ALL], ignore_index=True)
    bp = boot_pairs(store, pairs)
    per = {R.TITLES[ds]: summarize(pt, b, pairs[pairs.dataset == ds].significant.to_numpy())
           for ds, (pt, b) in bp.items()}
    for label, keep in (("All six (pooled)", list(bp)), ("Five, excl. SimpleQA (pooled)", [d for d in bp if d != "SimpleQA"])):
        pool_pt = [np.concatenate([bp[d][0][i] for d in keep]) for i in range(3)]
        pool_b = [np.concatenate([bp[d][1][i] for d in keep], 1) for i in range(3)]
        per[label] = summarize(pool_pt, pool_b, pairs[pairs.dataset.isin(keep)].significant.to_numpy())
    out = {"pair_window_pp": R.PAIR_PP * 100, "n_boot": N_BOOT, "pairs": per,
           "featured": {R.TITLES[ds]: featured(store, met, ds) for ds in R.MAIN_FIG3}}
    path = R.OUT / "fig3_stats.json"
    path.write_text(json.dumps(out, indent=2))
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print(pd.DataFrame(per).T.drop(columns=[c for c in pd.DataFrame(per).T.columns if c.endswith("_CI")]))
        for k, v in per.items():
            print(k, {c: [round(x, 3) for x in v[c]] for c in v if c.endswith("_CI")})
    print(json.dumps(out["featured"], indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
