"""
RQ1 step 1 -- selection pass (no final figures). For every benchmark, on the same common substrate as the
knowledge-state figure (knowledge_state_bars.select_models: no thinking models, every family present, same questions
AND same >= k shared paraphrases for every model), compute per model:

  acc       = P(Y_orig = 1)                   (canonical wording)
  R         = P(A = 1),  L = P(A > 0),  B = L - R      A(q) = accuracy over the shared PARAPHRASES (canonical excluded)
  over_gap  = acc - R,   under_gap = L - acc
  sev_over  = P(Y_orig = 1, A < 0.5),  sev_under = P(Y_orig = 0, A > 0.5),  reversal = sev_over + sev_under
  (+ conditional versions sev_over|Y=1, sev_under|Y=0)

TruthfulQA: 3-GPT binary majority labels ("maj3_binary": gpt-3.5-turbo, gpt-4.1-mini, gpt-5.4-mini), >= 5 shared
paraphrases (user decision 2026-09-24 for the RQ1 figures -> 21 models / 471 questions).

Outputs ($PAPER_REPRO_DIR/rq1/): rq1_per_model_metrics.csv, rq1_tableA_benchmarks.csv, rq1_tableB_fig2_reps.csv,
rq1_tableC_fig3_candidates.csv, rq1_per_question.pkl (per-question A and Y_orig per model, for the figure steps).

Usage: python paper/rq1_selection_pass.py  (or: bash paper/reproduce.sh)
"""
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import knowledge_state_bars as KB  # noqa: E402
from access_eval.analysis.common_profiles import build_common_profiles  # noqa: E402
from access_eval.core.paths import ProjectPaths  # noqa: E402

KB.MIN_PARA_OVERRIDE = {}          # TruthfulQA now >= 5 like everything else
KB.TQA_RULE = "maj3_binary"         # user decision 2026-09-24: 3-GPT binary majority (never drops a row)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_paths as PP  # noqa: E402  (PAPER_REPRO_DIR-aware output locations; see REPRODUCE.md)
OUT = PP.rq1_dir()
# Outlier models removed per benchmark (user decision 2026-09-25) before the common substrate is built.
EXCLUDE_MODELS = {"mmlu": ["google/gemma-3-1b-it"], "bfcl": ["google/gemma-3-1b-it"]}
TIE_PP = 0.03                       # Table C: "near-tied" benchmark accuracy window
MIN_R_SPREAD = 0.05                 # Table C: minimum reliable-share spread to count as visibly different
N_BOOT = 2000
RNG = np.random.default_rng(0)


def per_question(agg, models, k):
    prof = build_common_profiles(agg, models)
    prof = prof[prof["n_para"] >= k]
    A = {m: prof[f"correct_frac_{m}"].to_numpy(float) for m in models}
    Y = {m: (prof[f"orig_label_{m}"] == "correct").to_numpy(float) for m in models}
    return prof.index.to_numpy(), A, Y, prof["n_para"].to_numpy(float)


def metrics(a, y):
    R, L = (a >= 1).mean(), (a > 0).mean()
    so, su = ((y == 1) & (a < 0.5)).mean(), ((y == 0) & (a > 0.5)).mean()
    return {"acc": y.mean(), "R": R, "B": L - R, "L": L, "mean_A": a.mean(),
            "over_gap": y.mean() - R, "under_gap": L - y.mean(),
            "sev_over": so, "sev_under": su, "reversal": so + su,
            "sev_over_cond": so / y.mean() if y.mean() > 0 else np.nan,
            "sev_under_cond": su / (1 - y.mean()) if y.mean() < 1 else np.nan,
            "acc_outside_RL": not (R - 1e-9 <= y.mean() <= L + 1e-9)}


def boot_diff_R(a1, a2):
    """Paired bootstrap 95% CI of R(model1) - R(model2) over the shared questions."""
    n = len(a1)
    idx = RNG.integers(0, n, (N_BOOT, n))
    d = (a1[idx] >= 1).mean(1) - (a2[idx] >= 1).mean(1)
    return np.percentile(d, [2.5, 97.5])


def pick_fig2(df):
    """>= median acc; highest sev_over and highest sev_under; if the same model wins both, it keeps the side where its
    lead over the runner-up is larger and the runner-up takes the other side."""
    pool = df[df["acc"] >= df["acc"].median()]
    so = pool.sort_values(["sev_over", "acc"], ascending=False)
    su = pool.sort_values(["sev_under", "acc"], ascending=False)
    o, u = so.index[0], su.index[0]
    note = ""
    if o == u:
        lead_o = so["sev_over"].iloc[0] - so["sev_over"].iloc[1]
        lead_u = su["sev_under"].iloc[0] - su["sev_under"].iloc[1]
        if lead_o >= lead_u:
            u, note = su.index[1], f"{KB.short(o)} won both; kept overstatement (lead {lead_o:.3f} vs {lead_u:.3f})"
        else:
            o, note = so.index[1], f"{KB.short(u)} won both; kept understatement (lead {lead_u:.3f} vs {lead_o:.3f})"
    return pool, o, u, note


def fig3_candidates(df, A, top=3):
    """All 2-4 model subsets whose acc lies within TIE_PP; score = R spread; keep non-dominated best few."""
    rows = []
    ms = df.sort_values("acc").index.tolist()
    for size in (2, 3, 4):
        for combo in itertools.combinations(ms, size):
            accs = df.loc[list(combo), "acc"]
            if accs.max() - accs.min() > TIE_PP + 1e-9:
                continue
            Rs = df.loc[list(combo), "R"]
            spread = Rs.max() - Rs.min()
            if spread < MIN_R_SPREAD:
                continue
            hi, lo = Rs.idxmax(), Rs.idxmin()
            ci = boot_diff_R(A[hi], A[lo])
            rows.append({"models": list(combo), "size": size, "acc_min": accs.min(), "acc_max": accs.max(),
                         "R_spread": spread, "most_reliable": hi, "least_reliable": lo,
                         "meanA_spread": df.loc[list(combo), "mean_A"].max() - df.loc[list(combo), "mean_A"].min(),
                         "L_spread": df.loc[list(combo), "L"].max() - df.loc[list(combo), "L"].min(),
                         "R_diff_CI_lo": ci[0], "R_diff_CI_hi": ci[1], "n_families": len({KB.family(m) for m in combo})})
    if not rows:
        return pd.DataFrame()
    c = pd.DataFrame(rows)
    # rank: R spread first, prefer more models/families and higher acc (user: mid/high performers more interesting)
    c["score"] = c["R_spread"] + 0.01 * c["size"] + 0.01 * c["n_families"] + 0.05 * c["acc_min"]
    c = c.sort_values("score", ascending=False)
    keep, seen = [], []
    for r in c.itertuples():   # drop near-duplicates (a subset/superset of an already kept set sharing both extremes)
        if any({r.most_reliable, r.least_reliable} <= set(s) for s in seen):
            continue
        keep.append(r.Index)
        seen.append(r.models)
        if len(keep) == top:
            break
    return c.loc[keep]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = ProjectPaths()
    per_model, tabA, tabB, tabC, store = [], [], [], [], {}
    for ds in KB.LABELS:
        agg = KB.load(ds, paths)
        agg = agg[~agg["model"].isin(EXCLUDE_MODELS.get(ds, []))]
        k = KB.MIN_PARA
        models, n_q, _, fams = KB.select_models(agg, k)
        qidx, A, Y, npara = per_question(agg, models, k)
        assert len(qidx) == n_q, (ds, len(qidx), n_q)
        store[ds] = {"idx": qidx, "A": A, "Y": Y, "n_para": npara, "models": models}
        df = pd.DataFrame.from_records([metrics(A[m], Y[m]) for m in models], index=models)
        df["family"] = [KB.family(m) for m in df.index]
        df = df.sort_values("acc", ascending=False)
        df.insert(0, "dataset", ds)
        df.insert(1, "n_q", n_q)
        per_model.append(df)
        print(f"{ds}: {len(models)} models, {n_q} q, median para {np.median(npara):.0f}")

        tau, _ = kendalltau(df["acc"], df["R"])
        tabA.append({"dataset": ds, "n_models": len(models), "n_q": n_q,
                     "acc_range": f"{df['acc'].min():.2f}-{df['acc'].max():.2f}",
                     "median_over_gap": df["over_gap"].median(), "median_under_gap": df["under_gap"].median(),
                     "median_B": df["B"].median(), "median_reversal": df["reversal"].median(),
                     "max_reversal": df["reversal"].max(), "max_reversal_model": KB.short(df["reversal"].idxmax()),
                     "R_range_top_half": df[df["acc"] >= df["acc"].median()]["R"].pipe(lambda s: s.max() - s.min()),
                     "kendall_acc_vs_R": tau,
                     "n_acc_outside_RL": int(df["acc_outside_RL"].sum())})

        pool, o, u, note = pick_fig2(df)
        for side, m in (("overstatement", o), ("understatement", u)):
            r = df.loc[m]
            tabB.append({"dataset": ds, "role": side, "model": KB.short(m), "acc": r["acc"], "R": r["R"], "L": r["L"],
                         "sev_over": r["sev_over"], "sev_over_cond": r["sev_over_cond"],
                         "sev_under": r["sev_under"], "sev_under_cond": r["sev_under_cond"],
                         "n_q_in_region": int(round((r["sev_over"] if side == "overstatement" else r["sev_under"]) * n_q)),
                         "pool_size": len(pool), "note": note})

        c = fig3_candidates(df, A)
        for rank, r in enumerate(c.itertuples(), 1):
            tabC.append({"dataset": ds, "rank": rank, "models": "; ".join(f"{KB.short(m)} ({df.loc[m,'acc']:.2f}/{df.loc[m,'R']:.2f})"
                                                                            for m in r.models),
                         "acc_window": f"{r.acc_min:.3f}-{r.acc_max:.3f}", "R_spread": r.R_spread,
                         "L_spread": r.L_spread, "meanA_spread": r.meanA_spread,
                         "R_diff_95CI": f"[{r.R_diff_CI_lo:+.3f}, {r.R_diff_CI_hi:+.3f}]",
                         "significant": r.R_diff_CI_lo > 0, "n_families": r.n_families})

    pm = pd.concat(per_model).rename_axis("model").reset_index()
    pm["model"] = pm["model"].map(KB.short)
    pm.to_csv(OUT / "rq1_per_model_metrics.csv", index=False)
    pd.DataFrame(tabA).to_csv(OUT / "rq1_tableA_benchmarks.csv", index=False)
    pd.DataFrame(tabB).to_csv(OUT / "rq1_tableB_fig2_reps.csv", index=False)
    pd.DataFrame(tabC).to_csv(OUT / "rq1_tableC_fig3_candidates.csv", index=False)
    with open(OUT / "rq1_per_question.pkl", "wb") as f:
        pickle.dump(store, f)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 140,
                           "display.float_format", "{:.3f}".format):
        print(pd.DataFrame(tabA).to_string(index=False))
        print(pd.DataFrame(tabB).to_string(index=False))
        print(pd.DataFrame(tabC).to_string(index=False))


if __name__ == "__main__":
    main()
