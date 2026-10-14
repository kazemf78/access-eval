"""
RQ2 -- is formulation-induced variability more than decoding noise?  No new LLM calls.

Data (a separate decoding-variability study on the v00 paraphrase pools):
  paper/data/rq2/graded_variability_*.csv.gz   4 models x {SimpleQA 150, TruthfulQA 102, GSM8K 150}
      inquiries x 8 formulations (id 0 = original wording) x 5 samples at T=0.7
  paper/data/rq2/*_formulation_only_existing.csv.gz   greedy (T=0) answer for the same 8 formulations
  paper/data/rq2/determinism_full_summary.csv      5 greedy repeats of the original wording
      (200 other inquiries; Qwen3-0.6B instead of gpt-4o)
Only the 7 PARAPHRASES are used for formulation quantities (as in A(q)); the original wording is excluded.

Quantities per (benchmark, model):
  same-wording disagreement   P(two samples of one formulation differ in correctness)          [T=0.7; T=0 from determinism]
  cross-formulation disagr.   P(answers to two different formulations differ)                   [T=0.7 samples; T=0 greedy]
  Access@k / Reliable@k       k tries succeed at least once / every time, where the k tries are
                              (a) k distinct formulations, one T=0.7 sample each, or (b) k T=0.7 samples of ONE formulation.
                              (b) uses the unbiased hypergeometric estimator over the 5 samples; (a) enumerates all
                              k-subsets of the 7 paraphrases with per-formulation success rates c_f/5 (samples of
                              different formulations are independent, so the product is unbiased).
Outputs: $PAPER_REPRO_DIR/rq2/figs/<VERSION>/ (pdf/png, csv, manifest.json).
Usage: python paper/rq2_decoding_vs_formulation.py  (or: bash paper/reproduce.sh)
"""
import datetime
import itertools
import json
import sys
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq2_helpers as AEV  # noqa: E402
from rq1_render import NAMES, RC, SERIES, STYLES, C  # noqa: E402

VERSION = "v4"   # v4: no connector line, same/different-wording markers offset vertically, ★ = 95% CI of the
#     difference excludes 0 (T=0.7 disagreement and k-budget panels; greedy panel has no paired CI). v3: v3: NEW two-panel main fig_rq2_main_2panel (T=0.7 | k=5 budget); 3-panel fig_rq2_main -> appendix.
# v2: v2: NEW single main figure fig_rq2_main (greedy | T=0.7 | k=5 budget); v1 figs -> appendix
import paper_paths as PP  # noqa: E402
VAR = PP.DATA / "rq2"
DET = VAR / "determinism_full_summary.csv"
OUT = PP.out_dir(None, "rq2") / "figs" / VERSION
DATASETS = {"SimpleQA": "SimpleQA", "TruthfulQA": "TruthfulQA", "gsm8k_test": "GSM8K"}
MODELS = ["gpt-4.1-mini", "gpt-4o", "Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct"]
KS = [1, 2, 3, 4, 5]
N_BOOT = 2000
RNG = np.random.default_rng(0)


def _file_model(m):
    return m.replace("/", "_")


def t07_tensor(ds, m):
    Y, _ = AEV._variability_tensor(VAR / f"graded_variability_{ds}_{_file_model(m)}.csv.gz")
    idx = np.sort(pd.read_csv(VAR / f"graded_variability_{ds}_{_file_model(m)}.csv.gz", usecols=["idx"])["idx"].unique())
    return idx, Y[:, 1:, :]           # paraphrases only


def greedy_matrix(ds, m, idx):
    g = pd.read_csv(VAR / f"{ds}_formulation_only_existing.csv.gz")
    g = g[(g["model"] == m) & (g["formulation_id"] > 0)]
    g = g.assign(y=g["label"].map(lambda v: v is True or v == "True" or v == "correct").astype(np.int8))
    p = g.pivot(index="idx", columns="formulation_id", values="y").reindex(idx)
    assert not p.isna().any().any(), (ds, m)
    return p.to_numpy()


def cross_disagreement_greedy(G):
    F = G.shape[1]
    c = G.sum(1)
    return c * (F - c) / (F * (F - 1) / 2)


def boot_ci(v):
    b = v[RNG.integers(0, len(v), (N_BOOT, len(v)))].mean(1)
    return np.percentile(b, [2.5, 97.5])


def access_reliable(Y, k):
    """Per-inquiry (formulation Access@k, formulation Reliable@k, decoding Access@k, decoding Reliable@k)."""
    Q, F, R = Y.shape
    c = Y.sum(-1)
    p = c / R
    subsets = np.array(list(itertools.combinations(range(F), k)))
    fa = 1 - np.prod(1 - p[:, subsets], axis=-1).mean(1)
    fr = np.prod(p[:, subsets], axis=-1).mean(1)
    da = (1 - np.vectorize(lambda x: comb(R - x, k))(c) / comb(R, k)).mean(1)
    dr = (np.vectorize(lambda x: comb(x, k))(c) / comb(R, k)).mean(1)
    return fa, fr, da, dr


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    det = pd.read_csv(DET)
    det["T0_same"] = (100 - det["mean_pairwise_label_agreement"]) / 100
    rows, krows = [], []
    for ds, m in itertools.product(DATASETS, MODELS):
        idx, Y = t07_tensor(ds, m)
        G = greedy_matrix(ds, m, idx)
        within, cross = AEV._pairwise_disagreement(Y)
        g_cross = cross_disagreement_greedy(G)
        d0 = det[(det["dataset"] == ds) & (det["model"] == m)]
        r = {"dataset": DATASETS[ds], "model": NAMES[m], "n_inquiries": len(idx), "acc_T07": Y.mean(),
             "acc_T0": G.mean(), "same_T07": within.mean(), "cross_T07": cross.mean(), "cross_T0": g_cross.mean(),
             "same_T0": d0["T0_same"].iloc[0] if len(d0) else np.nan,
             "same_T0_n": int(d0["n_questions"].iloc[0]) if len(d0) else 0,
             "excess_T07_pp": 100 * (cross.mean() - within.mean())}
        for k, v in (("same_T07", within), ("cross_T07", cross), ("cross_T0", g_cross)):
            r[f"{k}_lo"], r[f"{k}_hi"] = boot_ci(v)
        r["excess_T07_lo_pp"], r["excess_T07_hi_pp"] = 100 * boot_ci(cross - within)
        r["share_inq_cross_gt_same"] = (cross > within + 1e-12).mean()
        rows.append(r)
        for k in KS:
            fa, fr, da, dr = access_reliable(Y, k)
            krows.append({"dataset": DATASETS[ds], "model": NAMES[m], "k": k, "access_form": fa.mean(),
                          "access_decode": da.mean(), "reliable_form": fr.mean(), "reliable_decode": dr.mean(),
                          "access_gap_lo": boot_ci(fa - da)[0], "access_gap_hi": boot_ci(fa - da)[1],
                          "reliable_gap_lo": boot_ci(dr - fr)[0], "reliable_gap_hi": boot_ci(dr - fr)[1]})
    t = pd.DataFrame(rows)
    kt = pd.DataFrame(krows)
    t.to_csv(OUT / "rq2_disagreement.csv", index=False)
    kt.to_csv(OUT / "rq2_access_reliable_at_k.csv", index=False)
    fig_disagreement(t)
    fig_k(kt)
    fig_main(t, kt)
    fig_main(t, kt, greedy=False, fname="fig_rq2_main_2panel")
    manifest = {"version": VERSION, "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "data": "decoding-variability study (T=0.7, 8 formulations x 5 samples) + greedy answers for the same "
                        "formulations + T=0 determinism repeats; v00 paraphrases; original wording excluded",
                "models": [NAMES[m] for m in MODELS], "benchmarks": list(DATASETS.values()),
                "files": sorted(p.name for p in OUT.iterdir())}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print(t[["dataset", "model", "n_inquiries", "acc_T0", "acc_T07", "same_T0", "cross_T0", "same_T07",
                 "cross_T07", "excess_T07_pp", "excess_T07_lo_pp", "share_inq_cross_gt_same"]].to_string(index=False))
        print(kt[kt.k.isin([1, 3, 5])].to_string(index=False))
    print(f"wrote {OUT}")


def fig_disagreement(t):
    ds_list = list(DATASETS.values())
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(12, 2.9), sharey=True, gridspec_kw={"wspace": 0.08})
        for ax, ds in zip(axes, ds_list):
            d = t[t.dataset == ds].reset_index(drop=True)
            y = np.arange(len(d))
            for yi, r in d.iterrows():
                ax.plot([r.same_T07, r.cross_T07], [yi + 0.12] * 2, color="#c8c7bf", lw=1.2, zorder=1)
                if not np.isnan(r.same_T0):
                    ax.plot([r.same_T0, r.cross_T0], [yi - 0.12] * 2, color="#c8c7bf", lw=1.2, zorder=1, ls=":")
            spec = [("same_T0", -0.12, SERIES[0], "white"), ("cross_T0", -0.12, SERIES[1], "white"),
                    ("same_T07", 0.12, SERIES[0], SERIES[0]), ("cross_T07", 0.12, SERIES[1], SERIES[1])]
            for col, dy, ec, fc in spec:
                xerr = None
                if f"{col}_lo" in d:
                    xerr = [d[col] - d[f"{col}_lo"], d[f"{col}_hi"] - d[col]]
                ax.errorbar(d[col], y + dy, xerr=xerr, fmt="o", ms=6, mfc=fc, mec=ec, ecolor=ec, mew=1.4, lw=1,
                            capsize=0, zorder=3)
            ax.set_yticks(y, d["model"])
            ax.set_ylim(len(d) - 0.5, -0.5)
            ax.set_xlim(left=0)
            ax.set_title(ds, loc="left")
            ax.set_xlabel("P(two answers differ in correctness)")
            ax.grid(axis="x", color=C["grid"], lw=0.7)
            ax.set_axisbelow(True)
        handles = [Line2D([], [], marker="o", ls="none", mfc=SERIES[0], mec=SERIES[0], label="same wording, T=0.7 samples"),
                   Line2D([], [], marker="o", ls="none", mfc=SERIES[1], mec=SERIES[1], label="different wordings, T=0.7"),
                   Line2D([], [], marker="o", ls="none", mfc="white", mec=SERIES[0], mew=1.4,
                          label="same wording, T=0 repeats"),
                   Line2D([], [], marker="o", ls="none", mfc="white", mec=SERIES[1], mew=1.4,
                          label="different wordings, T=0 (greedy)")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False)
        fig.savefig(OUT / "fig_rq2_disagreement.pdf", bbox_inches="tight")
        fig.savefig(OUT / "fig_rq2_disagreement.png", bbox_inches="tight", dpi=300)
        plt.close(fig)


OFF = 0.17       # vertical offset between same-wording and different-wording markers (rows are 1 apart)
STAR_DX = 0.004  # gap between a significant row's upper CI end and its star (probability units)


def fig_main(t, kt, k=5, greedy=True, fname="fig_rq2_main"):
    """One figure, 12 rows (benchmark x model), three aligned panels:
    (a) greedy decoding (the main-study setting): P(outcome differs) for repeats of one wording vs different wordings;
    (b) the same at T=0.7 (both on identical inquiries, with CIs);
    (c) fixed budget of k tries: Access@k and Reliable@k with k different wordings minus k samples of one wording."""
    ds_list = list(DATASETS.values())
    ypos, labels, groups, y = [], [], [], 0.0
    for ds in ds_list:
        groups.append((ds, y))
        for m in t[t.dataset == ds].model:
            ypos.append(y)
            labels.append(m)
            y += 1
        y += 0.9
    t = t.set_index(["dataset", "model"]).loc[[(d, m) for d in ds_list for m in t[t.dataset == d].model]]
    kk = kt[kt.k == k].set_index(["dataset", "model"]).loc[t.index]
    ypos = np.array(ypos)
    blue, orange = SERIES[0], SERIES[1]
    with plt.rc_context(RC):
        specs = [("same_T0", "cross_T0", "(a) Greedy decoding (main setting)", False),
                 ("same_T07", "cross_T07", "(b) Sampling, T = 0.7", True)]
        if not greedy:
            specs = [("same_T07", "cross_T07", "(a) Disagreement, T = 0.7", True)]
        n = len(specs) + 1
        fig, axes = plt.subplots(1, n, figsize=(3.85 * n, 4.2 if not greedy else 4.6), sharey=True,
                                 gridspec_kw={"wspace": 0.12, "width_ratios": [1] * (n - 1) + [1.15]})
        for ax, (same, cross, title, filled) in zip(axes[:-1], specs):
            for yi, (_, r) in zip(ypos, t.iterrows()):
                if np.isnan(r[same]):
                    ax.text(0.003, yi + OFF, "n/a", fontsize=7, color=C["muted"], va="center")
            for col, colr, dy in ((same, blue, OFF), (cross, orange, -OFF)):   # y axis is inverted: -OFF = higher
                xerr = [t[col] - t[f"{col}_lo"], t[f"{col}_hi"] - t[col]] if f"{col}_lo" in t else None
                ax.errorbar(t[col], ypos + dy, xerr=xerr, fmt="o", ms=5.5, mfc=colr if filled else "white", mec=colr,
                            ecolor=colr, mew=1.4, lw=1, capsize=0, zorder=3)
            if same == "same_T07":   # paired bootstrap CI of (different - same) is available only at T=0.7
                for yi, (_, r) in zip(ypos, t.iterrows()):
                    if r["excess_T07_lo_pp"] > 0 or r["excess_T07_hi_pp"] < 0:
                        ax.text(max(r[f"{cross}_hi"], r[f"{same}_hi"]) + STAR_DX, yi, "★", fontsize=9,
                                va="center", ha="left", color="#222")
            ax.set_xlim(0, np.nanmax(t[["cross_T0_hi", "cross_T07_hi"]].to_numpy()) * 1.05)   # shared scale (a)/(b)
            ax.set_title(title, loc="left", fontsize=10)
            ax.set_xlabel("P(two answers differ in correctness)")
        ax = axes[-1]
        ax.axvline(0, color="#3a3a36", lw=0.9, zorder=1)
        acc_gap = 100 * (kk["access_form"] - kk["access_decode"])
        rel_gap = 100 * (kk["reliable_form"] - kk["reliable_decode"])
        ax.errorbar(acc_gap, ypos - 0.15, xerr=[acc_gap - 100 * kk["access_gap_lo"], 100 * kk["access_gap_hi"] - acc_gap],
                    fmt="D", ms=5, color=C["Reliable"] if "Reliable" in C else "#0ca30c", lw=1, capsize=0, zorder=3)
        # reliable gap CI is stored as (decode - form); flip sign
        ax.errorbar(rel_gap, ypos + 0.15, xerr=[rel_gap + 100 * kk["reliable_gap_hi"], -100 * kk["reliable_gap_lo"] - rel_gap],
                    fmt="s", ms=5, color="#d03b3b", lw=1, capsize=0, zorder=3)
        for yi, ag, rg, (_, r) in zip(ypos, acc_gap, rel_gap, kk.iterrows()):
            if r["access_gap_lo"] > 0 or r["access_gap_hi"] < 0:
                ax.text(100 * r["access_gap_hi"] + 0.4, yi - 0.15, "★", fontsize=8, va="center", color="#222")
            if r["reliable_gap_lo"] > 0 or r["reliable_gap_hi"] < 0:
                ax.text(-100 * r["reliable_gap_hi"] - 0.4, yi + 0.15, "★", fontsize=8, va="center", ha="right",
                        color="#222")
        ax.set_title(f"({'c' if greedy else 'b'}) {k} tries: wordings − samples", loc="left", fontsize=10)
        ax.set_xlabel(f"Difference in points")
        lim = max(100 * kk["access_gap_hi"].max(), 100 * kk["reliable_gap_hi"].max()) * 1.08 + 1.5   # room for ★
        ax.set_xlim(-lim, lim)
        for a in axes:
            a.grid(axis="x", color=C["grid"], lw=0.7)
            a.set_axisbelow(True)
            for ds, y0 in groups[1:]:
                a.axhline(y0 - 0.95, color=C["grid"], lw=0.8)
        axes[0].set_yticks(ypos, labels)
        axes[0].set_ylim(ypos.max() + 0.6, -1.3)
        for ds, y0 in groups:
            axes[0].text(-0.02, y0 - 0.85, ds, transform=axes[0].get_yaxis_transform(), ha="right", va="center",
                         fontsize=9.5, fontweight="bold")
        handles = ([Line2D([], [], marker="o", ls="none", mfc="white", mec=blue, mew=1.4, label="same wording (a: repeats)"),
                    Line2D([], [], marker="o", ls="none", mfc="white", mec=orange, mew=1.4, label="different wordings (a)"),
                    Line2D([], [], marker="o", ls="none", color=blue, label="same wording (b: samples)"),
                    Line2D([], [], marker="o", ls="none", color=orange, label="different wordings (b)")] if greedy else
                   [Line2D([], [], marker="o", ls="none", color=blue, label="same wording, repeated samples"),
                    Line2D([], [], marker="o", ls="none", color=orange, label="different wordings")]) + [
                   Line2D([], [], marker="D", ls="none", color="#0ca30c", label=f"Δ Access@{k}"),
                   Line2D([], [], marker="s", ls="none", color="#d03b3b", label=f"Δ Reliable@{k}"),
                   Line2D([], [], marker="$★$", ls="none", color="#222", ms=8, label="95% CI of difference excludes 0")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=4 if greedy else 5,
                   frameon=False, fontsize=8.5, handletextpad=0.3, columnspacing=1.0)
        fig.savefig(OUT / f"{fname}.pdf", bbox_inches="tight")
        fig.savefig(OUT / f"{fname}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)


def fig_k(kt):
    ds_list = list(DATASETS.values())
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), gridspec_kw={"hspace": 0.42, "wspace": 0.25})
        for j, ds in enumerate(ds_list):
            for i, (metric, lab) in enumerate((("access", "Access@k  (≥1 of k tries correct)"),
                                               ("reliable", "Reliable@k  (all k tries correct)"))):
                ax = axes[i, j]
                for mi, m in enumerate(kt.model.unique()):
                    d = kt[(kt.dataset == ds) & (kt.model == m)]
                    ax.plot(d.k, d[f"{metric}_form"], color=SERIES[mi], lw=1.8, marker="o", ms=4)
                    ax.plot(d.k, d[f"{metric}_decode"], color=SERIES[mi], lw=1.4, ls="--", marker="o", ms=4,
                            mfc="white")
                ax.set_xticks(KS)
                ax.set_xlabel("k tries")
                if j == 0:
                    ax.set_ylabel(lab)
                ax.set_title(ds if i == 0 else "", loc="left")
                ax.grid(color=C["grid"], lw=0.7)
                ax.set_axisbelow(True)
        handles = [Line2D([], [], color=SERIES[i], lw=1.8, label=m) for i, m in enumerate(kt.model.unique())]
        handles += [Line2D([], [], color="#555", lw=1.8, label="k different formulations"),
                    Line2D([], [], color="#555", lw=1.4, ls="--", marker="o", mfc="white",
                           label="k samples of one formulation")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=6, frameon=False)
        fig.savefig(OUT / "fig_rq2_access_reliable_at_k.pdf", bbox_inches="tight")
        fig.savefig(OUT / "fig_rq2_access_reliable_at_k.png", bbox_inches="tight", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
