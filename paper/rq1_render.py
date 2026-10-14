"""
RQ1 -- render the paper figures from rq1_selection_pass.py's per-question store (rq1_figs/rq1_per_question.pkl).
Model selections are recomputed with the SAME rules (rq1_selection_pass.pick_fig2 / fig3_candidates rank 1).

  Fig 1  capability envelope: stacked Reliable / Brittle / Absent (sorted by benchmark accuracy), black tick = benchmark
         (canonical-wording) accuracy, blue diamond = latent L = P(A>0).
  Fig 2  benchmark representativeness: mirrored histograms of A(q) -- canonical successes above the axis, canonical
         failures below -- with the A = 0.5 line; P(A<0.5 | Y_orig=1) and P(A>0.5 | Y_orig=0) highlighted.
  Fig 2' canonical majority reversals: per benchmark, one dot per model for P(Y_orig=1, A<0.5) (canonical success,
         most formulations fail) and P(Y_orig=0, A>0.5) (canonical failure, most formulations succeed), as a share
         of all inquiries (main), median bar + IQR. The _conditional variant (appendix) plots P(A<0.5 | Y_orig=1) and
         P(A>0.5 | Y_orig=0) instead; groups with < MIN_GROUP canonical inquiries are dropped there.
  Fig 3  reliability curves R(tau) = P(A(q) >= tau), tau in (0,1] (left end = 0+ = latent), for the near-tied group;
         benchmarks without a qualifying group are omitted (named in the manifest / caption).
  Fig 4  accessible breadth L vs reliable breadth R: all models gray, the most divergent near-tied pairs highlighted
         and connected; the impossible R > L region shaded.
  Fig 5  near-tied pairs, aggregate: every model pair with |d acc| <= PAIR_PP; x = |d acc|, y = |d R| (filled = paired
         bootstrap 95% CI of d R excludes 0).
  Table  rq1_summary_table.{tex,csv} (per-benchmark medians) + rq1_near_tied_pairs.csv / _summary.csv.

Outputs go to $PAPER_REPRO_DIR/rq1/figs/<VERSION>/ with a manifest.json (settings + what changed). Data files stay in $PAPER_REPRO_DIR/rq1/.
Main-paper versions use MAIN = (TruthfulQA, GSM8K, BFCL); *_appendix versions cover all benchmarks.
Usage: python paper/rq1_render.py  (or: bash paper/reproduce.sh) [--fig1-main-only]
  --only fig1_main,fig2_reversals,fig3_main   render only the named figures (+ manifest)
"""
import datetime
import itertools
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import knowledge_state_bars as KB  # noqa: E402
from model_names import MODEL_NAMES  # noqa: E402
import rq1_selection_pass as SP  # noqa: E402

VERSION = "v6"
CHANGES = ["v6 renders ONLY fig1 main with an MMLU panel (--only fig1_main): MAIN_FIG1 = TruthfulQA, GSM8K, MMLU, BFCL, 4 narrower columns, Absent red; everything else unchanged", "--- v5 changes below ---", "v5 renders ONLY fig3_reliability_curves_main_legend_inside (--only fig3_main_inside): same data/models "
           "as v4 fig3 main, legend moved inside each panel (lower left, empty region) to save vertical space; "
           "everything else unchanged from v4",
           "--- v4 changes below ---","BFCL added to MAIN (TruthfulQA, GSM8K, BFCL); main multi-panel figs use 3 columns",
           "v4 renders ONLY fig1 main (Absent red) and the NEW fig2 canonical majority reversals "
           "(--only fig1_main,fig2_reversals,fig3_main); other figs unchanged from v3",
           "fig3 main = GSM8K + BFCL (MAIN_FIG3)",
           "NEW rq1_main_table.{tex,csv} (--only table_main): per-benchmark medians over models of acc, R, L, "
           "brittle = L - R (envelope width), majority reversal P(Y=1,A<0.5)+P(Y=0,A>0.5); Kendall tau(acc, R)",
           "NEW fig2_majority_reversals (main): all 6 benchmarks, one dot per model, GLOBAL rates "
           "P(Y=1, A<0.5) and P(Y=0, A>0.5) as a share of ALL inquiries, median + IQR; no models dropped",
           "fig2_majority_reversals_conditional (appendix): same design with P(A<0.5|Y=1), P(A>0.5|Y=0); model "
           "dropped from a group with < 20 canonical inquiries. Mirrored-histogram fig2_representativeness -> appendix"]
DATA = SP.OUT
OUT = DATA / "figs" / VERSION
MAIN = ["TruthfulQA", "gsm8k", "bfcl"]
MAIN_FIG3 = ["gsm8k", "bfcl"]
MAIN_FIG1 = ["TruthfulQA", "gsm8k", "mmlu", "bfcl"]   # v6: MMLU added to the main envelope figure   # user, v4: reliability curves in the main text use GSM8K + BFCL
ALL = list(KB.LABELS)
TITLES = {"SimpleQA": "SimpleQA", "TruthfulQA": "TruthfulQA", "gsm8k": "GSM8K", "math_500": "MATH-500",
          "mmlu": "MMLU", "bfcl": "BFCL"}
PAIR_PP = 0.03
ABSENT = {"gray": "#c9c8c1", "red": "#d03b3b"}   # plan vs dashboards (plotting.POSTER_STATE_COLORS)
ABSENT_DEFAULT = "gray"
C = {"Reliable": "#0ca30c", "Brittle": "#fab219", "Absent": ABSENT[ABSENT_DEFAULT], "acc": "#0b0b0b",
     "latent": "#2a78d6", "over": "#eb6834", "under": "#2a78d6", "neutral": "#b5b4ad", "grid": "#e1e0d9",
     "muted": "#6b6a64", "context": "#bdbcb5", "over_light": "#f6bba3", "under_light": "#9ec5f4"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]   # categorical slots 1,2,3,7 (skip yellow: it is Brittle)
DS_COLORS = dict(zip(ALL, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]))  # slots 1-5,7
DS_MARKERS = dict(zip(ALL, ["o", "s", "^", "D", "v", "P"]))
STYLES = ["-", "--", "-.", ":"]
RC = {"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
      "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "axes.spines.top": False,
      "axes.spines.right": False, "pdf.fonttype": 42}


NAME_OVERRIDES = {"mistralai/Mistral-7B-Instruct-v0.2": "Mistral 7B"}   # paper-only; map keeps v0.2 for paper 1
NAMES = {**MODEL_NAMES, **NAME_OVERRIDES}


def name(m):
    if m not in NAMES:
        raise KeyError(f"{m!r} missing from model_names.MODEL_NAMES")
    return NAMES[m]


def load():
    with open(DATA / "rq1_per_question.pkl", "rb") as f:
        store = pickle.load(f)
    met = {}
    for ds, s in store.items():
        df = pd.DataFrame.from_records([SP.metrics(s["A"][m], s["Y"][m]) for m in s["models"]], index=s["models"])
        met[ds] = df.sort_values("acc", ascending=False)
    return store, met


def save(fig, fname):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _top_legend(fig, axes, handles, ncol, pad=0.012):
    y = max(a.get_position().y1 for a in axes.flat if a.axison)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y + pad + 0.035 / axes.shape[0]),
               ncol=ncol, frameon=False, handlelength=1.4, columnspacing=1.2)


def _grid(ax, axis="x"):
    ax.grid(axis=axis, color=C["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Fig 1
def fig1(store, met, dss, fname, ncols, absent=ABSENT_DEFAULT, panel_w=3.6, wspace=0.5):
    nrows = -(-len(dss) // ncols)
    nmax = max(len(met[d]) for d in dss)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, 0.21 * nmax * nrows + 0.9 * nrows),
                                 squeeze=False, gridspec_kw={"wspace": wspace, "hspace": 0.32})
        for ax, ds in zip(axes.flat, dss):
            df = met[ds]
            y = np.arange(len(df))
            left = np.zeros(len(df))
            for st, col in (("Reliable", "R"), ("Brittle", "B")):
                v = df[col].to_numpy(float)
                ax.barh(y, v, left=left, height=0.74, color=C[st], edgecolor="white", linewidth=0.8, zorder=2)
                left += v
            ax.barh(y, 1 - left, left=left, height=0.74, color=ABSENT[absent], edgecolor="white", linewidth=0.8,
                    zorder=2)
            ax.plot(df["acc"], y, "|", color=C["acc"], markersize=9, markeredgewidth=2, zorder=5)
            ax.plot(df["L"], y, "D", color=C["latent"], markersize=4.2, markeredgecolor="white",
                    markeredgewidth=0.6, zorder=5)
            ax.set_yticks(y, [name(m) for m in df.index])
            ax.set_ylim(len(df) - 0.5, -0.5)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", ".25", ".5", ".75", "1"])
            _grid(ax)
            ax.set_title(TITLES[ds], loc="left", fontsize=10.5)
            ax.set_xlabel("Fraction of inquiries")
        for ax in list(axes.flat)[len(dss):]:
            ax.axis("off")
        handles = [Patch(color=C["Reliable"], label="Reliable  A(q)=1"),
                   Patch(color=C["Brittle"], label="Brittle  0<A(q)<1"),
                   Patch(color=ABSENT[absent], label="Absent  A(q)=0"),
                   Line2D([], [], marker="|", color=C["acc"], ls="none", markersize=9, markeredgewidth=2,
                          label="Benchmark accuracy"),
                   Line2D([], [], marker="D", color=C["latent"], ls="none", markersize=5, label="Latent  P(A>0)")]
        _top_legend(fig, axes, handles, ncol=5 if ncols > 1 else 3)
        save(fig, fname)


# ---------------------------------------------------------------- Fig 2
BOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5}
END_X, END_W = 0.085, 0.07   # A = 0 and A = 1 get their own bars just outside [0, 1]


def _positions():
    """(x, width, mask_fn): exact A=0, ten interior bins [i/10, (i+1)/10) restricted to 0<A<1, exact A=1."""
    pos = [(-END_X, END_W, lambda a: a <= 1e-9)]
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        pos.append((lo + 0.05, 0.088, lambda a, lo=lo, hi=hi: (a > 1e-9) & (a < 1 - 1e-9) & (a >= lo - 1e-9)
                    & (a < hi - 1e-9)))
    pos.append((1 + END_X, END_W, lambda a: a >= 1 - 1e-9))
    return pos


def _stack(ax, a, n, sign, cats):
    """cats: [(colour, mask over a)], drawn stacked (away from the axis) inside every bar position."""
    for x, w, fn in _positions():
        in_bin, base = fn(a), 0.0
        for col, cat in cats:
            h = (in_bin & cat).sum() / n
            if h:
                ax.bar([x], [sign * h], bottom=[sign * base], width=w, color=col, zorder=2, linewidth=0)
                base += h
    return max((fn(a).sum() / n for _, _, fn in _positions()), default=0)


def fig2_panel(ax, a, y, title):
    succ, fail = a[y == 1], a[y == 0]
    ns, nf = max(len(succ), 1), max(len(fail), 1)
    ts = _stack(ax, succ, ns, 1, [(C["over"], succ < 0.5), (C["over_light"], (succ >= 0.5) & (succ < 1)),
                                  (C["neutral"], succ >= 1)])
    tf = _stack(ax, fail, nf, -1, [(C["under"], fail > 0.5), (C["under_light"], (fail > 0) & (fail <= 0.5)),
                                   (C["neutral"], fail <= 0)])
    ax.axhline(0, color="#3a3a36", linewidth=0.8, zorder=4)
    ax.axvline(0.5, color="#3a3a36", linewidth=1.0, linestyle="--", zorder=4)
    for xe in (-END_X + END_W / 2 + 0.0075, 1 + END_X - END_W / 2 - 0.0075):   # separators around the exact bars
        ax.axvline(xe, color=C["grid"], linewidth=0.8, zorder=1)
    top = max(ts, tf) * 1.22
    ax.set_xlim(-END_X - END_W / 2 - 0.01, 1 + END_X + END_W / 2 + 0.01)
    ax.set_xticks([-END_X, 0.25, 0.5, 0.75, 1 + END_X], ["0", ".25", ".5", ".75", "1"])
    ax.set_ylim(-top, top)
    ticks = [t for t in ax.get_yticks() if abs(t) <= top]
    ax.set_yticks(ticks, [f"{abs(t):.0%}" for t in ticks])
    ax.set_ylim(-top, top)
    s1, s05 = int((succ < 1).sum()), int((succ < 0.5).sum())
    f0, f05 = int((fail > 0).sum()), int((fail > 0.5).sum())
    ax.text(-0.1, top * 0.96, f"canonical success (n={len(succ)})\n"
            f"P(A<1 | success) = {s1 / ns:.0%}  ({s1})\nP(A<0.5 | success) = {s05 / ns:.0%}  ({s05})",
            va="top", ha="left", fontsize=8, color="#222", linespacing=1.35, zorder=6, bbox=BOX)
    ax.text(1.14, -top * 0.96, f"canonical failure (n={len(fail)})\n"
            f"P(A>0 | failure) = {f0 / nf:.0%}  ({f0})\nP(A>0.5 | failure) = {f05 / nf:.0%}  ({f05})",
            va="bottom", ha="right", fontsize=8, color="#222", linespacing=1.35, zorder=6, bbox=BOX)
    ax.set_title(title, loc="left", fontsize=9.5)
    ax.set_xlabel("Accessibility A(q)")
    _grid(ax, "y")


def fig2(store, met, dss, fname):
    with plt.rc_context(RC):
        fig, axes = plt.subplots(len(dss), 2, figsize=(9.0, 3.1 * len(dss)), squeeze=False,
                                 gridspec_kw={"hspace": 0.55, "wspace": 0.22})
        for row, ds in zip(axes, dss):
            _, o, u, _ = SP.pick_fig2(met[ds])
            for ax, m in zip(row, (o, u)):
                ax_title = f"{TITLES[ds]} · {name(m)}  (acc {met[ds].loc[m, 'acc']:.2f})"
                fig2_panel(ax, store[ds]["A"][m], store[ds]["Y"][m], ax_title)
            row[0].set_ylabel("Share within group")
        handles = [Patch(color=C["over"], label="canonical success · majority of formulations fail  (A<0.5)"),
                   Patch(color=C["over_light"], label="canonical success · some formulation fails  (0.5≤A<1)"),
                   Patch(color=C["neutral"], label="all formulations agree with canonical  (A=1 / A=0)"),
                   Patch(color=C["under"], label="canonical failure · majority of formulations succeed  (A>0.5)"),
                   Patch(color=C["under_light"], label="canonical failure · some formulation succeeds  (0<A≤0.5)")]
        _top_legend(fig, axes, handles, ncol=2, pad=0.02)
        save(fig, fname)


# ---------------------------------------------------------------- Fig 2' (majority reversals)
MIN_GROUP = 20   # min canonical successes (failures) for a model's conditional rate to be plotted


def reversal_points(store, met, dss):
    rows = []
    for ds in dss:
        n = len(store[ds]["idx"])
        for m, r in met[ds].iterrows():
            ns = int(round(r["acc"] * n))
            for side, k in (("success", ns), ("failure", n - ns)):
                g, c = ("sev_over", "sev_over_cond") if side == "success" else ("sev_under", "sev_under_cond")
                rows.append({"dataset": ds, "model": m, "side": side, "global_rate": r[g], "cond_rate": r[c],
                             "n_group": k, "n_inquiries": n, "kept_cond": k >= MIN_GROUP})
    return pd.DataFrame(rows)


def fig2_reversals(store, met, dss, fname, conditional=False):
    pts = reversal_points(store, met, dss)
    kept = (pts[pts.kept_cond] if conditional else pts).assign(
        rate=lambda d: d["cond_rate"] if conditional else d["global_rate"])
    rng = np.random.default_rng(0)
    side_style = {"success": (-0.19, C["over"]), "failure": (0.19, C["under"])}
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7.2, 3.3))
        top = kept["rate"].max() * 100 * 1.12
        for i, ds in enumerate(dss):
            if i % 2:
                ax.axvspan(i - 0.5, i + 0.5, color="#f5f4ef", zorder=0, linewidth=0)
            for side, (dx, col) in side_style.items():
                v = kept[(kept.dataset == ds) & (kept.side == side)]["rate"].to_numpy() * 100
                n_all = len(met[ds])
                ax.text(i + dx, -top * 0.025, f"{len(v)}/{n_all}" if len(v) < n_all else "", ha="center", va="top",
                        fontsize=6.5, color=C["muted"])
                if not len(v):
                    continue
                x = i + dx + rng.uniform(-0.07, 0.07, len(v))
                ax.scatter(x, v, s=16, color=col, alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3)
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                ax.plot([i + dx, i + dx], [q1, q3], color="#222", linewidth=1.4, zorder=4, solid_capstyle="butt")
                ax.plot([i + dx - 0.12, i + dx + 0.12], [med, med], color="#222", linewidth=2.2, zorder=4)
                ax.text(i + dx, max(v.max(), q3) + top * 0.025, f"{med:.1f}%" if med < 10 else f"{med:.0f}%", ha="center", va="bottom",
                        fontsize=7.5, color="#222")
        ax.set_xticks(range(len(dss)), [TITLES[d] for d in dss])
        ax.tick_params(axis="x", length=0, pad=16 if conditional else 6)
        ax.set_xlim(-0.5, len(dss) - 0.5)
        ax.set_ylim(0, top)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda t, _: f"{t:.0f}%"))
        ax.set_ylabel("Share of canonical successes / failures" if conditional else "Share of all inquiries")
        _grid(ax, "y")
        handles = [Line2D([], [], marker="o", ls="none", color=C["over"], markersize=5,
                          label="canonical success, most formulations fail  "
                          + ("P(A<0.5 | Y=1)" if conditional else "P(Y=1, A<0.5)")),
                   Line2D([], [], marker="o", ls="none", color=C["under"], markersize=5,
                          label="canonical failure, most formulations succeed  "
                          + ("P(A>0.5 | Y=0)" if conditional else "P(Y=0, A>0.5)")),
                   Line2D([], [], color="#222", linewidth=2.2, label="median (value above), IQR bar; 1 dot = 1 model")]
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=1, frameon=False, fontsize=8.3,
                  handlelength=1.6)
        save(fig, fname)
    return pts


# ---------------------------------------------------------------- Fig 3
TAU = np.linspace(0, 1, 501)[1:]   # (0, 1]


def fig3(store, met, dss, fname, ncols, legend_inside=False):
    dss = [d for d in dss if not SP.fig3_candidates(met[d], store[d]["A"], top=1).empty]
    nrows = -(-len(dss) // ncols)
    rows = []
    with plt.rc_context(RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, (2.9 if legend_inside else 3.3) * nrows),
                                 squeeze=False, gridspec_kw={"hspace": 0.4 if legend_inside else 0.95,
                                                             "wspace": 0.3})
        for ax, ds in zip(axes.flat, dss):
            cand = SP.fig3_candidates(met[ds], store[ds]["A"], top=1).iloc[0]
            models = sorted(cand["models"], key=lambda m: -met[ds].loc[m, "R"])
            lo = 1.0
            for i, m in enumerate(models):
                a = store[ds]["A"][m]
                s = (a[None, :] >= TAU[:, None] - 1e-9).mean(1)
                lo = min(lo, s.min())
                r = met[ds].loc[m]
                ax.step(TAU, s, where="post", color=SERIES[i], linestyle=STYLES[i], linewidth=1.8,
                        label=f"{name(m)}  acc {r['acc']:.2f} · R {r['R']:.2f}")
                ax.plot([1], [r["R"]], "o", color=SERIES[i], markersize=5, markeredgecolor="white", zorder=5,
                        clip_on=False)
                rows.append({"dataset": ds, "model": name(m), "acc": r["acc"], "L": r["L"], "R": r["R"],
                             "mean_A": r["mean_A"], **{f"R@{t:.1f}": (a >= t - 1e-9).mean() for t in (0.5, 0.8)}})
            ax.set_xlim(0, 1.02)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0⁺", "0.2", "0.4", "0.6", "0.8", "1"])
            top = min(1, max(met[ds].loc[models, "L"]) + 0.03)
            ax.set_ylim(max(0, np.floor(lo * 10) / 10 - 0.05), 1.0 if lo > 0.5 else top)
            ax.set_xlabel("Reliability requirement τ")
            ax.set_ylabel("R(τ) = P(A(q) ≥ τ)")
            _grid(ax, "both")
            ax.set_title(f"{TITLES[ds]}  (acc within {100 * (cand['acc_max'] - cand['acc_min']):.1f} pp)",
                         loc="left", fontsize=10)
            if legend_inside:   # curves hug the top; the lower-left region (small tau, low R) is empty
                leg = ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=7.6, ncol=1,
                                handlelength=2.2, borderaxespad=0.5, borderpad=0.4, labelspacing=0.3)
                leg.get_frame().set_edgecolor(C["grid"])
                leg.get_frame().set_linewidth(0.6)
            else:
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False, fontsize=8.3, ncol=1,
                          handlelength=2.4)
        for ax in list(axes.flat)[len(dss):]:
            ax.axis("off")
        save(fig, fname)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- near-tied pairs (Fig 4 highlights + Fig 5)
def near_tied_pairs(store, met, ds):
    df, A = met[ds], store[ds]["A"]
    rows = []
    for m1, m2 in itertools.combinations(df.index, 2):
        d_acc = abs(df.loc[m1, "acc"] - df.loc[m2, "acc"])
        if d_acc > PAIR_PP + 1e-9:
            continue
        hi, lo = (m1, m2) if df.loc[m1, "R"] >= df.loc[m2, "R"] else (m2, m1)
        ci = SP.boot_diff_R(A[hi], A[lo])
        rows.append({"dataset": ds, "more_reliable": hi, "less_reliable": lo, "d_acc": d_acc,
                     "d_R": df.loc[hi, "R"] - df.loc[lo, "R"], "d_L": df.loc[hi, "L"] - df.loc[lo, "L"],
                     "ci_lo": ci[0], "ci_hi": ci[1], "significant": ci[0] > 0,
                     "mean_acc": (df.loc[m1, "acc"] + df.loc[m2, "acc"]) / 2})
    return pd.DataFrame(rows)


def top_pairs(pairs, min_acc, k=2, max_pp=0.02):
    """k most divergent (largest d_R) significant pairs with |d acc| <= max_pp and mean acc >= min_acc (the user wants
    mid/high performers), no model reused."""
    used, out = set(), []
    ok = (pairs.d_acc <= max_pp + 1e-9) & pairs.significant & (pairs.mean_acc >= min_acc)
    for r in pairs[ok].sort_values("d_R", ascending=False).itertuples():
        if {r.more_reliable, r.less_reliable} & used:
            continue
        out.append(r)
        used |= {r.more_reliable, r.less_reliable}
        if len(out) == k:
            break
    return out


def fig4(store, met, pairs, dss, fname, ncols):
    nrows = -(-len(dss) // ncols)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.8 * nrows), squeeze=False,
                                 gridspec_kw={"hspace": 0.5, "wspace": 0.35})
        for ax, ds in zip(axes.flat, dss):
            df = met[ds]
            ax.fill_between([0, 1], [0, 1], [1, 1], color="#f1f0ea", zorder=0, linewidth=0)
            ax.plot([0, 1], [0, 1], color=C["muted"], linewidth=0.9, linestyle="--", zorder=1)
            ax.text(0.04, 0.93, "impossible: R > L", fontsize=7.5, color=C["muted"], va="top")
            ax.scatter(df["L"], df["R"], s=30, color=C["context"], edgecolors="white", linewidths=0.6, zorder=2)
            for i, r in enumerate(top_pairs(pairs[pairs.dataset == ds], df["acc"].median())):
                pts = df.loc[[r.more_reliable, r.less_reliable]]
                ax.plot(pts["L"], pts["R"], color=SERIES[i], linewidth=1.3, zorder=3)
                ax.scatter(pts["L"], pts["R"], s=46, color=SERIES[i], marker=["o", "s"][i], edgecolors="white",
                           linewidths=0.8, zorder=4)
                for m, dy in ((r.more_reliable, 6), (r.less_reliable, -10)):
                    ax.annotate(f"{name(m)} ({df.loc[m, 'acc']:.2f})", (df.loc[m, "L"], df.loc[m, "R"]),
                                xytext=(-6, dy), textcoords="offset points", ha="right", fontsize=7.5, color="#222")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xlabel("Accessible breadth  L = P(A>0)")
            ax.set_ylabel("Reliable breadth  R = P(A=1)")
            ax.set_title(TITLES[ds], loc="left", fontsize=10)
            _grid(ax, "both")
        for ax in list(axes.flat)[len(dss):]:
            ax.axis("off")
        handles = [Line2D([], [], marker="o", ls="none", color=C["context"], label="model"),
                   Line2D([], [], marker="o", color=SERIES[0], label="near-tied pair (|Δacc| ≤ 2 pp, ≥ median acc); label = acc"),
                   Line2D([], [], color=C["muted"], ls="--", label="R = L (no brittle mass)")]
        _top_legend(fig, axes, handles, ncol=3)
        save(fig, fname)


def fig5(pairs, dss, fname):
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        p = pairs[pairs.dataset.isin(dss)]
        lim = max(0.05, p["d_R"].max() * 1.08) * 100
        ax.plot([0, PAIR_PP * 100], [0, PAIR_PP * 100], color=C["muted"], linewidth=0.9, linestyle="--", zorder=1)
        rng = np.random.default_rng(0)
        for ds in dss:
            q = p[p.dataset == ds]
            jit = rng.uniform(-0.04, 0.04, len(q))
            for sig, face in ((True, DS_COLORS[ds]), (False, "white")):
                s = q[q.significant == sig]
                ax.scatter(s["d_acc"] * 100 + jit[q.significant.to_numpy() == sig], s["d_R"] * 100, s=26,
                           marker=DS_MARKERS[ds], facecolors=face, edgecolors=DS_COLORS[ds], linewidths=1.0,
                           zorder=3, label=f"{TITLES[ds]}  (n={len(q)})" if sig else None)
        ax.set_xlim(-0.15, PAIR_PP * 100 + 0.15)
        ax.set_ylim(0, lim)
        ax.set_xlabel("|Δ benchmark accuracy|  (pp)")
        ax.set_ylabel("|Δ reliable share R|  (pp)")
        _grid(ax, "both")
        ax.text(PAIR_PP * 100 * 0.55, PAIR_PP * 100 * 0.55 - 0.9, "|ΔR| = |Δacc|", fontsize=7.5, color=C["muted"],
                rotation=np.degrees(np.arctan(PAIR_PP * 100 / lim * 4.0 / 5.6 * 0.8)), ha="center")
        h, lab = ax.get_legend_handles_labels()
        h.append(Line2D([], [], marker="o", ls="none", markerfacecolor="white", markeredgecolor="#555",
                        label="hollow: 95% CI of ΔR includes 0"))
        ax.legend(handles=h, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8.5)
        ax.set_title(f"Model pairs within {PAIR_PP * 100:.0f} pp benchmark accuracy", loc="left", fontsize=10)
        save(fig, fname)


# ---------------------------------------------------------------- tables
def summary_table(store, met):
    rows = []
    for ds in ALL:
        df = met[ds]
        rows.append({"Benchmark": TITLES[ds], "Models": len(df), "Inquiries": len(store[ds]["idx"]),
                     "acc − R": df["over_gap"].median(), "L − acc": df["under_gap"].median(),
                     "Brittle": df["B"].median(), "Reversal": df["reversal"].median(),
                     "Reversal (max)": df["reversal"].max(),
                     "Over|Y=1": df["sev_over_cond"].median(), "Under|Y=0": df["sev_under_cond"].median()})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "rq1_summary_table.csv", index=False)
    pct = lambda v: f"{100 * v:.1f}"  # noqa: E731
    lines = [r"\begin{tabular}{lrr|ccc|ccc}", r"\toprule",
             r"Benchmark & \#M & \#Inq & $A_{\rm orig}-R$ & $L-A_{\rm orig}$ & Brittle & Reversal & "
             r"Sev.\ over $|\,Y{=}1$ & Sev.\ under $|\,Y{=}0$ \\", r"\midrule"]
    for r in t.itertuples(index=False):
        lines.append(f"{r[0]} & {r[1]} & {r[2]} & {pct(r[3])} & {pct(r[4])} & {pct(r[5])} & "
                     f"{pct(r[6])} (max {pct(r[7])}) & {pct(r[8])} & {pct(r[9])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "rq1_summary_table.tex").write_text("\n".join(lines) + "\n")
    return t


def main_table(store, met):
    """Compact main-text table: one row per benchmark, medians over models (all six benchmarks)."""
    from scipy.stats import kendalltau
    rows = []
    for ds in ALL:
        df = met[ds]
        rows.append({"Benchmark": TITLES[ds], "Models": len(df), "Inquiries": len(store[ds]["idx"]),
                     "Acc": df["acc"].median(), "R": df["R"].median(), "L": df["L"].median(),
                     "Brittle": df["B"].median(), "Brittle_max": df["B"].max(), "Reversal": df["reversal"].median(),
                     "Reversal_max": df["reversal"].max(), "tau_acc_R": kendalltau(df["acc"], df["R"])[0]})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "rq1_main_table.csv", index=False)
    pct = lambda v: f"{100 * v:.1f}"  # noqa: E731
    lines = [r"\begin{tabular}{lrr|ccc|cc}", r"\toprule",
             r"Benchmark & \#M & \#Inq & Acc & $R$ & $L$ & Brittle ($L-R$) & Majority reversal \\", r"\midrule"]
    for r in t.itertuples(index=False):
        lines.append(f"{r.Benchmark} & {r.Models} & {r.Inquiries} & {pct(r.Acc)} & {pct(r.R)} & {pct(r.L)} & "
                     f"{pct(r.Brittle)} (max {pct(r.Brittle_max)}) & {pct(r.Reversal)} (max {pct(r.Reversal_max)}) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "rq1_main_table.tex").write_text("\n".join(lines) + "\n")
    return t


def pairs_summary(pairs):
    g = pairs.groupby("dataset", sort=False)
    s = pd.DataFrame({"pairs": g.size(), "median_dR_pp": g["d_R"].median() * 100, "max_dR_pp": g["d_R"].max() * 100,
                      "share_dR_ge5pp": g["d_R"].apply(lambda v: (v >= 0.05).mean()),
                      "share_significant": g["significant"].mean()}).reindex(ALL)
    s.index = [TITLES[d] for d in s.index]
    return s


def _manifest(store, extra):
    manifest = {"version": VERSION, "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "changes_vs_previous": CHANGES, "tqa_rule": KB.TQA_RULE, "min_shared_paraphrases": KB.MIN_PARA,
                "models_per_benchmark": {TITLES[d]: len(store[d]["models"]) for d in ALL},
                "inquiries_per_benchmark": {TITLES[d]: len(store[d]["idx"]) for d in ALL},
                "model_set": "all non-thinking models (Qwen3 [with_thinking] variants excluded) on a common substrate",
                "main": [TITLES[d] for d in MAIN], **extra,
                "files": sorted(p.name for p in OUT.iterdir() if p.suffix in (".pdf", ".png", ".csv", ".tex"))}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    store, met = load()
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    if only:
        if "fig1_main" in only:
            fig1(store, met, MAIN_FIG1, "fig1_capability_envelope_main_absentred", ncols=len(MAIN_FIG1), absent="red",
                 panel_w=3.35, wspace=0.95)
        if "fig2_reversals" in only:
            pts = fig2_reversals(store, met, ALL, "fig2_majority_reversals")
            fig2_reversals(store, met, ALL, "fig2_majority_reversals_conditional", conditional=True)
            pts.assign(model=pts.model.map(name), dataset=pts.dataset.map(TITLES)).to_csv(
                OUT / "fig2_majority_reversals_points.csv", index=False)
            tot = pts.groupby(["dataset", "model"], sort=False)["global_rate"].sum().groupby(level=0, sort=False)
            print(pts.groupby(["dataset", "side"], sort=False)["global_rate"].describe()[
                ["25%", "50%", "75%", "max"]].round(3).to_string())
            print("total reversal (sum), median/max:\n", pd.DataFrame({"median": tot.median(), "max": tot.max()})
                  .round(3).to_string())
        if "table_main" in only:
            print(main_table(store, met).round(3).to_string(index=False))
        if "fig3_main_inside" in only:
            fig3(store, met, MAIN_FIG3, "fig3_reliability_curves_main_legend_inside", ncols=len(MAIN_FIG3),
                 legend_inside=True).to_csv(OUT / "fig3_main_curve_points.csv", index=False)
        if "fig3_main" in only:
            fig3(store, met, MAIN_FIG3, "fig3_reliability_curves_main", ncols=len(MAIN_FIG3)).to_csv(
                OUT / "fig3_main_curve_points.csv", index=False)
        _manifest(store, {"scope": only, "fig2_min_group": MIN_GROUP, "fig3_main": [TITLES[d] for d in MAIN_FIG3]})
        print(f"wrote {OUT}  ({only})")
        return
    pairs = pd.concat([near_tied_pairs(store, met, ds) for ds in ALL], ignore_index=True)
    pairs.assign(more_reliable=pairs.more_reliable.map(name), less_reliable=pairs.less_reliable.map(name)).to_csv(
        OUT / "rq1_near_tied_pairs.csv", index=False)
    ps = pairs_summary(pairs)
    ps.to_csv(OUT / "rq1_near_tied_pairs_summary.csv")

    for absent in ABSENT:
        fig1(store, met, MAIN, f"fig1_capability_envelope_main_absent{absent}", ncols=len(MAIN), absent=absent)
        fig1(store, met, ALL, f"fig1_capability_envelope_appendix_absent{absent}", ncols=3, absent=absent)
    fig2_reversals(store, met, ALL, "fig2_majority_reversals")
    fig2_reversals(store, met, ALL, "fig2_majority_reversals_conditional", conditional=True)
    fig2(store, met, MAIN, "fig2_representativeness_main")
    fig2(store, met, ALL, "fig2_representativeness_appendix")
    fig3(store, met, MAIN_FIG3, "fig3_reliability_curves_main", ncols=len(MAIN_FIG3))
    fig3(store, met, ALL, "fig3_reliability_curves_appendix", ncols=3).to_csv(OUT / "fig3_curve_points.csv",
                                                                             index=False)
    fig4(store, met, pairs, MAIN, "fig4_breadth_vs_reliability_main", ncols=len(MAIN))
    fig4(store, met, pairs, ALL, "fig4_breadth_vs_reliability_appendix", ncols=3)
    fig5(pairs, MAIN, "fig5_near_tied_pairs_main")
    fig5(pairs, [d for d in ALL if d != "SimpleQA"], "fig5_near_tied_pairs_appendix")
    t = summary_table(store, met)

    no_fig3 = [TITLES[d] for d in ALL if SP.fig3_candidates(met[d], store[d]["A"], top=1).empty]
    _manifest(store, {"fig3_omitted": no_fig3, "pair_window_pp": PAIR_PP * 100})
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(t.to_string(index=False))
        print(ps.to_string())
    print(f"wrote {OUT}  (fig3 omitted: {no_fig3})")


if __name__ == "__main__":
    main()
