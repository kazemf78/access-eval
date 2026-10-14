"""
RQ3 -- what changes in the accessibility profile when a model changes, that the change in benchmark accuracy misses?
No new generation or grading: everything comes from the tables in paper/data/. SimpleQA is left out on purpose.

Intervention families (each "pair" = one model change, before -> after, on inquiries + paraphrases shared by both):
  scaling    consecutive size steps AND smallest->largest per family (Qwen3 0.6-32B, Gemma-3 1-27B, GPT-4.1
             nano/mini/full), from the RQ1 per-question store (same inquiries as RQ1; Gemma-3 1B absent on MMLU/BFCL).
  thinking   Qwen3 base -> [with_thinking], 6 sizes, paper_v1 data (TruthfulQA 3-GPT majority labels, math fallback
             grader), pairwise common substrate (build_common_profiles on the two models).
  cot        regular prompt -> prompted CoT, TruthfulQA only, non-thinking models (sub-study: v00 paraphrases and
             v00 judges; table paper/data/rq3/cot_TruthfulQA.parquet).
  popt       original prompt -> GEPA / MIPROv2, 5 models, GSM8K + TruthfulQA (sub-study: v00 paraphrases, v00 judges,
             popt math loader; IFBench left out -- ~2.6 paraphrases per inquiry).
Every pair keeps inquiries with >= 5 shared paraphrases.

Per pair (no thresholds; only signs and the landmarks 0, 0.5, 1):
  d_acc, d_mu, d_R (A=1), d_L (A>0), d_brittle, d_Beff, d_D (plug-in, as in RQ1), log-split of d_mu into breadth/repeatability
  accessibility direction  up (A1>A0) / same / down  x  canonical change  flip-up (0->1) / same / flip-down (1->0)
  hidden share   = P(canonical unchanged | A changed)
  gross vs net   = P(up)+P(down) vs P(up)-P(down)
  majority flips = A crosses 0.5 (A0<0.5<A1 or A0>0.5>A1), vs canonical flips
  state transitions Absent/Brittle/Reliable (3x3)
Outputs: $PAPER_REPRO_DIR/rq3/figs/<VERSION>/ (figures, rq3_pairs.csv, rq3_cells.csv, rq3_transitions.csv, manifest).
Usage: python paper/rq3_interventions.py  (or: bash paper/reproduce.sh)
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
import rq1_selection_pass as SP  # noqa: E402  (sets KB.TQA_RULE = maj3_binary)
from rq1_render import NAMES, RC, C, TITLES  # noqa: E402
# v7: RQ3 figures are placed at \textwidth (~5.5 in); draw them on narrower canvases with larger fonts so text stays
# >= ~7 pt after scaling.
RC3 = {**RC, "font.size": 12.5, "axes.titlesize": 13.5, "axes.labelsize": 12.5, "xtick.labelsize": 11.5,
       "ytick.labelsize": 11.5, "legend.fontsize": 11.5}
from state_bars_survival import bd_metrics  # noqa: E402
from access_eval.analysis.common_profiles import build_common_profiles  # noqa: E402
from access_eval.analysis.profile_comparison import build_paired_profiles  # noqa: E402

VERSION = "v7"   # v7: readability pass on all RQ3 figures (user): RC3 larger fonts, narrower canvases (less
#     shrink when placed at \textwidth), bigger in-figure annotations/markers, shared x label on scatters, A_q notation.
# v6: v6: NEW fig_rq3_two_views_compact (main text, 7 rows): benchmarks on which every intervention
#     family was run (TruthfulQA, GSM8K), size steps only, GEPA+MIPROv2 pooled into one prompt-optimization row.
# v5: v5: fig_rq3_dacc_vs_dR diagonal counts use a 1e-9 tie tolerance (exact ties were counted as
#     "above" via float noise, e.g. MATH-500 13/16 -> 11 above, 3 tied, 2 below); ties shown separately.
# v4: v4: D renamed "repeatability" (labels + file name fig_rq3_breadth_vs_repeatability); figure moved to appendix.
# v3: trimmed main two-views (no smallest->largest rows) + full appendix version; y=x counts
import paper_paths as PP  # noqa: E402
OUT = PP.out_dir(None, "rq3") / "figs" / VERSION
TABLES = PP.DATA / "rq3"
CACHE = TABLES / "rq3_pairs_cache.pkl"   # all pairs; delete it (or REBUILD_RQ3_PAIRS=1) to rebuild from the tables
BENCH = ["TruthfulQA", "gsm8k", "math_500", "mmlu", "bfcl"]
K = 5
SIZES = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]
LADDERS = {"Qwen3": [f"Qwen/Qwen3-{s}" for s in SIZES],
           "Gemma-3": [f"google/gemma-3-{s}-it" for s in ("1b", "4b", "12b", "27b")],
           "GPT-4.1": ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]}
FAMILIES = ["scaling", "thinking", "cot", "popt"]
FAM_LABEL = {"scaling": "Scaling (size step)", "thinking": "Native thinking", "cot": "Prompted CoT",
             "popt": "Prompt optimization"}
FAM_COLOR = {"scaling": "#2a78d6", "thinking": "#1baf7a", "cot": "#eb6834", "popt": "#4a3aa7"}
DIR_COLORS = {"up": "#1baf7a", "same": "#d9d8d1", "down": "#e34948"}
TITLES = {**TITLES, "gsm8k_test": "GSM8K"}


def name(m):
    return NAMES.get(m, m.split("/")[-1])


# ------------------------------------------------------------------ loading pairs
def _pair(fam, ds, label, before, after, A0, A1, Y0, Y1, n, sub=""):
    return {"family": fam, "dataset": ds, "label": label, "before": before, "after": after, "sub": sub,
            "A0": np.asarray(A0, float), "A1": np.asarray(A1, float), "Y0": np.asarray(Y0, float),
            "Y1": np.asarray(Y1, float), "n": np.asarray(n, float)}


def scaling_pairs():
    with open(SP.OUT / "rq1_per_question.pkl", "rb") as f:
        store = pickle.load(f)
    out = []
    for ds in BENCH:
        d = store[ds]
        for fam, ladder in LADDERS.items():
            ms = [m for m in ladder if m in d["A"]]
            steps = list(zip(ms, ms[1:])) + ([(ms[0], ms[-1])] if len(ms) > 2 else [])
            for a, b in steps:
                sub = "endpoint" if (a, b) == (ms[0], ms[-1]) and len(ms) > 2 else "step"
                out.append(_pair("scaling", ds, f"{name(a)} → {name(b)}", a, b, d["A"][a], d["A"][b], d["Y"][a],
                                 d["Y"][b], d["n_para"], sub=f"{fam}:{sub}"))
    return out


def thinking_pairs(paths):
    out = []
    for ds in BENCH:
        agg = KB.load(ds, paths)
        for s in SIZES:
            b, t = f"Qwen/Qwen3-{s}", f"Qwen/Qwen3-{s}[with_thinking]"
            prof = build_common_profiles(agg, [b, t])
            prof = prof[prof["n_para"] >= K]
            out.append(_pair("thinking", ds, f"Qwen-3 {s}: base → thinking", b, t,
                             prof[f"correct_frac_{b}"], prof[f"correct_frac_{t}"],
                             prof[f"orig_label_{b}"] == "correct", prof[f"orig_label_{t}"] == "correct",
                             prof["n_para"]))
        print(f"thinking {ds}: {len(prof)} inquiries (32B pair)")
    return out


def _paired(prof, tag_a, tag_b):
    fa = "correct_frac_a" if "correct_frac_a" in prof else "pcorrect_a"
    fb = fa[:-1] + "b"
    prof = prof[prof["n_para"] >= K]
    return (prof[fa], prof[fb], prof["orig_label_a"] == "correct", prof["orig_label_b"] == "correct", prof["n_para"])


def _tables_index():
    return json.loads((TABLES / "index.json").read_text())


def cot_pairs(paths):
    ds = "TruthfulQA"
    models = _tables_index()["cot_TruthfulQA"]["models"]
    combined = pd.read_parquet(TABLES / "cot_TruthfulQA.parquet")   # <model>__regular / <model>__cot rows
    cot_models = {m[:-5] for m in combined["model"].unique() if m.endswith("__cot")}
    out = []
    for m in models:
        if m not in cot_models or "[with_thinking]" in m:
            continue
        ta, tb = f"{m}__regular", f"{m}__cot"
        prof = build_paired_profiles(combined, ta, tb, min_agreement_fraction=0.5)
        A0, A1, Y0, Y1, n = _paired(prof, ta, tb)
        if len(A0):
            out.append(_pair("cot", ds, f"{name(m)}: regular → CoT prompt", m, m, A0, A1, Y0, Y1, n))
    print(f"cot TruthfulQA: {len(out)} models")
    return out


def popt_pairs(paths):
    out = []
    for key, t in _tables_index().items():          # table order = (dataset, model, method) of the original loop
        if not key.startswith("popt_"):
            continue
        ds, m, meth, ta, tb = t["dataset"], t["model"], t["method"], t["tag_a"], t["tag_b"]
        prof = build_paired_profiles(pd.read_parquet(TABLES / f"{key}.parquet"), ta, tb)
        A0, A1, Y0, Y1, n = _paired(prof, ta, tb)
        if len(A0):
            out.append(_pair("popt", "gsm8k" if ds == "gsm8k_test" else ds, f"{name(m)}: orig → {meth}", m, m,
                             A0, A1, Y0, Y1, n, sub=meth))
    print(f"popt: {len(out)} model x method x dataset pairs")
    return out


def load_all():
    """All before/after pairs: the shipped pair cache, or (REBUILD_RQ3_PAIRS=1 / no cache) rebuilt from the tables."""
    import os
    if CACHE.exists() and not os.environ.get("REBUILD_RQ3_PAIRS"):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    paths = None
    pairs = scaling_pairs() + thinking_pairs(paths) + cot_pairs(paths) + popt_pairs(paths)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "rq3_pairs_cache_rebuilt.pkl", "wb") as f:
        pickle.dump(pairs, f)
    return pairs


# ------------------------------------------------------------------ metrics
def state(a):
    return np.where(a <= 1e-9, 0, np.where(a >= 1 - 1e-9, 2, 1))


def pair_metrics(p):
    A0, A1, Y0, Y1, n = p["A0"], p["A1"], p["Y0"], p["Y1"], p["n"]
    dA, dY = A1 - A0, Y1 - Y0
    up, down = dA > 1e-9, dA < -1e-9
    moved = up | down
    mu0, B0, D0 = bd_metrics(A0, n)
    mu1, B1, D1 = bd_metrics(A1, n)
    r = {k: p[k] for k in ("family", "dataset", "label", "before", "after", "sub")}
    r.update({"n_q": len(A0), "acc0": Y0.mean(), "acc1": Y1.mean(), "d_acc": dY.mean(), "mu0": mu0, "mu1": mu1,
              "d_mu": dA.mean(), "d_R": (A1 >= 1 - 1e-9).mean() - (A0 >= 1 - 1e-9).mean(),
              "d_L": (A1 > 1e-9).mean() - (A0 > 1e-9).mean(),
              "d_brittle": (state(A1) == 1).mean() - (state(A0) == 1).mean(),
              "d_Beff": B1 - B0, "d_D": D1 - D0,
              "dlog_B": np.log(B1 / B0) if B0 > 0 and B1 > 0 else np.nan,
              "dlog_D": np.log(D1 / D0) if D0 > 0 and D1 > 0 else np.nan,
              "dlog_mu": np.log(mu1 / mu0) if mu0 > 0 and mu1 > 0 else np.nan,
              "up": up.mean(), "down": down.mean(), "same": (~moved).mean(),
              "flip_up": (dY > 0).mean(), "flip_down": (dY < 0).mean(), "canon_same": (dY == 0).mean(),
              "hidden_share": ((moved & (dY == 0)).sum() / moved.sum()) if moved.any() else np.nan,
              "opposite_share": ((moved & (np.sign(dA) == -np.sign(dY)) & (dY != 0)).sum() / moved.sum())
              if moved.any() else np.nan,
              "up_canon_same": (up & (dY == 0)).mean(), "down_canon_same": (down & (dY == 0)).mean(),
              "maj_up": ((A0 < 0.5) & (A1 > 0.5)).mean(), "maj_down": ((A0 > 0.5) & (A1 < 0.5)).mean(),
              "maj_flip_canon_same": ((((A0 < 0.5) & (A1 > 0.5)) | ((A0 > 0.5) & (A1 < 0.5))) & (dY == 0)).mean()})
    r["gross"], r["net"] = r["up"] + r["down"], r["up"] - r["down"]
    # d_mu = gain - loss exactly: mean positive / negative part of dA over ALL inquiries; plus the average size of a move
    r["gain"], r["loss"] = np.clip(dA, 0, None).mean(), np.clip(-dA, 0, None).mean()
    r["size_up"] = dA[up].mean() if up.any() else np.nan
    r["size_down"] = -dA[down].mean() if down.any() else np.nan
    s0, s1 = state(A0), state(A1)
    trans = {f"T_{a}{b}": ((s0 == a) & (s1 == b)).mean() for a in range(3) for b in range(3)}
    r.update(trans)
    return r


def cell_key(r):
    if r["family"] == "scaling":
        return "scaling-step" if r["sub"].endswith("step") else "scaling-endpoint"
    if r["family"] == "popt":
        return f"popt-{r['sub']}"
    return r["family"]


CELL_LABEL = {"scaling-step": "Scaling · size step", "scaling-endpoint": "Scaling · smallest→largest",
              "thinking": "Native thinking", "cot": "Prompted CoT", "popt-GEPA": "Prompt opt. · GEPA",
              "popt-MIPROv2": "Prompt opt. · MIPROv2", "popt": "Prompt optimization"}
CELL_ORDER = list(CELL_LABEL)


# ------------------------------------------------------------------ figures
def save(fig, fname):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_two_views(cells, fname="fig_rq3_two_views", drop=(), row_h=0.78):
    """Per (intervention cell, benchmark): canonical view (flip up / same / flip down) vs accessibility view
    (A up / same / down), mean over the cell's pairs."""
    rows = cells[~cells["cell"].isin(drop)].sort_values(["cell_order", "ds_order"])
    with plt.rc_context(RC3):
        fig, ax = plt.subplots(figsize=(8.2, row_h * len(rows) + 1.3))
        ylab = []
        for i, r in enumerate(rows.itertuples()):
            for dy, parts, tag in ((-0.18, (r.flip_up, r.canon_same, r.flip_down), "benchmark label"),
                                   (0.18, (r.up, r.same, r.down), "accessibility $A_q$")):
                left = 0
                for v, col in zip(parts, (DIR_COLORS["up"], DIR_COLORS["same"], DIR_COLORS["down"])):
                    ax.barh(i + dy, v, left=left, height=0.34, color=col, edgecolor="white", linewidth=0.6)
                    if v >= 0.08 and col != DIR_COLORS["same"]:
                        ax.text(left + v / 2, i + dy, f"{v:.0%}", ha="center", va="center", fontsize=10.5,
                                color="white", fontweight="bold")
                    left += v
                ax.text(1.015, i + dy, tag, va="center", fontsize=10, color=C["muted"])
            ax.text(1.29, i, f"Δacc {100 * r.d_acc:+5.1f}\nΔμ   {100 * r.d_mu:+5.1f} = +{100 * r.gain:.1f} − {100 * r.loss:.1f}",
                    va="center", fontsize=10.5, color="#222", family="monospace")
            ylab.append(f"{CELL_LABEL[r.cell]} · {TITLES[r.dataset]}  ({r.n_pairs})")
        ax.set_yticks(range(len(rows)), ylab)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", ".25", ".5", ".75", "1"])
        ax.set_xlabel("Share of inquiries (mean over the model changes in each row)")
        for k in range(1, len(rows)):
            if rows.iloc[k].cell != rows.iloc[k - 1].cell:
                ax.axhline(k - 0.5, color="#8a8a85", lw=0.8)
        handles = [Patch(color=DIR_COLORS["up"], label="improves"),
                   Patch(color=DIR_COLORS["same"], label="unchanged"),
                   Patch(color=DIR_COLORS["down"], label="worsens")]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
        ax.text(1.29, -0.62, "points;  Δμ = gain − loss", fontsize=9.5, color=C["muted"], va="bottom")
        save(fig, fname)


QUAD = {(1, 1): "B↑ D↑", (-1, 1): "B↓ D↑", (1, -1): "B↑ D↓", (-1, -1): "B↓ D↓"}   # v7: compact corner labels


def fig_scatter(pr, x, y, xl, yl, fname, diag=True, quadrants=False, diag_counts=None):
    dss = [d for d in BENCH if d in set(pr.dataset)]
    with plt.rc_context(RC3):
        fig, axes = plt.subplots(1, len(dss), figsize=(2.55 * len(dss), 3.0), gridspec_kw={"wspace": 0.42})
        for ax, ds in zip(axes, dss):
            d = pr[pr.dataset == ds]
            lim = np.nanmax(np.abs(d[[x, y]].to_numpy())) * 1.1 + 1e-3
            ax.axhline(0, color="#55554f", lw=0.9, zorder=1)
            ax.axvline(0, color="#55554f", lw=0.9, zorder=1)
            if quadrants:
                ax.plot([-0.6 * lim, 0.6 * lim], [0.6 * lim, -0.6 * lim], color=C["muted"], lw=0.8, ls=":",
                        zorder=1)   # kept short so it never runs into the corner labels
                for (sx, sy), txt in QUAD.items():
                    n_in = int(((np.sign(d[x]) == sx) & (np.sign(d[y]) == sy)).sum())
                    ax.text(sx * 0.95 * lim, sy * 0.95 * lim, f"{txt}\nn={n_in}", fontsize=9.5, color=C["muted"],
                            ha="right" if sx > 0 else "left", va="top" if sy > 0 else "bottom")
            if diag:
                ax.plot([-lim, lim], [-lim, lim], color=C["muted"], lw=0.8, ls="--")
            if diag_counts:
                diff = d[y] - d[x]
                above, below = int((diff > 1e-9).sum()), int((diff < -1e-9).sum())
                tied = len(d) - above - below
                ax.text(-0.95 * lim, 0.95 * lim, f"{diag_counts[0]}: {above}/{len(d)}", fontsize=10,
                        color="#222", ha="left", va="top",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1}, zorder=5)
                ax.text(0.95 * lim, -0.95 * lim, f"{diag_counts[1]}: {below}/{len(d)}"
                        + (f"\n({tied} tied)" if tied else ""), fontsize=10, color="#222", ha="right", va="bottom",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1}, zorder=5)
            for fam in FAMILIES:
                f = d[d.family == fam]
                ax.scatter(f[x], f[y], s=40, color=FAM_COLOR[fam], edgecolors="white", linewidths=0.6, zorder=3)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, symmetric=True))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, symmetric=True))
            ax.set_title(TITLES[ds], loc="left")
            if ds == dss[0]:
                ax.set_ylabel(yl)
        handles = [Line2D([], [], marker="o", ls="none", color=FAM_COLOR[f], label=FAM_LABEL[f]) for f in FAMILIES]
        if diag:
            handles.append(Line2D([], [], color=C["muted"], ls="--", label="y = x"))
        if quadrants:
            handles.append(Line2D([], [], color=C["muted"], ls=":", label="μ unchanged (Δlog B + Δlog D = 0)"))
        fig.canvas.draw()   # aspect="equal" shrinks the axes; anchor legend and x label to where they actually end up
        top = max(a.get_position().y1 for a in axes)
        bottom = min(a.get_position().y0 for a in axes)
        fig.text(0.5, bottom - 0.13, xl, ha="center", va="top", fontsize=RC3["axes.labelsize"])
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, top + 0.08), ncol=3, frameon=False,
                   columnspacing=1.2, handletextpad=0.3)
        save(fig, fname)


def fig_majority(cells):
    rows = cells.sort_values(["cell_order", "ds_order"])
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.2))
        y = np.arange(len(rows))
        ax.barh(y - 0.18, rows.flip_up + rows.flip_down, height=0.32, color="#8a8a85",
                label="benchmark label flips (right↔wrong on the canonical wording)")
        ax.barh(y + 0.18, rows.maj_up + rows.maj_down - rows.maj_flip_canon_same, height=0.32, color=C["latent"],
                label="majority of formulations flips (A crosses 0.5) AND label flips")
        ax.barh(y + 0.18, rows.maj_flip_canon_same, left=rows.maj_up + rows.maj_down - rows.maj_flip_canon_same,
                height=0.32, color="#9ec5f4", label="majority of formulations flips, label does NOT flip")
        ax.set_yticks(y, [f"{CELL_LABEL[r.cell]} · {TITLES[r.dataset]}" for r in rows.itertuples()])
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xlabel("Share of inquiries (mean over model changes)")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)
        ax.grid(axis="x", color=C["grid"], lw=0.7)
        ax.set_axisbelow(True)
        save(fig, "fig_rq3_majority_flips")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = load_all()
    pr = pd.DataFrame([pair_metrics(p) for p in pairs])
    pr["cell"] = pr.apply(cell_key, axis=1)
    pr.drop(columns=[c for c in pr if c.startswith("T_")]).to_csv(OUT / "rq3_pairs.csv", index=False)
    tcols = [c for c in pr if c.startswith("T_")]
    pr.groupby(["cell", "dataset"])[tcols].mean().rename(
        columns={f"T_{a}{b}": f"{'ABR'[a]}->{'ABR'[b]}" for a in range(3) for b in range(3)}).to_csv(
        OUT / "rq3_transitions.csv")

    num = ["d_acc", "d_mu", "d_R", "d_L", "d_brittle", "d_Beff", "d_D", "up", "down", "same", "flip_up",
           "flip_down", "canon_same", "hidden_share", "opposite_share", "gross", "net", "maj_up", "maj_down",
           "maj_flip_canon_same", "n_q", "gain", "loss", "size_up", "size_down"]
    cells = pr.groupby(["cell", "dataset"])[num].mean().reset_index()
    cells["n_pairs"] = pr.groupby(["cell", "dataset"]).size().to_numpy()
    cells["gross_over_net"] = cells["gross"] / cells["net"].abs().clip(lower=1e-3)
    cells["cell_order"] = cells["cell"].map(CELL_ORDER.index)
    cells["ds_order"] = cells["dataset"].map(BENCH.index)
    cells.sort_values(["cell_order", "ds_order"]).to_csv(OUT / "rq3_cells.csv", index=False)

    fig_two_views(cells, "fig_rq3_two_views_main", drop=("scaling-endpoint",), row_h=0.52)
    # compact main-text version: rule = benchmarks every intervention family was run on; GEPA+MIPROv2 pooled
    pc = pr[(pr.cell != "scaling-endpoint") & pr.dataset.isin(["TruthfulQA", "gsm8k"])].copy()
    pc["cell"] = pc["cell"].where(~pc["cell"].str.startswith("popt"), "popt")
    cc = pc.groupby(["cell", "dataset"])[num].mean().reset_index()
    cc["n_pairs"] = pc.groupby(["cell", "dataset"]).size().to_numpy()
    cc["cell_order"] = cc["cell"].map(CELL_ORDER.index)
    cc["ds_order"] = cc["dataset"].map(BENCH.index)
    cc.sort_values(["cell_order", "ds_order"]).to_csv(OUT / "rq3_cells_compact.csv", index=False)
    fig_two_views(cc, "fig_rq3_two_views_compact", row_h=0.5)
    fig_two_views(cells, "fig_rq3_two_views_appendix")
    pts = pr[pr.cell != "scaling-endpoint"].assign(d_acc=lambda d: 100 * d.d_acc, d_R=lambda d: 100 * d.d_R)
    fig_scatter(pts, "d_acc", "d_R", "Δ benchmark accuracy (points)", "Δ reliable share R (points)",
                "fig_rq3_dacc_vs_dR", diag_counts=("above", "below"))
    fig_scatter(pr[pr.cell != "scaling-endpoint"], "d_acc", "d_mu", "Δ benchmark accuracy",
                "Δ mean accessibility μ", "fig_rq3_dacc_vs_dmu")
    fig_scatter(pr[pr.cell != "scaling-endpoint"], "dlog_B", "dlog_D", "Δ log effective breadth B   (B↑ = broader, D↑ = more repeatable)",
                "Δ log repeatability D", "fig_rq3_breadth_vs_repeatability", diag=False, quadrants=True)
    fig_majority(cells)

    (OUT / "manifest.json").write_text(json.dumps({
        "version": VERSION, "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "benchmarks": [TITLES[d] for d in BENCH], "min_shared_paraphrases": K,
        "families": {c: int((pr.cell == c).sum()) for c in CELL_ORDER},
        "notes": ["SimpleQA excluded", "cot + popt are sub-studies on v00 paraphrases/judges (not regraded)",
                  "IFBench excluded (too few paraphrases)", "no thresholds: signs and landmarks 0/0.5/1 only"],
        "files": sorted(p.name for p in OUT.iterdir())}, indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        print(cells.sort_values(["cell_order", "ds_order"])[
            ["cell", "dataset", "n_pairs", "n_q", "d_acc", "d_mu", "d_R", "d_L", "up", "down", "flip_up",
             "flip_down", "hidden_share", "opposite_share", "gross", "net", "maj_up", "maj_down",
             "maj_flip_canon_same"]].to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
