"""
RQ3 paired accessibility profiles: before -> after stacked bars (Reliable / Brittle / Absent, accuracy tick, latent
diamond), the same visual vocabulary as RQ1 Fig. 1, for model changes (scaling steps, native thinking, prompted CoT,
prompt optimization). No new generation: pairs come from rq3_interventions' pair cache.

Outputs ($PAPER_REPRO_DIR/rq3_paired/<VERSION>/):
  fig_rq3_paired_main.pdf      3 principled examples, one per pattern (no prompt optimization / CoT examples, per user):
                               (a) accuracy barely moves, profile reshuffled   Gemma-3 4B -> 12B, MMLU (scaling)
                               (b) accuracy up, reliable share down            Qwen3-8B base -> thinking, MMLU
                               (c) reliability gains far beyond accuracy       Gemma-3 1B -> 4B, GSM8K (scaling)
  fig_rq3_paired_<family>.pdf  every pair of a family, one column per benchmark (appendix).
  rq3_paired_table.tex/.csv    per family x benchmark means: Acc, mu, R, L, Access@5, Reliable@5 before -> after,
                               share of inquiries with A_q up / down (appendix).
Usage: python paper/rq3_paired_profiles.py  (or: bash paper/reproduce.sh)
"""
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import rq3_interventions as Q  # noqa: E402

VERSION = "v1"
OUT = Q.PP.out_dir(None, "rq3_paired") / VERSION
COL = {"R": "#0ca30c", "B": "#fab219", "A": "#d03b3b", "acc": "#0b0b0b", "lat": "#2a78d6"}
MAIN = [("scaling", "mmlu", "Gemma-3 4B → Gemma-3 12B", "Scaling · MMLU", "accuracy barely moves; profile reshuffled"),
        ("thinking", "mmlu", "Qwen-3 8B: base → thinking", "Native thinking · MMLU", "accuracy up, reliable share down"),
        ("scaling", "gsm8k", "Gemma-3 1B → Gemma-3 4B", "Scaling · GSM8K", "reliability gains exceed accuracy gains")]
FAM_TITLE = {"scaling": "Scaling (consecutive sizes)", "thinking": "Native thinking (Qwen3)", "cot": "Prompted CoT",
             "popt": "Prompt optimization"}


def stats(a, y):
    return {"acc": y.mean(), "R": (a >= 1 - 1e-9).mean(), "L": (a > 1e-9).mean(), "mu": a.mean()}


def at5(a, n, k=5):
    c, n = np.rint(a * n).astype(int), n.astype(int)
    rel = np.mean([comb(ci, k) / comb(ni, k) for ci, ni in zip(c, n)])
    acc = np.mean([1 - comb(ni - ci, k) / comb(ni, k) for ci, ni in zip(c, n)])
    return acc, rel


def bar(ax, y, s, h=0.34):
    left = 0.0
    for key, v in (("R", s["R"]), ("B", s["L"] - s["R"]), ("A", 1 - s["L"])):
        ax.barh(y, v, left=left, height=h, color=COL[key], edgecolor="white", linewidth=0.6, zorder=2)
        left += v
    ax.plot([s["acc"]], [y], "|", color=COL["acc"], markersize=13, markeredgewidth=2.4, zorder=5)
    ax.plot([s["L"]], [y], "D", color=COL["lat"], markersize=6, markeredgecolor="white", markeredgewidth=0.7, zorder=5)


def legend_handles():
    return [Patch(color=COL["R"], label="Reliable"), Patch(color=COL["B"], label="Brittle"),
            Patch(color=COL["A"], label="Absent"),
            Line2D([], [], marker="|", color=COL["acc"], ls="none", markersize=11, markeredgewidth=2.2,
                   label="Accuracy"),
            Line2D([], [], marker="D", color=COL["lat"], ls="none", markersize=6, label="Latent")]


def find(pairs, fam, ds, label):
    for p in pairs:
        if p["family"] == fam and p["dataset"] == ds and p["label"] == label:
            return p
    raise KeyError((fam, ds, label))


def fig_main(pairs):
    rows = [find(pairs, f, d, l) for f, d, l, _, _ in MAIN]
    with plt.rc_context(Q.RC3):
        fig, ax = plt.subplots(figsize=(10.5, 2.75))
        yt, yl = [], []
        for i, (p, (_, _, lab, title, pattern)) in enumerate(zip(rows, MAIN)):
            s0, s1 = stats(p["A0"], p["Y0"]), stats(p["A1"], p["Y1"])
            y0 = i * 1.05
            bar(ax, y0 - 0.2, s0)
            bar(ax, y0 + 0.2, s1)
            if ":" in lab:                                     # "Qwen-3 8B: base → thinking"
                model, change = lab.split(": ")
                before, after = (f"{model} ({c})" for c in change.split(" → "))
            else:                                              # "Gemma-3 4B → Gemma-3 12B"
                before, after = lab.split(" → ")
            yt += [y0 - 0.2, y0 + 0.2]
            yl += [before, after]
            ax.text(-0.02, y0 - 0.44, f"{'abc'[i]}) {title}: {pattern}", transform=ax.get_yaxis_transform(),
                    ha="left", va="bottom", fontsize=11.5, fontweight="bold")
            dA, dY = p["A1"] - p["A0"], p["Y1"] - p["Y0"]
            ax.text(1.02, y0, f"Δacc {100 * (s1['acc'] - s0['acc']):+.1f}\nΔR   {100 * (s1['R'] - s0['R']):+.1f}\n"
                    f"$A_q$↑ {100 * (dA > 1e-9).mean():.0f}%  ↓ {100 * (dA < -1e-9).mean():.0f}%",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=10.5, family="monospace")
        ax.set_yticks(yt, yl)
        ax.set_ylim(len(rows) * 1.05 - 0.6, -0.72)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", ".25", ".5", ".75", "1"])
        ax.set_xlabel("Fraction of inquiries")
        ax.grid(axis="x", color="#e1e0d9", lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        fig.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.45, 0.88), ncol=5, frameon=False,
                   handlelength=1.3, columnspacing=1.0)
        _save(fig, "fig_rq3_paired_main")


def short_label(p):
    lab = p["label"]
    if p["family"] == "thinking":
        return lab.split(":")[0]                                   # "Qwen-3 8B"
    if p["family"] == "scaling":
        a, b = lab.split(" → ")
        fa, fb = a.rsplit(" ", 1), b.rsplit(" ", 1)
        if fa[0] == fb[0]:
            return f"{fa[0]} {fa[1]}→{fb[1]}"                      # "Qwen-3 0.6B→1.7B"
        return f"{a}→{b.replace('GPT-4.1-', '')}"                  # "GPT-4.1-Nano→Mini"
    if p["family"] == "popt":
        return f"{lab.split(':')[0]} · {p['sub']}"                 # "GPT-4.1-Mini · GEPA"
    return lab.split(":")[0]                                       # CoT: model name


def fig_family(pairs, fam, ncols_max=3):
    fp = [p for p in pairs if p["family"] == fam and not p["sub"].endswith("endpoint")]
    dss = [d for d in Q.BENCH if any(p["dataset"] == d for p in fp)]
    nmax = max(sum(p["dataset"] == d for p in fp) for d in dss)
    ncols = min(ncols_max, len(dss))
    nrows = -(-len(dss) // ncols)
    with plt.rc_context(Q.RC3):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, (0.62 * nmax + 1.0) * nrows), squeeze=False,
                                 gridspec_kw={"wspace": 1.0, "hspace": 0.35})
        for ax, ds in zip(axes.flat, dss):
            ps = [p for p in fp if p["dataset"] == ds]
            for i, p in enumerate(ps):
                s0, s1 = stats(p["A0"], p["Y0"]), stats(p["A1"], p["Y1"])
                bar(ax, i - 0.19, s0, h=0.32)
                bar(ax, i + 0.19, s1, h=0.32)
                ax.text(1.03, i, f"{100 * (s1['acc'] - s0['acc']):+.1f}\n{100 * (s1['R'] - s0['R']):+.1f}",
                        transform=ax.get_yaxis_transform(), va="center", fontsize=10, family="monospace")
            ax.set_yticks(range(len(ps)), [short_label(p) for p in ps], fontsize=10.5)
            ax.set_ylim(len(ps) - 0.5, -0.5)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1], ["0", ".5", "1"])
            ax.set_title(Q.TITLES[ds], loc="left")
            ax.text(1.03, -0.62, "Δacc\nΔR", transform=ax.get_yaxis_transform(), va="bottom", fontsize=9.5,
                    family="monospace", color="#6b6a64")
        for ax in list(axes.flat)[len(dss):]:
            ax.axis("off")
        fig.canvas.draw()
        top = max(a.get_position().y1 for a in axes.flat if a.axison)
        h_in = fig.get_size_inches()[1]
        fig.text(0.02, top + 0.55 / h_in, f"{FAM_TITLE[fam]}: before (upper bar) → after (lower bar)", ha="left",
                 va="bottom", fontsize=13.5, fontweight="bold")
        fig.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.5, top + 0.95 / h_in), ncol=5,
                   frameon=False, handlelength=1.3, columnspacing=1.0)
        _save(fig, f"fig_rq3_paired_{fam}")


def table(pairs):
    rows = []
    for p in pairs:
        if p["sub"].endswith("endpoint"):
            continue
        s0, s1 = stats(p["A0"], p["Y0"]), stats(p["A1"], p["Y1"])
        a0, r0 = at5(p["A0"], p["n"])
        a1, r1 = at5(p["A1"], p["n"])
        dA = p["A1"] - p["A0"]
        rows.append({"family": p["family"], "dataset": p["dataset"], **{f"{k}0": v for k, v in s0.items()},
                     **{f"{k}1": v for k, v in s1.items()}, "Acc5_0": a0, "Acc5_1": a1, "Rel5_0": r0, "Rel5_1": r1,
                     "up": (dA > 1e-9).mean(), "down": (dA < -1e-9).mean()})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "rq3_paired_per_pair.csv", index=False)
    g = d.groupby(["family", "dataset"], sort=False).mean(numeric_only=True)
    g["n"] = d.groupby(["family", "dataset"], sort=False).size()
    order = [(f, ds) for f in Q.FAMILIES for ds in Q.BENCH if (f, ds) in g.index]
    g = g.loc[order]
    g.to_csv(OUT / "rq3_paired_table.csv")
    fmt = lambda a, b: f"{100 * a:.1f}$\\to${100 * b:.1f}"  # noqa: E731
    fam_short = {"scaling": "Scaling", "thinking": "Thinking", "cot": "CoT", "popt": "Prompt opt."}
    lines = [r"\begin{tabular}{llr|cccccc|cc}", r"\toprule",
             r"Change & Benchmark & $n$ & Acc & $\mu$ & $R$ & $L$ & Access@5 & Reliable@5 & $A_q\uparrow$ & $A_q\downarrow$ \\",
             r"\midrule"]
    for (f, ds), r in g.iterrows():
        lines.append(f"{fam_short[f]} & {Q.TITLES[ds]} & {int(r.n)} & {fmt(r.acc0, r.acc1)} & {fmt(r.mu0, r.mu1)} & "
                     f"{fmt(r.R0, r.R1)} & {fmt(r.L0, r.L1)} & {fmt(r.Acc5_0, r.Acc5_1)} & {fmt(r.Rel5_0, r.Rel5_1)} & "
                     f"{100 * r.up:.0f}\\% & {100 * r.down:.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "rq3_paired_table.tex").write_text("\n".join(lines) + "\n")
    return g


def _save(fig, fname):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = Q.load_all()
    fig_main(pairs)
    for fam in Q.FAMILIES:
        fig_family(pairs, fam)
    g = table(pairs)
    print(g[["n", "acc0", "acc1", "R0", "R1", "L0", "L1", "up", "down"]].round(3).to_string())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
