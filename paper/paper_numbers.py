"""
Every number quoted in the paper text, computed in one place from the paper's data products.

Numbers that no other script outputs (they were first computed ad hoc while writing) are computed here from the RQ1
per-question store and the RQ3 pair table; numbers that another script already outputs are read from that script's
output. Result: paper_numbers.json (machine-readable, keyed by paper location) + paper_numbers.md (readable table).

Inputs (PAPER_REPRO_DIR-aware, see paper_paths.py / REPRODUCE.md):
  RQ1 store rq1_per_question.pkl      <- rq1_selection_pass.py
  fig3_stats.json, access_reliable_tradeoff.json  <- rq1_fig3_stats.py, access_reliable_tradeoff.py
  rq2_disagreement.csv, rq2_access_reliable_at_k.csv <- rq2_decoding_vs_formulation.py
  rq3_pairs.csv, rq3_cells.csv       <- rq3_interventions.py
  null_gsm8k.json                     <- rerun_null/analyze.py
  equal_cost_budget/summary.csv       <- equal_cost_budget.py
Usage: python paper/paper_numbers.py  (or: bash paper/reproduce.sh)
"""
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import paper_paths as PP
import rq1_render as R
import rq2_decoding_vs_formulation as R2
import rq3_interventions as Q
import equal_cost_budget as E

EXCLUDE_FROM_RERUN_BASELINE = ["meta-llama/Llama-3.1-8B-Instruct"]   # Feb-2026 answers drift vs same-env rerun
NT = ["TruthfulQA", "gsm8k", "math_500", "mmlu", "bfcl"]            # near-tied analyses exclude SimpleQA


def first_existing(*paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError(paths)


def rq1(store, out):
    T = R.TITLES
    per = {}
    for ds, s in store.items():
        ms = s["models"]
        A, Y = s["A"], s["Y"]
        acc = np.array([Y[m].mean() for m in ms]); mu = np.array([A[m].mean() for m in ms])
        Rr = np.array([(A[m] >= 1).mean() for m in ms]); L = np.array([(A[m] > 0).mean() for m in ms])
        rev = np.array([((Y[m] == 1) & (A[m] < .5)).mean() + ((Y[m] == 0) & (A[m] > .5)).mean() for m in ms])
        cond = lambda f: np.array([f(A[m], Y[m]) for m in ms])  # noqa: E731
        fail_any = cond(lambda a, y: (a[y == 0] > 0).mean() if (y == 0).any() else np.nan)
        fail_maj = cond(lambda a, y: (a[y == 0] > .5).mean() if (y == 0).any() else np.nan)
        succ_any = cond(lambda a, y: (a[y == 1] < 1).mean() if (y == 1).any() else np.nan)
        kt = lambda a, b: float(kendalltau(a, b)[0])  # noqa: E731
        per[T[ds]] = {
            "n_models": len(ms), "n_inquiries": int(len(s["idx"])),
            "formulations_median": float(np.median(s["n_para"])), "formulations_min": int(np.min(s["n_para"])),
            "formulations_max": int(np.max(s["n_para"])),
            "brittle_median_pct": 100 * float(np.median(L - Rr)), "acc_minus_R_median_pct": 100 * float(np.median(acc - Rr)),
            "L_minus_acc_median_pct": 100 * float(np.median(L - acc)),
            "reversal_median_pct": 100 * float(np.median(rev)), "reversal_max_pct": 100 * float(rev.max()),
            "cond_fail_any_median_pct": 100 * float(np.nanmedian(fail_any)),
            "cond_fail_any_range_pct": [100 * float(np.nanmin(fail_any)), 100 * float(np.nanmax(fail_any))],
            "cond_fail_majority_median_pct": 100 * float(np.nanmedian(fail_maj)),
            "cond_fail_majority_range_pct": [100 * float(np.nanmin(fail_maj)), 100 * float(np.nanmax(fail_maj))],
            "cond_succ_any_fail_median_pct": 100 * float(np.nanmedian(succ_any)),
            "cond_succ_any_fail_range_pct": [100 * float(np.nanmin(succ_any)), 100 * float(np.nanmax(succ_any))],
            "kendall_acc_mu": kt(acc, mu), "kendall_acc_R": kt(acc, Rr), "kendall_R_L": kt(Rr, L)}
    out["rq1_per_benchmark"] = per
    nt = [per[R.TITLES[d]] for d in NT]
    out["rq1_ranges"] = {
        "brittle_median_pct_GSM8K_MATH_BFCL": [min(per[k]["brittle_median_pct"] for k in ("GSM8K", "MATH-500", "BFCL")),
                                               max(per[k]["brittle_median_pct"] for k in ("GSM8K", "MATH-500", "BFCL"))],
        "kendall_acc_R_all": [min(p["kendall_acc_R"] for p in per.values()), max(p["kendall_acc_R"] for p in per.values())],
        "kendall_acc_mu_all": [min(p["kendall_acc_mu"] for p in per.values()), max(p["kendall_acc_mu"] for p in per.values())],
        "kendall_R_L_all": [min(p["kendall_R_L"] for p in per.values()), max(p["kendall_R_L"] for p in per.values())],
        "inquiries_range_excl_MATH": [min(p["n_inquiries"] for k, p in per.items() if k != "MATH-500"),
                                      max(p["n_inquiries"] for k, p in per.items() if k != "MATH-500")],
        "formulations_median_range": [min(p["formulations_median"] for p in nt + [per["SimpleQA"]]),
                                      max(p["formulations_median"] for p in nt + [per["SimpleQA"]])]}
    # near-tied accuracy pairs: how often the more accurate model has the smaller reliable share
    # same benchmark order as rq1_render/rq1_fig3_stats (all six), so the shared bootstrap RNG gives identical CIs
    met = _met(store)
    pairs = pd.concat([R.near_tied_pairs(store, met, d) for d in R.ALL], ignore_index=True)
    pairs = pairs[pairs.dataset.isin(NT)]
    acc_hi = np.array([met[r.dataset].loc[r.more_reliable, "acc"] for r in pairs.itertuples()])
    acc_lo = np.array([met[r.dataset].loc[r.less_reliable, "acc"] for r in pairs.itertuples()])
    reversed_ = acc_hi < acc_lo
    out["rq1_near_tied"] = {"pairs": int(len(pairs)), "median_dacc_pp": 100 * float(pairs.d_acc.median()),
                            "dR_gt_dacc": int((pairs.d_R > pairs.d_acc).sum()),
                            "more_accurate_is_less_reliable": int(reversed_.sum()),
                            "more_accurate_is_less_reliable_pct": 100 * float(reversed_.mean()),
                            "of_which_significant": int((reversed_ & pairs.significant.to_numpy()).sum())}
    # deployment paragraph: expected attempts per success on latent inquiries, BFCL near-equal-mu pair
    s = store["bfcl"]
    nm = {R.name(m): m for m in s["models"]}
    ea = {}
    for x in ("GPT-5.4-Mini", "Qwen-3 1.7B"):
        a = s["A"][nm[x]]
        ea[x] = {"mu": float(a.mean()), "E_inv_A_given_latent": float((1 / a[a > 0]).mean())}
    ea["ratio_Qwen_over_GPT"] = ea["Qwen-3 1.7B"]["E_inv_A_given_latent"] / ea["GPT-5.4-Mini"]["E_inv_A_given_latent"]
    out["discussion_attempts_bfcl"] = ea


_MET = {}


def _met(store):
    if not _MET:
        import rq1_selection_pass as SP
        for ds, s in store.items():
            _MET[ds] = pd.DataFrame.from_records([SP.metrics(s["A"][m], s["Y"][m]) for m in s["models"]],
                                                 index=s["models"]).sort_values("acc", ascending=False)
    return _MET


def from_script_outputs(out):
    f3 = first_existing(R.OUT / "fig3_stats.json", PP.REFERENCE / "rq1_figs/figs/v4/fig3_stats.json")
    tr = first_existing(R.OUT / "access_reliable_tradeoff.json", PP.REFERENCE / "rq1_figs/figs/v4/access_reliable_tradeoff.json")
    out["rq1_fig3_stats"] = json.loads(f3.read_text())
    out["rq1_access_reliable_tradeoff"] = json.loads(tr.read_text())
    out["_sources"] = {"fig3_stats": str(f3), "access_reliable_tradeoff": str(tr)}
    # RQ2
    d = pd.read_csv(R2.OUT / "rq2_disagreement.csv")
    k = pd.read_csv(R2.OUT / "rq2_access_reliable_at_k.csv")
    k5 = k[k.k == 5].assign(dacc=lambda x: 100 * (x.access_form - x.access_decode),
                            drel=lambda x: 100 * (x.reliable_form - x.reliable_decode))
    sig_acc = (k5.access_gap_lo > 0).sum()
    sig_rel = (k5.reliable_gap_lo > 0).sum()
    excl = ~((k5.dataset == "SimpleQA") & (k5.model == "LLaMA-3.1 8B"))
    out["rq2"] = {"settings": int(len(d)), "cross_gt_same_T07_significant": int((d.excess_T07_lo_pp > 0).sum()),
                  "excess_T07_pp_range_excl_llama_sqa": [float(d.excess_T07_pp[d.excess_T07_lo_pp > 0].min()),
                                                         float(d.excess_T07_pp.max())],
                  "same_T0_pct_range": [100 * float(d.same_T0.min()), 100 * float(d.same_T0.max())],
                  "cross_T0_pct_max": 100 * float(d.cross_T0.max()),
                  "access5_gap_pp_range": [float(k5.dacc[excl].min()), float(k5.dacc.max())],
                  "reliable5_gap_pp_range": [float(-k5.drel[excl].max()), float(-k5.drel.min())],
                  "access5_significant": int(sig_acc), "reliable5_significant": int(sig_rel),
                  "qwen8b_tqa_same_vs_cross_T07_pct": [100 * float(d[(d.dataset == "TruthfulQA") & (d.model == "Qwen-3 8B")].same_T07.iloc[0]),
                                                       100 * float(d[(d.dataset == "TruthfulQA") & (d.model == "Qwen-3 8B")].cross_T07.iloc[0])]}
    # RQ3
    p = pd.read_csv(Q.OUT / "rq3_pairs.csv")
    p = p[p.cell != "scaling-endpoint"]
    c = pd.read_csv(Q.OUT / "rq3_cells.csv")
    th_mmlu = c[(c.cell == "thinking") & (c.dataset == "mmlu")].iloc[0]
    diag = {}
    for ds, g in p.groupby("dataset"):
        dd = g.d_R - g.d_acc
        diag[R.TITLES.get(ds, ds)] = {"above": int((dd > 1e-9).sum()), "below": int((dd < -1e-9).sum()),
                                      "tied": int((dd.abs() <= 1e-9).sum()), "n": int(len(g))}
    th = p[(p.family == "thinking") & (p.dataset == "mmlu")].set_index("label")
    improving = p[p.d_acc > 0]
    near0 = p[p.d_acc.abs() <= 0.01 + 1e-9]
    gsm = c[c.dataset == "gsm8k"].set_index("cell")
    out["rq3"] = {"n_changes": int(len(p)),
                  "thinking_mmlu_mean": {"d_acc_pp": 100 * float(th_mmlu.d_acc), "up_pct": 100 * float(th_mmlu.up),
                                         "down_pct": 100 * float(th_mmlu.down), "d_R_pp": 100 * float(th_mmlu.d_R)},
                  "thinking_mmlu_qwen8b": {"d_acc_pp": 100 * float(th.loc["Qwen-3 8B: base → thinking", "d_acc"]),
                                           "d_R_pp": 100 * float(th.loc["Qwen-3 8B: base → thinking", "d_R"])},
                  "thinking_mmlu_qwen32b": {"d_acc_pp": 100 * float(th.loc["Qwen-3 32B: base → thinking", "d_acc"]),
                                            "d_R_pp": 100 * float(th.loc["Qwen-3 32B: base → thinking", "d_R"])},
                  # MEAN over changes (= what the two-views figure shows) and MEDIAN over changes (older text)
                  "gsm8k_moved_pct_mean": {k2: 100 * float(gsm.loc[k2, "up"] + gsm.loc[k2, "down"]) for k2 in gsm.index},
                  "gsm8k_down_pct_mean": {k2: 100 * float(gsm.loc[k2, "down"]) for k2 in gsm.index},
                  "gsm8k_moved_pct_median": {k2: 100 * float((g.up + g.down).median()) for k2, g in
                                             p[p.dataset == "gsm8k"].groupby("cell")},
                  "gsm8k_down_pct_median": {k2: 100 * float(g.down.median()) for k2, g in
                                            p[p.dataset == "gsm8k"].groupby("cell")},
                  "gsm8k_d_acc_pp": {k2: 100 * float(gsm.loc[k2, "d_acc"]) for k2 in gsm.index},
                  "diagonal_counts": diag,
                  "improving_changes": int(len(improving)),
                  "improving_median_down_pct": 100 * float(improving.down.median()),
                  "improving_median_loss_over_gain": float((improving.loss / improving.gain).median()),
                  "near_zero_dacc_changes": int(len(near0)),
                  "near_zero_median_moved_pct": 100 * float((near0.up + near0.down).median()),
                  "near_zero_max_moved_pct": 100 * float((near0.up + near0.down).max())}
    # rerun baseline
    nj = json.loads(first_existing(PP.out_dir(None, "rerun_null") / "null_gsm8k.json",
                                   PP.REFERENCE / "rerun_null/null_gsm8k.json").read_text())["null"]
    keep = {m: v for m, v in nj.items() if m not in EXCLUDE_FROM_RERUN_BASELINE}
    rng = lambda key: [100 * min(v[key] for v in keep.values()), 100 * max(v[key] for v in keep.values())]  # noqa: E731
    out["rerun_baseline_gsm8k"] = {"models": sorted(keep), "formulation_flip_pct": rng("formulation_flip_rate"),
                                   "moved_pct": rng("moved"), "down_pct": rng("down"), "per_model": nj}
    # fixed budget
    b = pd.read_csv(E.OUT / "summary.csv")
    out["appendix_budget_K2"] = b[b.K == 2].set_index("dataset")[["n", "acc", "mean_A", "latent", "reliable"]].to_dict("index")


def to_md(out):
    lines = ["# Numbers quoted in the paper (generated by paper_numbers.py; do not edit)", ""]

    def walk(prefix, x):
        if isinstance(x, dict):
            for k2, v in x.items():
                walk(f"{prefix}.{k2}" if prefix else k2, v)
        elif isinstance(x, list) and len(x) <= 4 and all(isinstance(v, (int, float)) for v in x):
            lines.append(f"| `{prefix}` | {', '.join(f'{v:.3g}' if isinstance(v, float) else str(v) for v in x)} |")
        elif isinstance(x, (int, float)):
            lines.append(f"| `{prefix}` | {x:.4g} |" if isinstance(x, float) else f"| `{prefix}` | {x} |")
    lines += ["| key | value |", "|---|---|"]
    walk("", {k2: v for k2, v in out.items() if k2 not in ("rq1_fig3_stats", "rq1_access_reliable_tradeoff", "_sources")})
    return "\n".join(lines) + "\n"


def main():
    store = pickle.load(open(PP.rq1_store(), "rb"))
    out = {}
    rq1(store, out)
    from_script_outputs(out)
    dest = PP.out_dir(None, "paper_numbers")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "paper_numbers.json").write_text(json.dumps(out, indent=1, default=float))
    (dest / "paper_numbers.md").write_text(to_md(out))
    print("wrote", dest)


if __name__ == "__main__":
    main()
