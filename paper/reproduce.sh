#!/bin/bash
# Rebuild every figure, table and quoted number of the paper from paper/data/ into a fresh folder, then check the
# result against the versions used in the paper (paper/reference/): data value by value, figures pixel by pixel.
#
#   bash paper/reproduce.sh [OUT_DIR]          (default: paper/reproduction/<timestamp>)
#   FIG1_VARIANT=4panel bash paper/reproduce.sh   (Fig. 1 with an MMLU panel; default 3 panels = the paper)
#   PYTHON=/path/to/python bash paper/reproduce.sh
#
# Needs: pip install -e . && pip install -r paper/requirements.txt; pdftoppm (poppler) for the figure check.
# Runtime: 10-25 minutes on one CPU core.
#
# Stages (order = data dependencies):
#   1 rq1_selection_pass   RQ1 per-question store (common substrate per benchmark, per-model metrics)
#   2 rq1_render           Fig 1 (3 panels; 4 with FIG1_VARIANT=4panel), Fig 3, appendix figures, reversals, Table 1
#   3 rq1_fig3_stats, access_reliable_tradeoff, equal_cost_budget   RQ1 statistics + fixed-budget appendix
#   4 rq2_decoding_vs_formulation   RQ2 figures (main 2-panel, appendix 3-panel, k-curves)
#   5 rq3_interventions             RQ3 two-views (compact + appendix), dAcc-vs-dR, breadth/repeatability
#   6 rq3_paired_profiles           paired before/after profiles (main strip, per-family appendix, table)
#   7 rerun_null/analyze, rerun_null/formulation_flip_model_changes   GSM8K run-to-run baseline
#   8 paper_numbers                 every number quoted in the text -> paper_numbers.json / .md
#   9 collect the paper's figures/tables + verify_reproduction.py
cd "$(dirname "$0")/.." || exit 1
OUT=${1:-paper/reproduction/$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT" && OUT=$(cd "$OUT" && pwd)
export PAPER_REPRO_DIR="$OUT" PYTHONPATH="src:paper${PYTHONPATH:+:$PYTHONPATH}" MPLBACKEND=Agg
PY=${PYTHON:-python}
LOG="$OUT/reproduce.log"
run() { echo "=== $(date +%H:%M:%S) $*" | tee -a "$LOG"; "$PY" "$@" >>"$LOG" 2>&1 || { echo "FAILED: $* (see $LOG)"; exit 1; }; }

run paper/rq1_selection_pass.py
run paper/rq1_render.py                                        # full run: appendix envelope, near-tied pairs, ...
if [ "${FIG1_VARIANT:-3panel}" = "4panel" ]; then ONLY=fig1_main,fig2_reversals,table_main,fig3_main_inside
else ONLY=fig2_reversals,table_main,fig3_main_inside; fi
run paper/rq1_render.py --only "$ONLY"                         # paper versions of these
run paper/rq1_fig3_stats.py
run paper/access_reliable_tradeoff.py
run paper/equal_cost_budget.py
run paper/rq2_decoding_vs_formulation.py
run paper/rq3_interventions.py
run paper/rq3_paired_profiles.py
run paper/rerun_null/analyze.py
run paper/rerun_null/formulation_flip_model_changes.py
run paper/paper_numbers.py

# ---- collect the exact files the paper uses, under their LaTeX names
A="$OUT/paper_assets"; mkdir -p "$A/figs/rq1" "$A/figs/rq2" "$A/figs/rq3" "$A/tables"
ver() { sed -n 's/^VERSION = "\(v[0-9]*\)".*/\1/p' "$1"; }
R1="$OUT/rq1/figs/$(ver paper/rq1_render.py)"
R2="$OUT/rq2/figs/$(ver paper/rq2_decoding_vs_formulation.py)"
R3="$OUT/rq3/figs/$(ver paper/rq3_interventions.py)"
RP="$OUT/rq3_paired/$(ver paper/rq3_paired_profiles.py)"
for f in fig1_capability_envelope_main_absentred fig3_reliability_curves_main_legend_inside \
         fig1_capability_envelope_appendix_absentred fig2_majority_reversals fig2_majority_reversals_conditional \
         fig5_near_tied_pairs_appendix; do cp "$R1/$f.pdf" "$A/figs/rq1/"; done
cp "$R1/rq1_main_table.tex" "$A/tables/"
for f in fig_rq2_main_2panel fig_rq2_main fig_rq2_access_reliable_at_k; do cp "$R2/$f.pdf" "$A/figs/rq2/"; done
for f in fig_rq3_two_views_compact fig_rq3_two_views_appendix fig_rq3_dacc_vs_dR fig_rq3_breadth_vs_repeatability; do
  cp "$R3/$f.pdf" "$A/figs/rq3/"; done
for f in main scaling thinking cot popt; do cp "$RP/fig_rq3_paired_$f.pdf" "$A/figs/rq3/"; done
cp "$RP/rq3_paired_table.tex" "$A/tables/"
echo "=== paper assets: $A" | tee -a "$LOG"

run paper/verify_reproduction.py "$OUT"
echo "Done. Report: $OUT/verify_report.md"
