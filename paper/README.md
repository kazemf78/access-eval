# Reproducing the paper

*Revealing what benchmark scores hide: Jointly measuring the breadth and reliability of LLM capabilities using
accessibility profiles.*

## One command

```bash
pip install -e . && pip install -r paper/requirements.txt     # Python 3.11; pdftoppm (poppler-utils) for the figure check
bash paper/reproduce.sh                                       # -> paper/reproduction/<timestamp>/
```

It rebuilds every figure, table and quoted number from `paper/data/` (10-25 minutes on one CPU core, no GPU, no API
calls), collects the figures and tables under the file names the LaTeX source uses (`<OUT>/paper_assets/`), and runs
`verify_reproduction.py`, which compares every data file value by value and every figure pixel by pixel with the
versions in the paper (`paper/reference/`) and writes `<OUT>/verify_report.md`.

## What produces what

| Paper element | Script | Output under `<OUT>/` |
|---|---|---|
| RQ1 per-question store (every RQ1 / RQ3-scaling analysis) | `rq1_selection_pass.py` | `rq1/rq1_per_question.pkl`, per-model metrics |
| Fig. 1 capability envelope; appendix envelopes, near-tied pairs, majority reversals; Table (per-benchmark summary) | `rq1_render.py` | `rq1/figs/v6/` |
| Fig. 3 reliability curves | `rq1_render.py --only fig3_main_inside` | `rq1/figs/v6/fig3_reliability_curves_main_legend_inside.pdf` |
| Near-tied statistics, Access@k vs Reliable@k disagreement | `rq1_fig3_stats.py`, `access_reliable_tradeoff.py` | `rq1/figs/v6/{fig3_stats,access_reliable_tradeoff}.json` |
| Appendix: fixed-budget estimation | `equal_cost_budget.py` | `equal_cost_budget/summary.csv` |
| RQ2 figures (decoding vs formulation) | `rq2_decoding_vs_formulation.py` | `rq2/figs/v4/` |
| RQ3 figures (interventions), paired profiles | `rq3_interventions.py`, `rq3_paired_profiles.py` | `rq3/figs/v7/`, `rq3_paired/v1/` |
| GSM8K run-to-run baseline; formulation flip rates of model changes | `rerun_null/analyze.py`, `rerun_null/formulation_flip_model_changes.py` | `rerun_null/` |
| Every number in the text | `paper_numbers.py` | `paper_numbers/paper_numbers.{json,md}` |

`paper_numbers.md` lists every quoted number with its key.

## Data (`paper/data/`)

| Folder | Content |
|---|---|
| `graded/<benchmark>.parquet` | one row per (model, inquiry, formulation): `model, idx, original_question, paraphrased_question` (equal to `original_question` for the original wording), `verdict`, and the judge-agreement columns `n_judges, n_agree, gpt_agree`. SimpleQA: 4 LLM judges; TruthfulQA: majority of 3 GPT judges; GSM8K / MATH-500: `math_verify` + fallback extractor; MMLU: letter extraction; BFCL: harness scoring. |
| `rq2/` | RQ2 labels: 4 models x 3 benchmarks x 8 formulations x 5 samples at T = 0.7, greedy answers, greedy repeats |
| `rq3/` | prompted-CoT and prompt-optimization tables (same format as `graded/`), and `rq3_pairs_cache.pkl` with every before/after pair. `REBUILD_RQ3_PAIRS=1` rebuilds the pairs from the tables. |
| `rerun_null/` | GSM8K greedy rerun, graded with the paper's math policy |

The full model answers, judge outputs and paraphrase-equivalence verdicts behind these tables are in
[access-eval-data](https://github.com/kazemf78/access-eval-data).

## Generating new data

Generation and grading need GPUs and API calls; the toolkit (`access-eval`, see the main README) runs the same
pipeline: formulations (paraphrase generation + equivalence judges + NLI filter), answers, grading, profiles. The
paper's validated formulation sets are shipped in `src/access_eval/data/validated/`.
