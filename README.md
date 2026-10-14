# access-eval

**Accessibility profiles: what benchmark scores hide.** Code and data for

> *Revealing what benchmark scores hide: Jointly measuring the breadth and reliability of LLM capabilities using
> accessibility profiles.* arXiv preprint, 2026.

Benchmarks score each question on one fixed wording. `access-eval` asks every *inquiry* in several
meaning-preserving *formulations* and measures its **accessibility**: the probability that the model succeeds
across formulations. The distribution of accessibility over a benchmark (the model's **accessibility profile**)
separates two things a single accuracy number merges:

- **breadth**: on how many inquiries can the model succeed at all? (latent share `L`, `Access@k`)
- **reliability**: on how many does it succeed every time? (reliable share `R`, `Reliable@k`)

Two models with the same accuracy can differ in either direction, and which one is better depends on whether one
success in `k` tries is enough (`Access@k`) or all `k` must succeed (`Reliable@k`).

This repository has two parts:

| | What | Where |
|---|---|---|
| **Toolkit** | profile any model (local vLLM, OpenAI, or any OpenAI-compatible server) on GSM8K, MATH-500, MMLU, SimpleQA, TruthfulQA, or your own data | `src/access_eval/` |
| **Paper reproduction** | rebuild every figure, table and number of the paper and check it against the published versions | `paper/` |

The full per-formulation model answers and judge grades used in the paper are in
[access-eval-data](https://github.com/kazemf78/access-eval-data).

## Install

```bash
git clone https://github.com/kazemf78/access-eval && cd access-eval
pip install -e .                 # toolkit with API / OpenAI-compatible endpoints
pip install -e ".[vllm]"         # + in-process local inference with vLLM (GPU)
pip install -e ".[paper,dev]"    # + paper reproduction and tests
```

## Profile a model

```bash
# GSM8K with the paper's validated formulations, a local model via vLLM
access-eval run --dataset gsm8k_test --model Qwen/Qwen3-8B --workspace ./ws

# an OpenAI model
export OPENAI_API_KEY=...
access-eval run --dataset math_500_test --model gpt-4.1-mini --workspace ./ws

# any OpenAI-compatible server (e.g. `vllm serve`)
access-eval run --dataset mmlu --model endpoint:my-model --endpoint-base-url http://localhost:8000/v1 --workspace ./ws
```

`run` executes four stages with the paper's defaults; each is also its own command:

| Stage | Command | What it does |
|---|---|---|
| 1 | `access-eval formulations` | installs the validated formulation set (default), a pool CSV of your own, or **generates** new paraphrases (`--source generate`: generator LLM, lexical-diversity filter, LLM-judge equivalence chain, NLI filter) |
| 2 | `access-eval answer` | answers every formulation (original wording + paraphrases) |
| 3 | `access-eval grade` | grades answers: rule-based for GSM8K / MATH-500 / MMLU, LLM judge(s) for SimpleQA / TruthfulQA |
| 4 | `access-eval profile` | the accessibility profile with bootstrap confidence intervals |
| 5 | `access-eval estimate` | optional: how well a cheaper, budget-limited evaluation recovers the profile |

Everything is cached in the workspace by content: formulations are built once per dataset and shared by every model,
and a rerun only generates or grades what is missing. Results are in `ws/results/<dataset>/<model>/`
(`profile.json`, `per_inquiry.csv`).

From Python:

```python
from access_eval import workflow as wf
prof = wf.profile_model("gsm8k_test", "Qwen/Qwen3-8B", workspace="./ws")
print(prof["acc"], prof["mean_A"], prof["R"], prof["L"], prof["access_at_5"], prof["reliable_at_5"])
```

### What the profile reports

Per inquiry `q`: `y_q` = correctness on the original wording, `m_q` = graded paraphrases, `c_q` = correct ones,
`A_q = c_q / m_q` (accessibility).

| Key | Definition | Meaning |
|---|---|---|
| `acc` | `P(y = 1)` | conventional single-wording accuracy |
| `mean_A` | `E[A]` | mean accessibility |
| `R` | `P(A = 1)` | reliable share: correct on every formulation |
| `L` | `P(A > 0)` | latent share: correct on at least one |
| `B` | `L - R` | brittle share |
| `access_at_k` | `E[1 - C(m-c,k)/C(m,k)]` | `k` random formulations, at least one correct |
| `reliable_at_k` | `E[C(c,k)/C(m,k)]` | `k` random formulations, all correct |
| `reversal` | `P(y=1, A<1/2) + P(y=0, A>1/2)` | the original wording contradicts the majority of formulations |

The metrics alone (no inference) are in `access_eval.accessibility.metrics`: give `per_inquiry()` one row per graded
formulation and `profile_metrics()` returns the profile.

### Your own data

Any pool CSV with columns `idx, original_question, paraphrased_question` works as a formulation set
(`--source my_pool.csv`); register a new dataset in `access_eval/datasets/registry.py` (question / answer columns,
answer prompt, evaluator).

## Reproduce the paper

See [`paper/README.md`](paper/README.md). One command rebuilds every figure, table and quoted number from the data in
this repository on one CPU core in 10-25 minutes, and verifies them value by value and pixel by pixel against the published
versions:

```bash
pip install -r paper/requirements.txt && pip install -e .
bash paper/reproduce.sh
```

## Data

| | Where | License |
|---|---|---|
| Validated formulation sets (5 benchmarks) | `src/access_eval/data/validated/` | CC-BY-4.0 |
| Per-formulation correctness of every paper model (inputs of all analyses) | `paper/data/` | CC-BY-4.0 |
| Full model answers + judge grades | [access-eval-data](https://github.com/kazemf78/access-eval-data) | CC-BY-4.0 |

The formulations and answers are derived from GSM8K (MIT), MATH (MIT), MMLU (MIT), SimpleQA (MIT), TruthfulQA
(Apache-2.0) and BFCL (Apache-2.0); please also cite the original benchmarks.

## Citation

```bibtex
@article{accessibility2026,
  title   = {Revealing what benchmark scores hide: Jointly measuring the breadth and reliability of {LLM}
             capabilities using accessibility profiles},
  author  = {TODO},
  journal = {arXiv preprint arXiv:TODO},
  year    = {2026}
}
```

## License

Code: [Apache-2.0](LICENSE). Data: CC-BY-4.0.
