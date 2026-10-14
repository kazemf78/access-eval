"""`access-eval` command: profile a model, one stage at a time or all at once.

    access-eval run          --dataset gsm8k_test --model Qwen/Qwen3-8B --workspace ./ws
    access-eval formulations --dataset gsm8k_test --workspace ./ws [--source validated|generate|pool.csv]
    access-eval answer       --dataset gsm8k_test --model Qwen/Qwen3-8B --workspace ./ws
    access-eval grade        --dataset gsm8k_test --model Qwen/Qwen3-8B --workspace ./ws
    access-eval profile      --dataset gsm8k_test --model Qwen/Qwen3-8B --workspace ./ws
    access-eval estimate     --dataset gsm8k_test --model Qwen/Qwen3-8B --workspace ./ws
"""
import argparse
import json

from access_eval import workflow as wf


def _csv(s, cast=str):
    return [cast(x.strip()) for x in str(s).split(",") if x.strip()] if s else None


def _print(rec, keys=None):
    rec = {k: v for k, v in rec.items() if not k.startswith("_")}
    if keys:
        rec = {k: rec[k] for k in keys if k in rec}
    print(json.dumps(rec, indent=1, default=str))


PROFILE_KEYS = ["dataset", "model", "n_inquiries", "acc", "mean_A", "R", "L", "B", "reversal", "access_at_2",
                "reliable_at_2", "access_at_5", "reliable_at_5"]


def main(argv=None):
    p = argparse.ArgumentParser(prog="access-eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, model=True):
        sp.add_argument("--dataset", required=True, choices=wf.DATASETS)
        if model:
            sp.add_argument("--model", required=True,
                            help="Hugging Face id (local vLLM), OpenAI model name, or endpoint:<served name>")
        sp.add_argument("--workspace", default="./access_eval_workspace")

    def answer_opts(sp):
        sp.add_argument("--max-tokens", type=int, help="default: the paper's budget per dataset")
        sp.add_argument("--prompt", help="answer prompt name under prompts/answer/ (default: the dataset's)")
        sp.add_argument("--temperature", type=float, default=0.0)
        sp.add_argument("--engine-kwargs", help='JSON vLLM engine overrides, e.g. {"kv_cache_dtype": "fp8"}')
        sp.add_argument("--endpoint-base-url", help="for endpoint:<model> (any OpenAI-compatible server)")
        sp.add_argument("--gpu-memory-utilization", type=float, default=0.85)
        sp.add_argument("--tensor-parallel-size", type=int)

    def grade_opts(sp):
        sp.add_argument("--judges", help="comma list (default: rule for math/MMLU, google/gemma-4-12B-it for QA)")
        sp.add_argument("--evaluator", help="registered evaluator name (default: the dataset's)")

    sp = sub.add_parser("formulations", help="install or generate the dataset's formulation set")
    common(sp, model=False)
    sp.add_argument("--source", default="validated", help="validated | generate | path to a pool CSV")
    sp.add_argument("--k", type=int, default=8, help="(generate) paraphrases per inquiry")
    sp.add_argument("--seed-with-validated", action="store_true", help="(generate) top up the paper's sets to k")
    sp.add_argument("--inquiries", type=int, help="(generate) only the first N inquiries")
    sp.add_argument("--generator", default="Qwen/Qwen3.5-9B")
    sp.add_argument("--gen-judges", default="google/gemma-4-12B-it", help="comma list; all must accept")
    sp.add_argument("--no-nli", action="store_true")

    sp = sub.add_parser("answer", help="answer every formulation with one model")
    common(sp)
    answer_opts(sp)

    sp = sub.add_parser("grade", help="grade the model's answers")
    common(sp)
    grade_opts(sp)
    sp.add_argument("--engine-kwargs")
    sp.add_argument("--endpoint-base-url")

    sp = sub.add_parser("profile", help="accessibility profile from the grades")
    common(sp)
    sp.add_argument("--min-para", type=int, default=5)
    sp.add_argument("--bootstrap", type=int, default=2000)

    sp = sub.add_parser("estimate", help="budget-limited estimation of the profile")
    common(sp)
    sp.add_argument("--betas", default="0.25,0.5,1,2,4", help="budgets as multiples of the conventional cost N")
    sp.add_argument("--ks", default="2,3", help="formulations per inquiry for the K-formulation design")
    sp.add_argument("--trials", type=int, default=2000)

    sp = sub.add_parser("run", help="formulations -> answer -> grade -> profile with paper defaults")
    common(sp)
    sp.add_argument("--source", default="validated")
    answer_opts(sp)
    grade_opts(sp)
    sp.add_argument("--min-para", type=int, default=5)
    sp.add_argument("--estimate", action="store_true", help="also run the budgeted-estimation study")

    a = p.parse_args(argv)
    answer_kw = lambda: dict(max_tokens=a.max_tokens, prompt=a.prompt, temperature=a.temperature,  # noqa: E731
                             engine_kwargs=a.engine_kwargs, endpoint_base_url=a.endpoint_base_url,
                             gpu_memory_utilization=a.gpu_memory_utilization,
                             tensor_parallel_size=a.tensor_parallel_size)
    if a.cmd == "formulations":
        _print(wf.formulations(a.dataset, a.workspace, a.source, k=a.k, seed_with_validated=a.seed_with_validated,
                               inquiries=a.inquiries, generator=a.generator, judges=_csv(a.gen_judges),
                               nli=not a.no_nli))
    elif a.cmd == "answer":
        _print(wf.answer(a.dataset, a.model, a.workspace, **answer_kw()))
    elif a.cmd == "grade":
        _print(wf.grade(a.dataset, a.model, a.workspace, judges=_csv(a.judges), evaluator=a.evaluator,
                        engine_kwargs=a.engine_kwargs, endpoint_base_url=a.endpoint_base_url))
    elif a.cmd == "profile":
        _print(wf.profile(a.dataset, a.model, a.workspace, min_para=a.min_para, bootstrap=a.bootstrap), PROFILE_KEYS)
    elif a.cmd == "estimate":
        _print(wf.estimate(a.dataset, a.model, a.workspace, betas=_csv(a.betas, float), ks=_csv(a.ks, int),
                           trials=a.trials))
    elif a.cmd == "run":
        rec = wf.profile_model(a.dataset, a.model, a.workspace, source=a.source, answer_kwargs=answer_kw(),
                               grade_kwargs=dict(judges=_csv(a.judges), evaluator=a.evaluator),
                               min_para=a.min_para, estimate_budget=a.estimate)
        _print(rec, None if a.estimate else PROFILE_KEYS)


if __name__ == "__main__":
    main()
