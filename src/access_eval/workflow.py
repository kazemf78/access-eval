"""Profile a model end to end: formulations -> answers -> grades -> accessibility profile (-> budgeted estimation).

Every stage writes into one WORKSPACE directory (the pipeline's datasets/ + run_artifacts/ layout, selected with
$ACCESS_EVAL_ROOT), so a formulation set is built once per dataset and shared by every model you profile, and a
rerun only computes what is missing (answers and judge verdicts are cached by content). Each stage also writes a
small JSON record under <workspace>/results/<dataset>/<model>/ that the next stage reads.

    from access_eval import workflow as wf
    wf.profile_model("gsm8k_test", "Qwen/Qwen3-8B", workspace="./ws")      # all stages, paper defaults

or one stage at a time with the `access-eval` command (see `access-eval --help`).
"""
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd

VALIDATED = Path(__file__).resolve().parent / "data" / "validated"
DATASETS = ("SimpleQA", "TruthfulQA", "gsm8k_test", "math_500_test", "mmlu")

# Paper settings (defaults; every one can be overridden).
DEFAULT_MAX_TOKENS = {"SimpleQA": 256, "TruthfulQA": 512, "gsm8k_test": 16384, "math_500_test": 32768, "mmlu": 1024}
DEFAULT_JUDGES = {"SimpleQA": ["google/gemma-4-12B-it"], "TruthfulQA": ["google/gemma-4-12B-it"],
                  "gsm8k_test": ["rule"], "math_500_test": ["rule"], "mmlu": ["rule"]}


# ---------------------------------------------------------------- workspace helpers

def use_workspace(workspace):
    """Point the pipeline at the workspace and make sure the benchmark source files are in it."""
    ws = Path(workspace).resolve()
    (ws / "datasets").mkdir(parents=True, exist_ok=True)
    for f in (VALIDATED / "datasets").glob("*.csv"):
        dst = ws / "datasets" / f.name
        if not dst.exists():
            shutil.copyfile(f, dst)
    os.environ["ACCESS_EVAL_ROOT"] = str(ws)
    return ws


def _safe(name):
    return str(name).replace("/", "_").replace(":", "_")


def record_dir(workspace, dataset, model=None):
    d = Path(workspace).resolve() / "results" / dataset
    return d / _safe(model) if model else d


@contextlib.contextmanager
def _dataset_lock(ws, dataset):
    """Serialize writers of one dataset's shared files (combined answers / per-judge grade files)."""
    (ws / ".locks").mkdir(exist_ok=True)
    with open(ws / ".locks" / f"{dataset}.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1, default=str))
    return payload


def _read(path):
    return json.loads(Path(path).read_text())


def _env(engine_kwargs=None, endpoint_base_url=None):
    if engine_kwargs:
        os.environ["ACCESS_EVAL_ENGINE_KWARGS"] = engine_kwargs if isinstance(engine_kwargs, str) else json.dumps(engine_kwargs)
    if endpoint_base_url:
        os.environ["ACCESS_EVAL_ENDPOINT_BASE_URL"] = str(endpoint_base_url)


# ---------------------------------------------------------------- stage 1: formulations

def formulations(dataset, workspace, source="validated", *, k=8, seed_with_validated=False, inquiries=None,
                 generator="Qwen/Qwen3.5-9B", judges=("google/gemma-4-12B-it",), judge_prompt=None, nli=True,
                 nli_threshold=0.75, overlap_max=0.8, numbers_check=False, max_rounds=3, oversample=2.0, seed=0):
    """Provide the formulation set of one dataset in the workspace (shared by every model).

    source = "validated"  the paper's validated sets (shipped with the package)
           = <path.csv>   any pool with columns idx, original_question, paraphrased_question
           = "generate"   generate + validate paraphrases: generator LLM -> lexical-diversity filter -> judge chain
                          (all must accept) -> NLI filter, until each inquiry has k paraphrases. With
                          seed_with_validated=True the paper's sets are only topped up to k.
    inquiries: None (all), an int (first N), or a list of inquiry ids (generation only).
    The set is identified by a hash of its source; its run directory is run_artifacts/<dataset>_ae_<hash>, so
    different sets never mix and an identical set is reused from cache.
    """
    ws = use_workspace(workspace)
    from access_eval.core.paths import ProjectPaths
    info = {"dataset": dataset, "source": str(source), "workspace": str(ws)}
    if source == "generate":
        from access_eval.pipeline.formulations import FormulationConfig, build_formulations, install_seed_pool
        seed_file = VALIDATED / "formulations" / f"{dataset}.csv" if seed_with_validated else None
        if isinstance(inquiries, int):
            inquiries = _first_inquiries(dataset, inquiries)
        if inquiries is None and seed_file is not None:
            inquiries = sorted(pd.read_csv(seed_file)["idx"].unique().tolist())
        gen = {"k": int(k), "seed_sha": _sha256_file(seed_file) if seed_file else None, "inquiries": inquiries,
               "generator": generator, "judges": list(judges), "judge_prompt": judge_prompt, "nli": bool(nli),
               "nli_threshold": float(nli_threshold), "overlap_max": float(overlap_max),
               "numbers_check": bool(numbers_check), "max_rounds": int(max_rounds), "oversample": float(oversample),
               "rng_seed": int(seed)}
        flag = "ae_g" + hashlib.sha256(json.dumps(gen, sort_keys=True).encode()).hexdigest()[:11]
        cfg = FormulationConfig(dataset=dataset, experiment_flag=flag, target_k=gen["k"], generator=generator,
                                judges=gen["judges"], judge_prompt=judge_prompt, nli=gen["nli"],
                                nli_threshold=gen["nli_threshold"], overlap_max=gen["overlap_max"],
                                numbers_check=gen["numbers_check"], max_rounds=gen["max_rounds"],
                                oversample=gen["oversample"], inquiries=inquiries, seed=gen["rng_seed"])
        if seed_file is not None:
            install_seed_pool(cfg, seed_file)
        info.update({"generation": {k_: v for k_, v in gen.items() if k_ != "inquiries"},
                     "generation_summary": build_formulations(cfg)})
        source_id = flag
    else:
        src = VALIDATED / "formulations" / f"{dataset}.csv" if source == "validated" else Path(source)
        pool = pd.read_csv(src)
        missing = {"idx", "original_question", "paraphrased_question"} - set(pool.columns)
        if missing:
            raise ValueError(f"{src}: missing columns {missing}")
        source_id = _sha256_file(src)
        flag = f"ae_{source_id[:12]}"
        rp = ProjectPaths().run_paths(dataset, flag)
        rp.ensure_dirs()
        if not rp.paraphrases_file().exists():
            pool.to_csv(rp.paraphrases_file(), index=False)
    rp = ProjectPaths().run_paths(dataset, flag)
    pool = pd.read_csv(rp.paraphrases_file())
    per_q = pool.groupby("idx").size()
    return _write(record_dir(ws, dataset) / "formulations.json", {
        **info, "experiment_flag": flag, "source_id": source_id, "pool_file": str(rp.paraphrases_file()),
        "n_inquiries": int(len(per_q)), "n_paraphrases": int(len(pool)), "min_paraphrases": int(per_q.min()),
        "cost_full_calls": int(len(pool) + len(per_q))})


def _first_inquiries(dataset, n):
    from access_eval.core.paths import ProjectPaths
    from access_eval.datasets.registry import get_dataset_spec
    ds = get_dataset_spec(dataset)
    data = pd.read_csv(ProjectPaths().dataset_file(ds["path"]))
    ok = [int(i) for i, q in data[ds["question_col"]].items() if len(str(q).split()) >= ds.get("min_length", 1)]
    return ok[:int(n)]


# ---------------------------------------------------------------- stage 2: answers

def answer(dataset, model, workspace, *, max_tokens=None, prompt=None, temperature=0.0, engine_kwargs=None,
           endpoint_base_url=None, gpu_memory_utilization=0.85, tensor_parallel_size=None, api_conc=32,
           batch_size=1024):
    """Answer every formulation (paraphrases + originals) with one model; cached by content, so a rerun only
    generates what is missing. model: a Hugging Face id (in-process vLLM), an OpenAI model name, or
    endpoint:<served name> (any OpenAI-compatible server at endpoint_base_url)."""
    ws = use_workspace(workspace)
    f = _read(record_dir(ws, dataset) / "formulations.json")
    _env(engine_kwargs, endpoint_base_url)
    from access_eval.models.factory import is_api_model
    from access_eval.pipeline.answer_generation import answer_config_hash, resolve_answer_prompt, run_answer_generation
    flag = f["experiment_flag"]
    max_tokens = DEFAULT_MAX_TOKENS[dataset] if max_tokens is None else int(max_tokens)
    api = is_api_model(model)
    with _dataset_lock(ws, dataset):
        run_answer_generation(
            dataset=dataset, experiment_flag=flag, temperature=float(temperature), question_subset="both",
            api_models=[model] if api else [], local_models=[] if api else [model],
            max_local_tokens=max_tokens, max_api_tokens=max_tokens, answer_prompt_name=prompt,
            batch_size=batch_size, api_conc=int(api_conc), save_every_n_chunks=1,
            gpu_memory_utilization=float(gpu_memory_utilization), tensor_parallel_size=tensor_parallel_size)
    name, template, _ = resolve_answer_prompt(dataset, prompt)
    return _write(record_dir(ws, dataset, model) / "answers.json", {
        **f, "model": model, "max_tokens": max_tokens, "answer_prompt": name, "temperature": float(temperature),
        "gen_config_hash": answer_config_hash(model, float(temperature), max_tokens, name, template)})


# ---------------------------------------------------------------- stage 3: grades

def grade(dataset, model, workspace, *, judges=None, evaluator=None, engine_kwargs=None, endpoint_base_url=None,
          batch_size=1024):
    """Grade one model's answers (judge-verdict cache: identical answer texts are judged once). judges = None
    ("rule" for GSM8K/MATH-500/MMLU, google/gemma-4-12B-it for SimpleQA/TruthfulQA) or a list; with several judges
    the profile takes their majority. evaluator = None (the dataset's default) or a registered evaluator name."""
    ws = use_workspace(workspace)
    a = _read(record_dir(ws, dataset, model) / "answers.json")
    _env(engine_kwargs, endpoint_base_url)
    from access_eval.evaluators import get_evaluator
    from access_eval.pipeline.evaluation import default_evaluator_kwargs, evaluator_config_hash, run_evaluation
    judges = list(judges) if judges else DEFAULT_JUDGES[dataset]
    kw = default_evaluator_kwargs(dataset)
    with _dataset_lock(ws, dataset):
        run_evaluation(dataset=dataset, experiment_flag=a["experiment_flag"], temperature=a["temperature"],
                       question_subset="both", judge_models=judges, batch_size=batch_size, save_every_n_chunks=1,
                       evaluator_kwargs=kw, evaluator_name=evaluator)
    hashes = {j: evaluator_config_hash(get_evaluator(dataset, evaluator_name=evaluator, judge_model=j,
                                                     temperature=a["temperature"], **kw),
                                       j, a["temperature"], evaluator, kw) for j in judges}
    return _write(record_dir(ws, dataset, model) / "grades.json", {
        **a, "judges": ",".join(judges), "evaluator": evaluator or "default",
        "label_rule": "single" if len(judges) == 1 else "majority", "_eval_config_hashes": hashes})


# ---------------------------------------------------------------- stage 4: profile

def graded_formulations(g):
    """One row per formulation of this run (model + gen_config_hash, eval_config_hash per judge): binary correctness
    (not_attempted counts as incorrect); several judges -> majority (a tie is incorrect); answers whose grading
    timed out are NaN (excluded)."""
    from access_eval.core.paths import ProjectPaths
    rp = ProjectPaths().run_paths(g["dataset"], g["experiment_flag"])
    votes = []
    for judge, eh in g["_eval_config_hashes"].items():
        d = pd.read_csv(rp.grades_file(subset="both", judge_model=judge))
        d = d[(d["model"] == g["model"]) & (d["gen_config_hash"] == g["gen_config_hash"]) & (d["eval_config_hash"] == eh)]
        if d.empty:
            raise RuntimeError(f"no grades for {g['model']} / {judge} / {eh} in {rp.grades_file('both', judge)}")
        timeout = d["grading_response"].astype(str).eq("GRADING_TIMEOUT") if "grading_response" in d else False
        d = d.assign(ok=(d["verdict"] == "correct").astype(float))
        d.loc[timeout, "ok"] = float("nan")
        votes.append(d[["idx", "original_question", "paraphrased_question", "ok"]]
                     .drop_duplicates(["idx", "paraphrased_question"]).assign(judge=judge))
    agg = pd.concat(votes).groupby(["idx", "original_question", "paraphrased_question"], dropna=False)["ok"]
    agg = agg.agg(["sum", "count"]).reset_index()
    agg["correct"] = (2 * agg["sum"] > agg["count"]).astype(float)
    agg.loc[agg["count"] == 0, "correct"] = float("nan")
    agg["is_original"] = agg["paraphrased_question"] == agg["original_question"]
    return agg


def profile(dataset, model, workspace, *, min_para=5, bootstrap=2000):
    """The model's accessibility profile on this formulation set (paper definitions). m_q counts paraphrases only;
    inquiries need >= min_para graded paraphrases. Also writes per_inquiry.csv (y, c, m per inquiry)."""
    ws = use_workspace(workspace)
    out = record_dir(ws, dataset, model)
    g = _read(out / "grades.json")
    from access_eval.accessibility.metrics import bootstrap_ci, per_inquiry, profile_metrics
    rows = graded_formulations(g)
    t = per_inquiry(rows, min_para=int(min_para))
    t.to_csv(out / "per_inquiry.csv", index=False)
    prof = profile_metrics(t["y"], t["c"], t["m"])
    prof.update(bootstrap_ci(t["y"], t["c"], t["m"], B=int(bootstrap)))
    prof["n_inquiries_dropped_min_para"] = int(rows.loc[~rows.is_original, "idx"].nunique() - len(t))
    prof["n_grading_excluded"] = int(rows["correct"].isna().sum())
    return _write(out / "profile.json", {**{k: v for k, v in g.items() if not k.startswith("_")},
                                         "min_para": int(min_para), **prof})


# ---------------------------------------------------------------- stage 5 (optional): budgeted estimation

def estimate(dataset, model, workspace, *, betas=(0.25, 0.5, 1, 2, 4), ks=(2, 3), trials=2000, tol=0.05,
             target=0.95, seed=0):
    """How well a budget-limited evaluation recovers this full profile. Costs are actual call counts:
    conventional = N, full = sum(m_q + 1), budget = beta * N. Designs: conventional, subsample, kform_K<k>."""
    ws = use_workspace(workspace)
    out = record_dir(ws, dataset, model)
    from access_eval.accessibility.estimation import budget_study, flatten
    prof = _read(out / "profile.json")
    t = pd.read_csv(out / "per_inquiry.csv")
    study = budget_study(t["y"], t["c"], t["m"], betas=[float(b) for b in betas], ks=[int(k) for k in ks],
                         trials=int(trials), tol=float(tol), target=float(target), seed=int(seed))
    (out / "estimate_full.json").write_text(json.dumps(study, indent=1))
    return _write(out / "estimate.json", {**prof, **flatten(study)})


# ---------------------------------------------------------------- everything

def profile_model(dataset, model, workspace, *, source="validated", answer_kwargs=None, grade_kwargs=None,
                  min_para=5, estimate_budget=False):
    """formulations -> answer -> grade -> profile (-> estimate) with paper defaults. Returns the profile dict."""
    formulations(dataset, workspace, source=source)          # cheap when the set is already in the workspace
    answer(dataset, model, workspace, **(answer_kwargs or {}))
    grade(dataset, model, workspace, **(grade_kwargs or {}))
    prof = profile(dataset, model, workspace, min_para=min_para)
    if estimate_budget:
        return estimate(dataset, model, workspace)
    return prof
