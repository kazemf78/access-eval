# src/access_eval/datasets/registry.py

import json
from pathlib import Path

from access_eval.core.paths import project_root_dir


def _sidecar() -> Path:
    """datasets/user_registry.json under the current project root (resolved per call, not at import, so
    $ACCESS_EVAL_ROOT set after import is honoured)."""
    return project_root_dir() / "datasets" / "user_registry.json"

DATASET_REGISTRY = {
    "SimpleQA": {
        "path": "simple_qa_test_set.csv",
        "question_col": "problem",
        "min_length": 10,
        "task_type": "factual_qa",
    },
    "TruthfulQA": {
        "path": "TruthfulQA.csv",
        "question_col": "Question",
        "min_length": 8,
        "task_type": "factual_qa",
    },
    "gsm8k_test": {
        "path": "openai_gsm8k_main_test.csv",
        "question_col": "question",
        "min_length": 10,
        "task_type": "math",
    },
    "math_500_test": {
        "path": "HuggingFaceH4_MATH_500_test.csv",
        "question_col": "problem",
        "min_length": 10,
        "task_type": "math",
    },
    # MMLU (ConStat test split, the paper's MMLU source): the stem is paraphrased, the options stay fixed per
    # question, and the prompt is the 5-shot harness prompt (prompts/builders.py); graded by letter match.
    "mmlu": {
        "path": "ConStat_mmlu_test_combined.csv",
        "question_col": "question",
        "answer_col": "answer_text",
        "min_length": 1,
        "task_type": "mcq",
        "prompt_builder": "mmlu_5shot",
    },
    "aime_2024_train": {
        "path": "HuggingFaceH4_aime_2024_train.csv",
        "question_col": "problem",
        "min_length": 10,
        "task_type": "math",
    },
}


def _safe(name: str) -> str:
    """Filesystem-safe version of a dataset name (same transform as hf_loader)."""
    return name.replace("/", "_").replace("-", "_")


def _merged() -> dict:
    sidecar = _sidecar()
    if sidecar.exists():
        with open(sidecar) as f:
            return {**DATASET_REGISTRY, **json.load(f)}
    return dict(DATASET_REGISTRY)


def _lookup(name: str, registry: dict) -> dict | None:
    """Try exact name first, then the filesystem-safe variant."""
    if name in registry:
        return registry[name]
    safe = _safe(name)
    if safe != name and safe in registry:
        return registry[safe]
    return None


def get_dataset_spec(name: str) -> dict:
    spec = _lookup(name, _merged())
    if spec is None:
        raise KeyError(
            f"Unknown dataset: {name!r}. "
            f"Pass --hf-name, --hf-question-col, and --hf-task-type to the pipeline "
            f"to auto-download from HuggingFace on first use."
        )
    return spec


def resolve_dataset(
    name: str,
    *,
    hf_name: str | None = None,
    question_col: str | None = None,
    task_type: str | None = None,
    hf_split: str = "test",
    hf_config: str | None = None,
    answer_col: str | None = None,
    min_length: int | None = None,
) -> dict:
    """
    Return the dataset spec, auto-downloading from HuggingFace if not registered.

    Accepts both raw HF paths (tatsu-lab/alpaca) and safe names (tatsu-lab_alpaca).
    On first use with an unknown dataset, pass hf_name / question_col / task_type.
    Subsequent runs find the dataset in user_registry.json and skip the download.
    """
    spec = _lookup(name, _merged())
    if spec is not None:
        return spec

    if not hf_name or not question_col or not task_type:
        raise KeyError(
            f"Dataset {name!r} is not registered. "
            f"Pass --hf-name, --hf-question-col, and --hf-task-type to auto-download "
            f"from HuggingFace, or run load_and_save_hf_dataset.py manually."
        )

    from access_eval.datasets.hf_loader import download_and_register
    registry_name = _safe(name)
    extra = {}
    if min_length is not None:
        extra["min_length"] = min_length
    download_and_register(
        hf_name=hf_name,
        registry_name=registry_name,
        question_col=question_col,
        task_type=task_type,
        split=hf_split,
        hf_config=hf_config,
        answer_col=answer_col,
        **extra,
    )
    return _merged()[registry_name]

# todo: Add answer_col in future if needed!
