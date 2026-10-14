# src/access_eval/datasets/hf_loader.py

import json
from pathlib import Path

from access_eval.core.paths import project_root_dir

# Tried in order; first match wins.
_ANSWER_COL_CANDIDATES = [
    "answer",
    "answers",
    "possible_answers",
    "gold_answer",
    "answer_text",
    "target",
    "solution",
    "correct_answer",
    "output",
    "label",
]


def _detect_answer_col(column_names: list[str]) -> str | None:
    """Return the first candidate column found, or None with a warning."""
    for candidate in _ANSWER_COL_CANDIDATES:
        if candidate in column_names:
            return candidate
    return None


def download_and_register(
    hf_name: str,
    registry_name: str,
    question_col: str,
    task_type: str,
    split: str = "test",
    hf_config: str | None = None,
    min_length: int = 5,
    answer_col: str | None = None,
) -> Path:
    """
    Download one split of an HF dataset, save it as a CSV in datasets/,
    and register it in datasets/user_registry.json.

    answer_col is auto-detected from the column list if not provided.
    Pass it explicitly in your YAML config only when auto-detection picks
    the wrong column.

    Returns the path to the saved CSV.
    """
    from datasets import load_dataset

    if task_type not in ("math", "factual_qa"):
        raise ValueError(f"task_type must be 'math' or 'factual_qa', got {task_type!r}")

    print(f"[hf_loader] Downloading '{hf_name}' (config={hf_config!r}, split={split!r}) ...")
    load_kwargs = {"name": hf_config} if hf_config else {}
    dataset = load_dataset(hf_name, split=split, **load_kwargs)

    if question_col not in dataset.column_names:
        raise ValueError(
            f"Column {question_col!r} not found in dataset. "
            f"Available columns: {dataset.column_names}"
        )

    # Resolve answer_col: explicit override > auto-detect
    if answer_col is not None:
        if answer_col not in dataset.column_names:
            raise ValueError(
                f"Specified answer_col={answer_col!r} not found. "
                f"Available columns: {dataset.column_names}"
            )
        print(f"[hf_loader] answer_col='{answer_col}' (from config)")
    else:
        answer_col = _detect_answer_col(dataset.column_names)
        if answer_col:
            print(f"[hf_loader] answer_col='{answer_col}' (auto-detected)")
        else:
            print(
                f"[hf_loader] Warning: could not auto-detect answer_col "
                f"from columns: {dataset.column_names}\n"
                f"  Add 'answer_col' to your YAML config to enable evaluation."
            )

    safe_hf = hf_name.replace("/", "_").replace("-", "_")
    if hf_config:
        safe_hf += f"_{hf_config}"
    csv_filename = f"{safe_hf}_{split}.csv"

    datasets_dir = project_root_dir() / "datasets"
    sidecar = datasets_dir / "user_registry.json"
    datasets_dir.mkdir(exist_ok=True)
    csv_path = datasets_dir / csv_filename
    print(f"[hf_loader] Saving {len(dataset):,} rows → {csv_path}")
    dataset.to_csv(str(csv_path))

    user_registry: dict = {}
    if sidecar.exists():
        with open(sidecar) as f:
            user_registry = json.load(f)

    entry = {
        "path": csv_filename,
        "question_col": question_col,
        "min_length": min_length,
        "task_type": task_type,
    }
    if answer_col is not None:
        entry["answer_col"] = answer_col
    user_registry[registry_name] = entry
    with open(sidecar, "w") as f:
        json.dump(user_registry, f, indent=2)
    print(f"[hf_loader] Registered {registry_name!r} in {sidecar.name}")

    return csv_path
