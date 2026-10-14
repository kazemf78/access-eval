# src/access_eval/io/artifacts.py

from pathlib import Path
import json
from datetime import datetime

def save_config(run_dir: Path, cfg: dict):
    path = run_dir / "config.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def save_pipeline_config(
    run_dir: Path,
    pipeline_name: str,
    cfg: dict,
    # overwrite: bool = False,
    overwrite: bool = True, # todo: Temporary and change later!
    keep_history: bool = True,
):
    """
    Save a pipeline-specific config file.

    The canonical `config.{pipeline_name}.json` is always overwritten (per the
    `overwrite=True` default above) so existing code that reads that fixed
    filename keeps working unchanged. When `keep_history` is True (default), a
    timestamped sidecar `config.{pipeline_name}.{timestamp}.json` is ALSO written
    alongside it, so incremental reruns with different parameters (e.g. adding a
    model, a new prompt variant) don't silently lose the record of what produced
    earlier artifacts still sitting in the same run directory.
    """
    path = run_dir / f"config.{pipeline_name}.json"

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. "
            "Use overwrite=True or version the config."
        )

    cfg = dict(cfg)
    cfg["pipeline"] = pipeline_name
    cfg["created_at"] = datetime.now().astimezone().isoformat()

    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)

    if keep_history:
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f")
        history_path = run_dir / f"config.{pipeline_name}.{timestamp}.json"
        with open(history_path, "w") as f:
            json.dump(cfg, f, indent=2)

    return path


def update_pipeline_config(
    run_dir: Path,
    pipeline_name: str,
    update_fn,
):
    """
    Explicitly update a pipeline config via a user-provided function.
    """
    path = run_dir / f"config.{pipeline_name}.json"

    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    with open(path, "r") as f:
        cfg = json.load(f)

    cfg = update_fn(cfg)
    cfg["updated_at"] = datetime.utcnow().isoformat() + "Z"

    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)

    return path