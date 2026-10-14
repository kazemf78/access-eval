"""
Input and output locations of the paper scripts.

Inputs: paper/data/ (compact tables shipped with the repository; see paper/README.md).
Outputs: every script writes under $PAPER_REPRO_DIR (set by paper/reproduce.sh; default paper/reproduction/latest),
and reads the intermediates produced by the other paper scripts, such as the RQ1 per-question store, from there.
Reference versions (the outputs used in the paper) are in paper/reference/ and are never written.
"""
import os
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
DATA = PAPER / "data"
REFERENCE = PAPER / "reference"
REPRO = os.environ.get("PAPER_REPRO_DIR") or str(PAPER / "reproduction" / "latest")


def out_dir(default: Path, name: str) -> Path:
    """Where a script writes: PAPER_REPRO_DIR/<name> (`default` is kept for compatibility with the private repo)."""
    return Path(REPRO) / name


def rq1_dir() -> Path:
    """Folder holding the RQ1 per-question store rq1_per_question.pkl (written by rq1_selection_pass.py)."""
    return out_dir(None, "rq1")


def rq1_store() -> Path:
    return rq1_dir() / "rq1_per_question.pkl"
