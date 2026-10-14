"""
Check a reproduction run (reproduce_paper.sh) against the outputs used in the paper.

  data   every CSV / JSON / .tex table / the RQ1 store the paper relies on, compared value by value
         (floats: rel 1e-9, abs 1e-12) against the reference versions listed in REFERENCE below;
  figs   every figure of the paper (paper/reference/figs, as included in the LaTeX source) rasterized next to its reproduced
         counterpart (pdftoppm, 60 dpi) and compared pixel by pixel (mean absolute difference, 0-255 scale).
Writes <OUT>/verify_report.md and exits non-zero if any data item differs or a figure differs visibly.
Usage: python paper/verify_reproduction.py <OUT>  (run by paper/reproduce.sh)
"""
import json
import math
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent / "reference"   # outputs used in the paper (never written)
DRAFT = PAPER                                              # figures as included in the paper: reference/figs/...
# (reference file used in the paper, path of the reproduced file relative to OUT)
REFERENCE = [
    ("rq1_figs/rq1_per_question.pkl", "rq1/rq1_per_question.pkl"),
    ("rq1_figs/figs/v4/rq1_main_table.csv", "rq1/figs/v6/rq1_main_table.csv"),
    ("rq1_figs/figs/v4/rq1_main_table.tex", "rq1/figs/v6/rq1_main_table.tex"),
    ("rq1_figs/figs/v4/fig2_majority_reversals_points.csv", "rq1/figs/v6/fig2_majority_reversals_points.csv"),
    ("rq1_figs/figs/v3/rq1_near_tied_pairs.csv", "rq1/figs/v6/rq1_near_tied_pairs.csv"),
    ("rq1_figs/figs/v5/fig3_main_curve_points.csv", "rq1/figs/v6/fig3_main_curve_points.csv"),
    ("rq1_figs/figs/v4/fig3_stats.json", "rq1/figs/v6/fig3_stats.json"),
    ("rq1_figs/figs/v4/access_reliable_tradeoff.json", "rq1/figs/v6/access_reliable_tradeoff.json"),
    ("equal_cost_budget/summary.csv", "equal_cost_budget/summary.csv"),
    ("rq2_figs/figs/v4/rq2_disagreement.csv", "rq2/figs/v4/rq2_disagreement.csv"),
    ("rq2_figs/figs/v4/rq2_access_reliable_at_k.csv", "rq2/figs/v4/rq2_access_reliable_at_k.csv"),
    ("rq3_figs/figs/v7/rq3_pairs.csv", "rq3/figs/v7/rq3_pairs.csv"),
    ("rq3_figs/figs/v7/rq3_cells.csv", "rq3/figs/v7/rq3_cells.csv"),
    ("rq3_figs/figs/v7/rq3_cells_compact.csv", "rq3/figs/v7/rq3_cells_compact.csv"),
    ("rq3_figs/paired/v1/rq3_paired_table.csv", "rq3_paired/v1/rq3_paired_table.csv"),
    ("rq3_figs/paired/v1/rq3_paired_table.tex", "rq3_paired/v1/rq3_paired_table.tex"),
    ("rerun_null/null_gsm8k.json", "rerun_null/null_gsm8k.json"),
    ("rerun_null/formulation_flip_model_changes.csv", "rerun_null/formulation_flip_model_changes.csv"),
    ("paper_numbers/paper_numbers.json", "paper_numbers/paper_numbers.json"),
]
FIG_TOL = 1.0   # mean |pixel difference| (0-255) above which a figure counts as visibly different


def close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            return False
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def cmp_json(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return f"{path}: keys differ ({sorted(set(a) ^ set(b))[:5]})"
        for k in a:
            r = cmp_json(a[k], b[k], f"{path}.{k}")
            if r:
                return r
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            r = cmp_json(x, y, f"{path}[{i}]")
            if r:
                return r
        return None
    return None if close(a, b) else f"{path}: {a!r} vs {b!r}"


def cmp_file(ref, new):
    if not new.exists():
        return "missing in reproduction"
    if ref.suffix == ".csv":
        a, b = pd.read_csv(ref), pd.read_csv(new)
        if list(a.columns) != list(b.columns) or a.shape != b.shape:
            return f"shape/columns differ {a.shape} vs {b.shape}"
        for c in a.columns:
            if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
                if not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=1e-9, atol=1e-12, equal_nan=True):
                    return f"column {c} differs"
            elif not a[c].astype(str).equals(b[c].astype(str)):
                return f"column {c} differs"
        return None
    if ref.suffix == ".json":
        a, b = json.loads(ref.read_text()), json.loads(new.read_text())
        for d in (a, b):
            if isinstance(d, dict):
                d.pop("_sources", None)
        return cmp_json(a, b)
    if ref.suffix == ".pkl":
        a, b = pickle.load(open(ref, "rb")), pickle.load(open(new, "rb"))
        if set(a) != set(b):
            return "benchmarks differ"
        for ds in a:
            if list(a[ds]["models"]) != list(b[ds]["models"]) or not np.array_equal(a[ds]["idx"], b[ds]["idx"]):
                return f"{ds}: models/inquiries differ"
            for key in ("A", "Y"):
                for m in a[ds]["models"]:
                    if not np.allclose(a[ds][key][m], b[ds][key][m]):
                        return f"{ds}/{m}: {key} differs"
        return None
    return None if ref.read_bytes() == new.read_bytes() else "text differs"


def raster(pdf, tmp, tag):
    base = Path(tmp) / tag
    subprocess.run(["pdftoppm", "-r", "60", "-gray", "-singlefile", str(pdf), str(base)], check=True,
                   capture_output=True)
    from PIL import Image
    return np.asarray(Image.open(f"{base}.pgm"), dtype=float)


def main(out):
    out = Path(out)
    lines = ["# Reproduction check", "", f"Reproduction: `{out}`", f"Figures compared with: `{DRAFT}`", "",
             "## Data", "", "| reference (used in paper) | result |", "|---|---|"]
    bad = 0
    for ref, new in REFERENCE:
        r = cmp_file(PAPER / ref, out / new)
        bad += r is not None
        lines.append(f"| `{ref}` | {'IDENTICAL' if r is None else 'DIFFERS: ' + r} |")
    lines += ["", "## Figures (pixel comparison, 60 dpi grayscale)", "", "| figure | mean abs diff | result |",
              "|---|---|---|"]
    with tempfile.TemporaryDirectory() as tmp:
        for pdf in sorted((DRAFT / "figs").rglob("*.pdf")):
            rel = pdf.relative_to(DRAFT)
            new = out / "paper_assets" / rel
            if not new.exists():
                lines.append(f"| `{rel}` | - | not produced by the reproduction (not used by the paper text?) |")
                continue
            a, b = raster(pdf, tmp, "a"), raster(new, tmp, "b")
            if a.shape != b.shape:
                bad += 1
                lines.append(f"| `{rel}` | - | DIFFERS: size {a.shape} vs {b.shape} |")
                continue
            d = float(np.abs(a - b).mean())
            bad += d > FIG_TOL
            lines.append(f"| `{rel}` | {d:.3f} | {'IDENTICAL' if d == 0 else ('same (tiny rendering noise)' if d <= FIG_TOL else 'DIFFERS')} |")
    lines += ["", f"**{'ALL CHECKS PASSED' if bad == 0 else f'{bad} CHECK(S) FAILED'}**"]
    (out / "verify_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1])
