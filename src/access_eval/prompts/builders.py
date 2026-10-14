"""
Answer-prompt builders for datasets whose prompt is more than `template.format(q=...)` -- e.g. a fixed per-question
answer-options block, or few-shot demonstrations that depend on a dataset column.

A dataset opts in with `"prompt_builder": "<name>"` in its registry spec; answer_generation then builds each prompt
with `builder.build(formulation_text, dataset_row)` and folds `builder.prompt_name` / `builder.template_text` into the
generation config hash (so changing a builder is a new answer bucket, as editing a template is).
"""
import json
import re
from functools import lru_cache
from pathlib import Path

_ANSWER_DIR = Path(__file__).resolve().parent / "answer"


def _harness_options(block: str) -> str:
    """'A) foo\\nB) bar' -> 'A. foo\\nB. bar' (lm-evaluation-harness option format)."""
    out = []
    for i, ln in enumerate(block.split("\n")):
        m = re.match(r"^([A-D])\)\s?(.*)$", ln)
        if not m:
            raise ValueError(f"unexpected option line: {ln!r}")
        out.append(f"{'ABCD'[i]}. {m.group(2)}")
    return "\n".join(out)


def options_list(block: str):
    """'A) foo\\nB) bar' -> ['foo', 'bar']."""
    return [re.match(r"^[A-D]\)\s?(.*)$", ln).group(1) for ln in block.split("\n")]


class MMLUFiveShot:
    """lm-evaluation-harness MMLU 5-shot generative prompt (the paper's MMLU protocol):
    5 dev demonstrations of the question's subject, the (possibly paraphrased) stem, and the question's fixed options.
    Demonstrations ship in prompts/answer/mmlu_dev_shots.json (first 5 cais/mmlu dev rows per subject)."""

    # Same identity strings as the paper's MMLU runs, so their answers and these share a gen_config_hash.
    prompt_name = "mmlu_lmharness_5shot_generative_v1"
    template_text = ("The following are multiple choice questions (with answers) about {subject}.\\n\\n{5 dev demos "
                     "'Q\\nA. ..\\nAnswer: X'}{question}\\nA. ..\\nB. ..\\nC. ..\\nD. ..\\nAnswer:  |  demos=first 5 of "
                     "cais/mmlu dev per subject | extractor=mmlu_pilot.extract_letter")

    @staticmethod
    @lru_cache(maxsize=1)
    def _shots():
        return json.loads((_ANSWER_DIR / "mmlu_dev_shots.json").read_text())

    def build(self, text: str, row) -> str:
        subject = row["subject"]
        demos = "".join(d + "\n\n" for d in self._shots()[subject])
        return (f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
                f"{demos}{text.strip()}\n{_harness_options(row['options_block'])}\nAnswer:")


BUILDERS = {"mmlu_5shot": MMLUFiveShot}


def get_prompt_builder(name):
    if not name:
        return None
    if name not in BUILDERS:
        raise KeyError(f"unknown prompt_builder {name!r}; available: {sorted(BUILDERS)}")
    return BUILDERS[name]()
