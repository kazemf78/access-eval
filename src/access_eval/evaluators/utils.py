# src/access_eval/evaluators/utils.py
"""
Shared utilities for evaluators.

call_llm_judge / call_llm_judge_batch are available for use inside
custom evaluate_row / evaluate_batch functions.
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# LLM judge singleton — set by the pipeline before custom evaluation runs
# ---------------------------------------------------------------------------

_judge_llm = None


def set_judge(model_name: str) -> None:
    """
    Initialize the module-level judge LLM used by call_llm_judge().
    Called automatically by the pipeline when evaluator_file is specified.
    Users do not need to call this directly.
    """
    global _judge_llm
    from access_eval.models.factory import get_llm_for_evaluation
    _judge_llm = get_llm_for_evaluation(model_name, max_tokens=32)
    _judge_llm.prepare()
    print(f"[evaluators.utils] Judge LLM ready: {model_name}")


def call_llm_judge(template: str, **kwargs) -> bool:
    """
    Call the LLM judge with a custom prompt template. Returns True (correct)
    or False (incorrect) based on a YES/NO response.

    Use this inside evaluate_row() for single-row LLM-based evaluation.
    For batch evaluation use call_llm_judge_batch() instead — it's much faster.

    Args:
        template: A Python format string. Standard placeholders: {answer},
                  {ground_truth}. Add any other {key} and pass it as a kwarg.
        **kwargs: Values for the template placeholders.

    Returns:
        True if the judge responds YES, False otherwise.

    Example:
        PROMPT = \"\"\"
        Question: {question}
        Ground truth: {ground_truth}
        Model answer: {answer}
        Is the model's answer correct? Reply YES or NO only.
        \"\"\"

        def evaluate_row(answer, ground_truth, row):
            return call_llm_judge(
                PROMPT,
                answer=answer,
                ground_truth=ground_truth,
                question=row["question"],
            )
    """
    _require_judge()
    prompt = template.format(**kwargs)
    response = (_judge_llm.single(prompt) or "").strip()
    return _parse_yes_no(response)


def call_llm_judge_batch(
    template: str,
    answers: list,
    ground_truths: list,
    rows: list,
    api_conc: int = 16,
) -> list:
    """
    Batch version of call_llm_judge. Runs all prompts concurrently via the
    LLM's batch() method — much faster than calling call_llm_judge() in a loop.

    Use this inside evaluate_batch() for efficient LLM-based evaluation.

    Args:
        template: A Python format string. {answer} and {ground_truth} are
                  filled from the lists. Any other {key} is filled from the
                  corresponding row dict (e.g. {question}, {category}).
        answers: List of model answer strings.
        ground_truths: List of ground truth strings (same length as answers).
        rows: List of row dicts from the dataset CSV (same length as answers).
        api_conc: Max concurrent API calls (default 16).

    Returns:
        list[bool] — True where the judge says YES, False where NO.

    Example:
        PROMPT = \"\"\"
        Question: {question}
        Ground truth: {ground_truth}
        Model answer: {answer}
        Is the model's answer correct? Reply YES or NO only.
        \"\"\"

        def evaluate_batch(answers, ground_truths, rows):
            return call_llm_judge_batch(PROMPT, answers, ground_truths, rows)
    """
    _require_judge()
    if not (len(answers) == len(ground_truths) == len(rows)):
        raise ValueError("answers, ground_truths, and rows must all have the same length")

    prompts = []
    for answer, ground_truth, row in zip(answers, ground_truths, rows):
        # row values fill any extra {placeholders}; answer/ground_truth take precedence
        merged = {**row, "answer": answer, "ground_truth": ground_truth}
        prompts.append(template.format(**merged))

    responses = _judge_llm.batch(prompts, concurrency=api_conc)
    return [_parse_yes_no(r or "") for r in responses]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_judge() -> None:
    if _judge_llm is None:
        raise RuntimeError(
            "No judge LLM initialized. "
            "Ensure 'models.judge' is set in your YAML config — "
            "the pipeline calls set_judge() automatically before evaluation starts."
        )


def _parse_yes_no(response: str) -> bool:
    r = response.strip().upper()
    if r.startswith("YES"):
        return True
    if r.startswith("NO"):
        return False
    # Fuzzy fallback: YES present and NO absent
    has_yes = bool(re.search(r"\bYES\b", r))
    has_no  = bool(re.search(r"\bNO\b", r))
    return has_yes and not has_no


# ---------------------------------------------------------------------------
# Math utilities (stubs — used by math evaluators)
# ---------------------------------------------------------------------------

def extract_number(text: str) -> Optional[float]:
    pass


def normalize_number(value: Optional[float]) -> Optional[float]:
    pass
