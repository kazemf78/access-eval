# src/access_eval/analysis/math_grading.py
"""
Math-dataset-specific grading helpers.

Depends on math_verify and sympy. Not imported by QA pipelines.
"""

import pandas as pd
import math_verify
from sympy import sympify, simplify
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanFunction


MAX_CHARS = 40_000


def score_math_answer(model_answer: str, gold_answer: str, max_chars: int = MAX_CHARS):
    """
    Grade a single model answer against a gold answer with the exact same
    math_verify configuration add_gold_columns() uses -- the single source
    of truth for "is this GSM8K/MATH answer correct", shared by the
    dataframe-based grading path here and prompt_optimization/metrics.py's
    optimization-time metric (added 2026-08-30 after finding those two used
    *different* graders: the metric did a lenient last-integer regex extract
    that doesn't penalize e.g. "\\boxed{27~\\text{dollars}}", while this
    strict parser fails to parse that at all (math_verify.parse() returns
    []) -- so GEPA could win its own search by producing units-in-the-box
    answers that then failed held-out grading. Same "optimize for one thing,
    measure another" mismatch class as the temperature bug, in the metric
    definition instead of the sampling condition.

    Returns (correct: bool, feedback: str).

    parsing_timeout=None on both parse() calls: math_verify's default timeout
    is implemented via signal.alarm(), which only works in a process's main
    thread -- add_gold_columns() (pandas .apply(), single-threaded) never
    hits this, but this function is called from dspy's metric-evaluation
    worker threads during optimization, which raised
    "ValueError: signal only works in main thread of the main interpreter"
    on every call (confirmed 2026-08-30, crashed a real GSM8K optimize run).
    math_verify's own parse() docstring recommends exactly this for
    multithreaded use; the tradeoff is no per-parse timeout, acceptable here
    since these are short numeric/LaTeX answers, not adversarial input.
    """
    gold_parsed = math_verify.parse(f"${gold_answer}$", parsing_timeout=None)
    ans_str = (model_answer or "").strip()

    if len(ans_str) > max_chars:
        return False, f"Answer too long ({len(ans_str)} chars, max {max_chars})."

    answer_parsed = math_verify.parse(
        ans_str,
        fallback_mode="no_fallback",
        extraction_config=[
            math_verify.LatexExtractionConfig(boxed_match_priority=0, try_extract_without_anchor=True),
        ],
        parsing_timeout=None,
    )

    if not answer_parsed:
        return False, (
            f"Could not parse a boxed answer from the response. The correct answer is "
            f"'{gold_answer}'. Ensure the final answer is in \\boxed{{}} with no extra "
            f"text (like units or words) inside the braces."
        )

    correct = math_verify.verify(gold_parsed, answer_parsed, timeout_seconds=None)
    if correct:
        return True, f"Correct. The answer is {gold_answer}."
    return False, f"Incorrect. Parsed answer was {answer_parsed}, but the correct answer is {gold_answer}."


def add_gold_columns(
    df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    max_chars: int = MAX_CHARS,
) -> pd.DataFrame:
    """
    Parse model answers and score them against gold answers.

    Adds columns:
      gold_answer        : raw gold answer string
      gold_answer_parsed : math_verify.parse() result for gold
      answer_len         : length of the model answer string
      answer_parsed      : math_verify.parse() result (None if answer is too long)
      correct            : bool, True iff answer_parsed verifies against gold

    Parameters
    ----------
    df          : answers DataFrame; must have 'idx' and 'answer' columns
    dataset_df  : original dataset DataFrame (indexed 0..N); must have 'answer' column
    max_chars   : answers longer than this are skipped (answer_parsed=None, correct=False)
    """
    df = df.copy()

    df["gold_answer"] = df["idx"].apply(lambda i: dataset_df.iloc[i]["answer"])
    df["gold_answer_parsed"] = df["idx"].apply(
        lambda i: math_verify.parse(f'${dataset_df.iloc[i]["answer"]}$')
    )

    ans_str = df["answer"].fillna("").astype(str)
    df["answer_len"] = ans_str.str.len()
    too_long = df["answer_len"] > max_chars

    df["answer_parsed"] = None

    to_parse = ~too_long
    df.loc[to_parse, "answer_parsed"] = ans_str[to_parse].apply(
        lambda ans: math_verify.parse(
            ans,
            fallback_mode="no_fallback",
            extraction_config=[
                math_verify.LatexExtractionConfig(
                    boxed_match_priority=0,
                    try_extract_without_anchor=True,
                ),
            ],
        )
    )

    df["correct"] = False
    df.loc[to_parse, "correct"] = df.loc[to_parse].apply(
        lambda row: math_verify.verify(row["gold_answer_parsed"], row["answer_parsed"]),
        axis=1,
    )

    return df


def extract_numeric_value(parsed) -> object:
    """
    Extract a plain numeric value from a math_verify.parse() result.

    Returns int, float, or str (for non-numeric simplified expressions),
    or None if parsing fails.

    Useful for value-level consistency analysis (distinct answer values
    across paraphrases, independent of binary correctness).
    """
    if not parsed:
        return None

    obj = parsed[0]
    expr_str = str(obj)

    if isinstance(obj, Relational):
        expr = obj
    else:
        try:
            expr = sympify(expr_str)
        except Exception:
            return None

    if isinstance(expr, (Relational, BooleanFunction)):
        try:
            rhs = expr.rhs
            if rhs.is_number:
                expr = rhs
        except Exception:
            pass

    simplified = simplify(expr)

    try:
        if simplified.is_Integer:
            return int(simplified)
        if simplified.is_Rational:
            return float(simplified) if simplified.q != 1 else int(simplified)
        if simplified.is_Float:
            iv = int(simplified)
            return iv if iv == simplified else float(simplified)

        return str(simplified)
    except ValueError:
        # A degenerate/garbage answer (typically from a weak local model) can
        # simplify to a symbolic expression with an astronomically large
        # integer sub-term -- stringifying it hits Python's built-in
        # int-to-str safety limit (default 4300 digits). Such an answer is
        # certainly wrong/unparseable anyway, so treat it as such rather than
        # crashing the whole alignment run over one row. Confirmed via direct
        # traceback: math_500_test_rebut_v00_more_models2 alignment crashed
        # here on 2026-07-28.
        return None


def add_normalized_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ans_norm and gold_norm columns via extract_numeric_value.

    Expects 'answer_parsed' and 'gold_answer_parsed' columns from add_gold_columns.
    Also adds 'correct_' as a sanity-check column (should match 'correct').
    """
    df = df.copy()
    df["ans_norm"]  = df["answer_parsed"].apply(extract_numeric_value)
    df["gold_norm"] = df["gold_answer_parsed"].apply(extract_numeric_value)
    df["correct_"]  = df["ans_norm"] == df["gold_norm"]
    return df


def filter_parseable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows with a non-empty, non-null parsed answer and a valid ans_norm.

    Requires add_gold_columns and add_normalized_values to have been called first.
    """
    return df[
        (df["answer_parsed"].str.len() > 0)
        & (df["answer_parsed"].notna())
        & (~df["ans_norm"].isnull())
    ]
