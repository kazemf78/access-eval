"""
The paper_v1 math grading policy ("fallback"; the grader used for the paper), packaged so
the evaluation stage grades GSM8K / MATH-500 exactly as the paper did. score_math_answer is inlined from
access_eval.analysis.math_grading: importing the analysis package pulls in its plotting stack.

grade_answer(ans, gold, policy):
  "fallback"  (paper default): strict parse; if it fails (or the answer is > 40k chars), fallback_grade().
  "incorrect" : strict parse; failures graded incorrect (paper_v1 behavior before 2026-09-24).
  "drop"      : strict parse; failures return None -> row excluded (pre-freeze preprocess_math_answers behavior).

fallback_grade chain (first parseable candidate wins, then math_verify.verify against gold):
  think_tail  : answer > 40k chars (thinking trace) that closed </think> -> strict parse of the text after it.
  fix_escape  : collapse literal double backslashes and drop a stray "\\" before a digit, then strict parse.
  boxed       : last \\boxed{...} content; a sentence -> its answer number (_sentence_number), else parsed as math.
  braces      : the same for a bare trailing {...} in the last 40 chars.
  final_line  : last line containing "final answer" -> _sentence_number of the text after its colon.
Never "last number anywhere in the response".
"""
import re

import math_verify

MAX_CHARS = 40_000


def score_math_answer(model_answer, gold_answer, max_chars=MAX_CHARS):
    """Verbatim logic of access_eval.analysis.math_grading.score_math_answer (inlined: importing the analysis
    package pulls in its plotting stack, which the evaluation image does not need). Returns (correct, feedback)."""
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
        return False, "Could not parse a boxed answer from the response."
    correct = math_verify.verify(gold_parsed, answer_parsed, timeout_seconds=None)
    return bool(correct), ("Correct." if correct else "Incorrect.")

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
WORD = re.compile(r"(?<![\\a-zA-Z])[a-zA-Z]{3,}")  # a 3+ letter word not part of a \latexcommand


def _last_braced(s, start):
    """Content of the brace group opening at s[start] == '{' (brace-matched), or None."""
    depth = 0
    for i in range(start, len(s)):
        depth += s[i] == "{"
        depth -= s[i] == "}"
        if depth == 0:
            return s[start + 1:i]
    return None


_AFTER_VERB = re.compile(r"(?:\b(?:is|are|was|were|be|equals)|=|:)\s*(?:approximately\s+|about\s+)?\$?\s*(-?\d[\d,]*(?:\.\d+)?)", re.I)
_AFTER_DOLLAR = re.compile(r"\$\s*(-?\d[\d,]*(?:\.\d+)?)")


def _sentence_number(text):
    """The answer number inside a natural-language sentence ("The turtle will crawl 120 feet in 40 seconds"):
    last number right after is/are/was/were/be/equals/=/: ; else first $-amount; else FIRST number (sentences
    state the answer first and trailing numbers are usually restated question context)."""
    for rx, pick in ((_AFTER_VERB, -1), (_AFTER_DOLLAR, 0)):
        m = rx.findall(text)
        if m:
            return m[pick].replace(",", "")
    nums = NUM.findall(text)
    return nums[0].replace(",", "") if nums else None


def _candidate_from_content(content):
    if content is None or not content.strip():
        return None
    if WORD.search(re.sub(r"\\text\{[^}]*\}", "", content)):  # a sentence, not a math expression
        return _sentence_number(content)
    return content


def _parse_cand(cand):
    return math_verify.parse(f"${cand}$", parsing_timeout=None) if cand else []


def fallback_grade(ans, gold):
    """Returns (correct, method, candidate) for an answer the strict parser failed on."""
    if len((ans or "").strip()) > 40_000:
        # thinking-mode answers carry the whole reasoning trace; if the model FINISHED (closed </think>),
        # grade only the final response after it. An unfinished trace (no </think>) stays incorrect.
        if "</think>" not in ans:
            return False, "too_long", None
        final = ans.rsplit("</think>", 1)[1]
        if len(final.strip()) > 40_000:
            return False, "too_long", None
        ok, fb = score_math_answer(final, gold)
        if not fb.startswith("Could not parse"):
            return ok, "think_tail", None
        ans = final  # finished but the final part is unparseable: continue with the text fallbacks on it
    gold_p = math_verify.parse(f"${gold}$", parsing_timeout=None)
    s = (ans or "").replace("\\\\", "\\")
    s = re.sub(r"\\(?=\d)", "", s)
    ok, fb = score_math_answer(s, gold)
    if not fb.startswith("Could not parse"):
        return ok, "fix_escape", None

    steps = []
    k = s.rfind("\\boxed{")
    if k >= 0:
        steps.append(("boxed", _candidate_from_content(_last_braced(s, k + len("\\boxed")))))
    tail = s.rstrip()[-40:]
    m = re.search(r"\{([^{}]*)\}\s*\.?\s*$", tail)
    if m:
        steps.append(("braces", _candidate_from_content(m.group(1))))
    lines = [ln for ln in s.splitlines() if "final answer" in ln.lower()]
    if lines:
        steps.append(("final_line", _sentence_number(lines[-1].split(":", 1)[-1])))

    for method, cand in steps:
        p = _parse_cand(cand)
        if p:
            return bool(math_verify.verify(gold_p, p, timeout_seconds=None)), method, cand
    return False, "unparseable", None




def grade_answer(ans, gold, policy="fallback"):
    """correct (bool) or None (= exclude row, only under policy "drop")."""
    ans = ans or ""
    if len(ans.strip()) <= MAX_CHARS:
        ok, fb = score_math_answer(ans, gold)
        if not fb.startswith("Could not parse"):
            return ok
    if policy == "incorrect":
        return False
    if policy == "drop":
        return None
    return fallback_grade(ans, gold)[0]
