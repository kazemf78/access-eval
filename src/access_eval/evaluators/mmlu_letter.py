# ============================================================================
# File: src/access_eval/evaluators/mmlu_letter.py
# ============================================================================
"""
Deterministic evaluator for multiple-choice answers (MMLU): extract the chosen letter with the paper's rules
(the paper's MMLU letter extractor) and compare it with the gold letter, i.e. the position of the dataset's
answer text in the question's options block. An unparseable answer is INCORRECT (paper policy: never re-judged by
an LLM). No judge model is used; run it with judge name "rule".
"""
import hashlib
import inspect
import re
from typing import List, Optional

from access_eval.data.schema import Answer, Evaluation, EvaluationLabel, Question
from access_eval.evaluators.base import Evaluator
from access_eval.evaluators.registry import register_evaluator
from access_eval.prompts.builders import options_list


def extract_letter(text) -> Optional[str]:
    """After dropping a closed <think>..</think> trace: the LAST 'answer/choice/option (is) [:] X' statement, else the
    LAST line starting with a bare option marker ('B', 'B.', '(B)', '**B.'); else None. No 'first letter anywhere'
    fallback (it would grab the article 'A' in prose)."""
    t = "" if text is None or isinstance(text, float) else str(text)
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[1]
    m = re.findall(r"(?:answer|choice|option)\s*(?:is)?\s*:?\s*\**\s*\(?([ABCD])\b(?![a-z])", t, flags=re.I)
    if m:
        return m[-1].upper()
    m = re.findall(r"^\s*\**\(?([ABCD])(?:[\).:]|\**\s*$)", t, flags=re.M)
    return m[-1] if m else None


_FINAL_PATTERNS = [
    r"(?:final\s+)?answer\s*(?:is|:)?\s*[:\-]?\s*\**\s*\(?([ABCD])\b(?![a-z])",   # "the answer is **D**", "Answer: D"
    r"\*\*\(?([ABCD])[\).]?\*\*",                                                # a bolded option, "**D**" / "**C.**"
    r"^\s*\(?([ABCD])[\).:]?\s*$",                                                 # a bare option on its own line
]


def extract_letter_final(text, tail_chars: int = 300) -> Optional[str]:
    """OPTIONAL (not the paper default): the letter stated in the closing part of the answer (last tail_chars after
    any closed <think>), trying an explicit 'answer is X', then a bolded option, then a bare option line; falls back to
    the paper's extract_letter. Fixes answers that discuss other options in prose before stating their choice."""
    t = "" if text is None or isinstance(text, float) else str(text)
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[1]
    tail = t[-tail_chars:]
    for pat in _FINAL_PATTERNS:
        m = re.findall(pat, tail, flags=re.I | re.M)
        if m:
            return m[-1].upper()
    return extract_letter(t)


class MCQLetterEvaluator(Evaluator):
    """Paper default extraction (extract_letter)."""
    extractor = staticmethod(extract_letter)
    eval_name = "mcq_letter"

    def __init__(self, judge_model: str = "rule", **kwargs):
        super().__init__(name=self.eval_name)
        src = inspect.getsource(extract_letter) + inspect.getsource(MCQLetterEvaluator)
        if self.extractor is not extract_letter:
            src += inspect.getsource(extract_letter_final) + repr(_FINAL_PATTERNS)
        self.prompt_template = f"{self.eval_name} src={hashlib.sha256(src.encode()).hexdigest()[:16]}"

    @staticmethod
    def gold_letter(ground_truth: Question) -> Optional[str]:
        meta = ground_truth.metadata or {}
        opts = options_list(meta["options_block"])
        gt = ground_truth.ground_truth
        return "ABCD"[opts.index(gt)] if gt in opts else None

    def evaluate_single(self, answer: Answer, ground_truth: Question) -> Evaluation:
        pred, gold = self.extractor(answer.text), self.gold_letter(ground_truth)
        label = EvaluationLabel.CORRECT if (pred is not None and pred == gold) else EvaluationLabel.INCORRECT
        return Evaluation(
            answer_idx=0,
            question_idx=answer.question_idx,
            model=answer.model,
            evaluator=self.name,
            label=label,
            extracted_value=pred,
            raw_response=None if pred is not None else "PARSE_FAILED",
            original_question=answer.original_question,
            paraphrased_question=answer.paraphrased_question,
            answer_text=answer.text,
            ground_truth=gold,
        )

    def evaluate_batch(self, answers: List[Answer], ground_truths: List[Question]) -> List[Evaluation]:
        return [self.evaluate_single(a, g) for a, g in zip(answers, ground_truths)]


class MCQLetterFinalEvaluator(MCQLetterEvaluator):
    """Optional end-anchored extraction (extract_letter_final); select with evaluator_name='mcq_letter_final'."""
    extractor = staticmethod(extract_letter_final)
    eval_name = "mcq_letter_final"


for _name in ("mmlu", "mcq_letter"):
    register_evaluator(_name, MCQLetterEvaluator)
register_evaluator("mcq_letter_final", MCQLetterFinalEvaluator)
