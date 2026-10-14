# src/access_eval/evaluators/multi_answer.py
"""
Evaluator for factual-QA datasets that provide multiple valid answers / aliases
(e.g. TriviaQA, PopQA).

Ground-truth format handled (all arrive as CSV-serialised strings):
  - Python list  : "['Germany', 'german', 'Deutschland']"
  - TriviaQA dict: "{'value': 'David Seville', 'aliases': [...], ...}"
  - Plain string : "Albert Einstein"
"""

import ast
import re
from typing import List, Optional

from access_eval.evaluators.base import Evaluator
from access_eval.evaluators.registry import register_evaluator
from access_eval.data.schema import Answer, Evaluation, Question, EvaluationLabel
from access_eval.prompts.loader import load_prompt
from access_eval.models.factory import get_llm_for_evaluation


def _parse_valid_answers(raw: Optional[str]) -> List[str]:
    """
    Parse a ground_truth string into a deduplicated list of valid answer strings.

    Handles:
      - Python list literal  → use items directly
      - TriviaQA dict literal → combine 'value' + 'aliases'
      - Plain string          → wrap in a one-element list
    """
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, list):
            return [str(a) for a in parsed if a]
        if isinstance(parsed, dict):
            value   = parsed.get("value", "")
            aliases = parsed.get("aliases", [])
            seen, out = set(), []
            for a in [value] + list(aliases):
                a = str(a).strip()
                if a and a.lower() not in seen:
                    seen.add(a.lower())
                    out.append(a)
            return out
    except (ValueError, SyntaxError):
        pass
    return [str(raw).strip()]


def _format_valid_answers(answers: List[str]) -> str:
    return "\n".join(f"- {a}" for a in answers)


class MultiAnswerEvaluator(Evaluator):
    """
    LLM-as-judge evaluator for datasets with multiple valid answers / aliases.
    Uses the multi_answer prompt which presents valid answers as a bullet list.
    """

    def __init__(self, judge_model: str = "gpt-4.1-mini", judge_llm=None, **kwargs):
        super().__init__(name=f"MultiAnswer_{judge_model}")

        if judge_llm is not None:
            self.llm = judge_llm
        else:
            self.llm = get_llm_for_evaluation(judge_model, **kwargs)

        self.prompt_template = load_prompt("evaluation/multi_answer")

    def prepare(self):
        if hasattr(self.llm, "prepare"):
            self.llm.prepare()

    def _build_prompt(self, answer: Answer, ground_truth: Question) -> str:
        valid = _parse_valid_answers(ground_truth.ground_truth)
        if not valid:
            valid = ["(no ground truth provided)"]
        MAX_CHARS = 30_000
        pred = answer.text
        if len(pred) > MAX_CHARS:
            pred = pred[:MAX_CHARS] + "\n\n[TRUNCATED]"
        return self.prompt_template.format(
            question=ground_truth.text,
            valid_answers=_format_valid_answers(valid),
            predicted_answer=pred,
        )

    def _parse_verdict(self, response: str) -> EvaluationLabel:
        match = re.search(r"\b([ABC])\b", response.upper())
        return EvaluationLabel.from_letter(match.group(1) if match else "C")

    def evaluate_single(self, answer: Answer, ground_truth: Question) -> Evaluation:
        prompt   = self._build_prompt(answer, ground_truth)
        response = self.llm.single(prompt)
        return Evaluation(
            answer_idx=0,
            question_idx=answer.question_idx,
            model=answer.model,
            evaluator=self.name,
            label=self._parse_verdict(response),
            raw_response=response,
            original_question=answer.original_question,
            paraphrased_question=answer.paraphrased_question,
            answer_text=answer.text,
            ground_truth=ground_truth.ground_truth,
        )

    def evaluate_batch(self, answers: List[Answer], ground_truths: List[Question]) -> List[Evaluation]:
        prompts = [self._build_prompt(a, gt) for a, gt in zip(answers, ground_truths)]
        print(f"[MultiAnswerEvaluator] evaluating batch of {len(prompts)}")
        responses = self.llm.batch(prompts)
        return [
            Evaluation(
                answer_idx=0,
                question_idx=ans.question_idx,
                model=ans.model,
                evaluator=self.name,
                label=self._parse_verdict(resp),
                raw_response=resp,
                original_question=ans.original_question,
                paraphrased_question=ans.paraphrased_question,
                answer_text=ans.text,
                ground_truth=gt.ground_truth,
            )
            for ans, gt, resp in zip(answers, ground_truths, responses)
        ]


# Register shorthand name for use in YAML configs (evaluator: "multi_answer")
# register_evaluator("trivia_qa_rc", MultiAnswerEvaluator)  # prefer YAML config override
# register_evaluator("popqa", MultiAnswerEvaluator)         # prefer YAML config override
register_evaluator("multi_answer", MultiAnswerEvaluator)
