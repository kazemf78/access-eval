# ============================================================================
# File: src/access_eval/evaluators/simple_qa.py
# ============================================================================
"""
Evaluator for SimpleQA dataset.
"""

import re
from typing import Optional, List

from access_eval.evaluators.base import Evaluator
from access_eval.evaluators.registry import register_evaluator
from access_eval.data.schema import Answer, Evaluation, Question, EvaluationLabel
from access_eval.prompts.loader import load_prompt
from access_eval.models.factory import get_llm_for_evaluation


class SimpleQAEvaluator(Evaluator):
    """
    Evaluator for SimpleQA dataset.
    Uses LLM judge to grade as CORRECT/INCORRECT/NOT_ATTEMPTED.
    """
    
    def __init__(self, judge_model: str = "gpt-4.1-mini", judge_llm=None, **kwargs):
        """
        Args:
            judge_model: Model name for grading
            judge_llm: Pre-initialized LLM (optional, overrides judge_model)
        """
        super().__init__(name=f"SimpleQA_{judge_model}")
        
        if judge_llm is not None:
            self.llm = judge_llm
        else:
            self.llm = get_llm_for_evaluation(judge_model, **kwargs)
        
        self.prompt_template = load_prompt("evaluation/SimpleQA")
    
    def prepare(self):
        """Initialize the LLM."""
        if hasattr(self.llm, 'prepare'):
            self.llm.prepare()
    
    def evaluate_single(self, answer: Answer, ground_truth: Question) -> Evaluation:
        """Evaluate a single answer."""
        MAX_CHARS = 30000  # to avoid VLLM length issues
        prompt = self.prompt_template.format(
            question=ground_truth.text,
            target=ground_truth.ground_truth,
            predicted_answer=answer.text if len(answer.text) < MAX_CHARS else answer.text[:MAX_CHARS] + "\n\n[TRUNCATED]",
        )
        
        response = self.llm.single(prompt)
        label = self._parse_verdict(response)
        
        return Evaluation(
            answer_idx=0,  # NOTE: Will be set by pipeline if needed
            question_idx=answer.question_idx,
            model=answer.model,
            evaluator=self.name,
            label=label,
            raw_response=response,
            # Verbose fields for reliable joining
            original_question=answer.original_question,
            paraphrased_question=answer.paraphrased_question,
            answer_text=answer.text, # CAN BE COMMENTED OUT TO SAVE SPACE?
            ground_truth=ground_truth.ground_truth,
        )
    
    def evaluate_batch(self, answers: List[Answer], ground_truths: List[Question]) -> List[Evaluation]:
        """Batch evaluation."""
        MAX_CHARS = 30000  # to avoid VLLM length issues
        prompts = [
            self.prompt_template.format(
                question=gt.text,
                target=gt.ground_truth,
                predicted_answer=ans.text if len(ans.text) < MAX_CHARS else ans.text[:MAX_CHARS] + "\n\n[TRUNCATED]",
            )
            for ans, gt in zip(answers, ground_truths)
        ]
        print("evaluating batch of size:", len(prompts))
        print(type(self.llm))
        responses = self.llm.batch(prompts)
        
        return [
            Evaluation(
                answer_idx=0,  # NOTE: Will be set by pipeline if needed
                question_idx=ans.question_idx,
                model=ans.model,
                evaluator=self.name,
                label=self._parse_verdict(resp),
                raw_response=resp,
                # Verbose fields for reliable joining
                original_question=ans.original_question,
                paraphrased_question=ans.paraphrased_question,
                answer_text=ans.text, # CAN BE COMMENTED OUT TO SAVE SPACE?
                ground_truth=gt.ground_truth,
            )
            for ans, gt, resp in zip(answers, ground_truths, responses)
        ]
    
    def _parse_verdict(self, response: str) -> EvaluationLabel:
        """Extract A/B/C from response."""
        match = re.search(r'\b([ABC])\b', response.upper())
        letter = match.group(1) if match else "C"
        return EvaluationLabel.from_letter(letter)


# Register this evaluator
register_evaluator("SimpleQA", SimpleQAEvaluator)
register_evaluator("simple_qa", SimpleQAEvaluator)
