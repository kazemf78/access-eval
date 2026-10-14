# src/access_eval/evaluators/base.py
"""
Base classes for answer evaluators.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from access_eval.data.schema import Answer, Evaluation, Question


class Evaluator(ABC):
    """Base class for all answer evaluators."""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def evaluate_single(self, answer: Answer, ground_truth: Question) -> Evaluation:
        """
        Evaluate a single answer against ground truth.
        
        Args:
            answer: The Answer object to evaluate
            ground_truth: The Question object with ground truth
            
        Returns:
            Evaluation object with results
        """
        pass
    
    def evaluate_batch(self, answers: List[Answer], ground_truths: List[Question]) -> List[Evaluation]:
        """
        Evaluate a batch of answers. Default implementation calls evaluate_single.
        Override for batch-optimized evaluation (e.g., batched LLM calls).
        
        Args:
            answers: List of Answer objects
            ground_truths: List of Question objects (must match order)
            
        Returns:
            List of Evaluation objects
        """
        assert len(answers) == len(ground_truths), "Answers and ground truths must match"
        return [self.evaluate_single(ans, gt) for ans, gt in zip(answers, ground_truths)]
    
    def prepare(self):
        """Optional preparation step (e.g., load models, connect to APIs)."""
        pass
    
    def cleanup(self):
        """Optional cleanup step (e.g., free GPU memory)."""
        pass


class LLMEvaluator(Evaluator):
    """
    Base class for evaluators that use an LLM as a judge.
    """
    
    def __init__(self, llm, prompt_template: str, name: Optional[str] = None):
        """
        Args:
            llm: An LLM instance (OpenAIAPILLM, VLLMLocalLLM, etc.)
            prompt_template: Template string for evaluation prompts
            name: Optional custom name for this evaluator
        """
        super().__init__(name)
        self.llm = llm
        self.prompt_template = prompt_template
    
    @abstractmethod
    def build_prompt(self, answer: Answer, ground_truth: Question) -> str:
        """Build evaluation prompt for a single answer."""
        pass
    
    @abstractmethod
    def parse_response(self, response: str, answer: Answer, ground_truth: Question) -> Evaluation:
        """Parse LLM response into an Evaluation object."""
        pass
    
    def evaluate_single(self, answer: Answer, ground_truth: Question) -> Evaluation:
        """Evaluate using LLM."""
        prompt = self.build_prompt(answer, ground_truth)
        response = self.llm.single(prompt)
        return self.parse_response(response, answer, ground_truth)
    
    def evaluate_batch(self, answers: List[Answer], ground_truths: List[Question]) -> List[Evaluation]:
        """Batch evaluation using LLM."""
        assert len(answers) == len(ground_truths), "Mismatch in batch sizes"
        
        prompts = [self.build_prompt(ans, gt) for ans, gt in zip(answers, ground_truths)]
        
        # Use LLM's batch method
        responses = self.llm.batch(prompts)
        
        return [
            self.parse_response(resp, ans, gt)
            for resp, ans, gt in zip(responses, answers, ground_truths)
        ]
    
    def prepare(self):
        """Prepare the LLM."""
        if hasattr(self.llm, 'prepare'):
            self.llm.prepare()


class EnsembleEvaluator(Evaluator):
    """
    Aggregates results from multiple evaluators.
    """
    
    def __init__(self, evaluators: List[Evaluator], aggregation: str = "majority", name: Optional[str] = None):
        """
        Args:
            evaluators: List of Evaluator instances
            aggregation: How to combine results ("majority", "unanimous", "any_correct")
            name: Optional custom name
        """
        super().__init__(name or "EnsembleEvaluator")
        self.evaluators = evaluators
        self.aggregation = aggregation
    
    def evaluate_single(self, answer: Answer, ground_truth: Question) -> Evaluation:
        """Evaluate with all evaluators and aggregate."""
        evaluations = [ev.evaluate_single(answer, ground_truth) for ev in self.evaluators]
        return self._aggregate(evaluations, answer, ground_truth)
    
    def evaluate_batch(self, answers: List[Answer], ground_truths: List[Question]) -> List[Evaluation]:
        """Batch evaluate with all evaluators."""
        all_evaluations = [
            ev.evaluate_batch(answers, ground_truths)
            for ev in self.evaluators
        ]
        
        # Transpose: from [evaluator][answer] to [answer][evaluator]
        grouped = [
            [all_evaluations[ev_idx][ans_idx] for ev_idx in range(len(self.evaluators))]
            for ans_idx in range(len(answers))
        ]
        
        return [
            self._aggregate(evals, answers[i], ground_truths[i])
            for i, evals in enumerate(grouped)
        ]
    
    def _aggregate(self, evaluations: List[Evaluation], answer: Answer, ground_truth: Question) -> Evaluation:
        """Aggregate multiple evaluations into one."""
        from collections import Counter
        from access_eval.data.schema import EvaluationLabel
        
        labels = [e.label for e in evaluations]
        label_counts = Counter(labels)
        
        if self.aggregation == "majority":
            majority_label = label_counts.most_common(1)[0][0]
        elif self.aggregation == "unanimous":
            majority_label = labels[0] if len(set(labels)) == 1 else EvaluationLabel.NOT_ATTEMPTED
        elif self.aggregation == "any_correct":
            majority_label = EvaluationLabel.CORRECT if EvaluationLabel.CORRECT in labels else label_counts.most_common(1)[0][0]
        else:
            majority_label = label_counts.most_common(1)[0][0]
        
        return Evaluation(
            answer_idx=evaluations[0].answer_idx,
            question_idx=answer.question_idx,
            model=answer.model,
            evaluator=self.name,
            label=majority_label,
            # Preserve verbose fields from first evaluation
            original_question=evaluations[0].original_question,
            paraphrased_question=evaluations[0].paraphrased_question,
            answer_text=evaluations[0].answer_text,
            metadata={
                "individual_evaluations": [e.to_dict() for e in evaluations],
                "label_distribution": dict(label_counts),
            }
        )
    
    def prepare(self):
        """Prepare all evaluators."""
        for ev in self.evaluators:
            ev.prepare()
    
    def cleanup(self):
        """Cleanup all evaluators."""
        for ev in self.evaluators:
            ev.cleanup()