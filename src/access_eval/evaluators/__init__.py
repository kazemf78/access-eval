
# ============================================================================
# File: src/access_eval/evaluators/__init__.py
# ============================================================================
"""
Evaluators for different datasets.

Usage:
    from access_eval.evaluators import SimpleQAEvaluator, get_evaluator
    
    # Direct instantiation
    evaluator = SimpleQAEvaluator(judge_model="gpt-4o")
    
    # Factory pattern
    evaluator = get_evaluator("SimpleQA", judge_model="gpt-4o")
"""

from access_eval.evaluators.base import Evaluator, EnsembleEvaluator
from access_eval.evaluators.simple_qa import SimpleQAEvaluator
from access_eval.evaluators.truthful_qa import TruthfulQAEvaluator
from access_eval.evaluators.multi_answer import MultiAnswerEvaluator
from access_eval.evaluators.function_evaluator import FunctionEvaluator, load_from_file
from access_eval.evaluators.math_answer import MathAnswerEvaluator
from access_eval.evaluators.mmlu_letter import MCQLetterEvaluator
from access_eval.evaluators.utils import call_llm_judge, call_llm_judge_batch, set_judge
from access_eval.evaluators.registry import (
    get_evaluator,
    register_evaluator,
    EVALUATOR_REGISTRY,
)

__all__ = [
    "Evaluator",
    "EnsembleEvaluator",
    "SimpleQAEvaluator",
    "TruthfulQAEvaluator",
    "MultiAnswerEvaluator",
    "FunctionEvaluator",
    "MathAnswerEvaluator",
    "MCQLetterEvaluator",
    "load_from_file",
    "call_llm_judge",
    "call_llm_judge_batch",
    "set_judge",
    "get_evaluator",
    "register_evaluator",
    "EVALUATOR_REGISTRY",
]