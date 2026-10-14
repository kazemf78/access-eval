# ============================================================================
# File: src/access_eval/evaluators/registry.py
# ============================================================================
"""
Registry for dataset evaluators.
"""

from typing import Dict, Type, Any
from access_eval.evaluators.base import Evaluator


# Global registry (populated by imports)
EVALUATOR_REGISTRY: Dict[str, Type[Evaluator]] = {}


def register_evaluator(dataset: str, evaluator_class: Type[Evaluator]):
    """
    Register an evaluator for a dataset.
    
    Args:
        dataset: Dataset name
        evaluator_class: Evaluator class
    """
    EVALUATOR_REGISTRY[dataset] = evaluator_class


def get_evaluator(dataset: str, evaluator_name: str | None = None, **kwargs) -> Evaluator:
    """
    Factory function to get evaluator for a dataset.

    If evaluator_name is provided (e.g. from a YAML config), it overrides the
    dataset-name lookup. Valid shorthand names: "simple_qa", "multi_answer",
    "truthful_qa". Falls back to SimpleQAEvaluator for unknown datasets.

    Args:
        dataset: Dataset name (used when evaluator_name is not given)
        evaluator_name: Optional explicit evaluator shorthand from config
        **kwargs: Arguments to pass to evaluator constructor

    Returns:
        Evaluator instance

    Example:
        evaluator = get_evaluator("SimpleQA", judge_model="gpt-4o")
        evaluator = get_evaluator("my_dataset", evaluator_name="multi_answer", judge_model="gpt-4.1-mini")
    """
    key = evaluator_name if evaluator_name else dataset
    if key not in EVALUATOR_REGISTRY:
        print(
            f"[evaluators] No dedicated evaluator for '{key}' — "
            f"falling back to SimpleQAEvaluator (LLM-as-judge)."
        )
        from access_eval.evaluators.simple_qa import SimpleQAEvaluator
        return SimpleQAEvaluator(**kwargs)

    evaluator_class = EVALUATOR_REGISTRY[key]
    return evaluator_class(**kwargs)



