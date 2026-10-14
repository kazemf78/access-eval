# src/access_eval/evaluators/function_evaluator.py
"""
FunctionEvaluator — wraps user-supplied evaluate_row / evaluate_batch functions.

The user provides a Python file that defines at minimum:

    def evaluate_row(answer: str, ground_truth: str, row: dict) -> bool: ...

And optionally, for efficient batched evaluation:

    def evaluate_batch(
        answers: list[str],
        ground_truths: list[str],
        rows: list[dict],
    ) -> list[bool]: ...

`row` / `rows` contain every column from the dataset CSV as plain dicts.
Functions can use any logic — string matching, external libraries, or the
call_llm_judge / call_llm_judge_batch utilities from evaluators.utils.

Reference this file in your YAML config:
    evaluation:
      evaluator_file: "path/to/my_eval.py"

Example file with LLM judge + batch support:
    from access_eval.evaluators.utils import call_llm_judge, call_llm_judge_batch

    PROMPT = \"\"\"
    Question: {question}
    Ground truth: {ground_truth}
    Model answer: {answer}
    Is the model's answer correct? Reply YES or NO only.
    \"\"\"

    def evaluate_row(answer, ground_truth, row):
        return call_llm_judge(PROMPT, answer=answer, ground_truth=ground_truth,
                              question=row["question"])

    def evaluate_batch(answers, ground_truths, rows):
        return call_llm_judge_batch(PROMPT, answers, ground_truths, rows)
"""

import importlib.util
from pathlib import Path
from typing import Callable, List, Optional

from access_eval.evaluators.base import Evaluator
from access_eval.data.schema import Answer, Evaluation, Question, EvaluationLabel


class FunctionEvaluator(Evaluator):
    """
    Wraps user-supplied evaluate_row() and optionally evaluate_batch().
    If evaluate_batch is not provided, falls back to sequential evaluate_single calls.
    """

    def __init__(
        self,
        fn: Callable,
        batch_fn: Optional[Callable] = None,
        name: str = "custom",
    ):
        super().__init__(name=name)
        self.fn = fn
        self.batch_fn = batch_fn

    def evaluate_single(self, answer: Answer, ground_truth: Question) -> Evaluation:
        row = ground_truth.metadata or {}
        result = self.fn(
            answer=answer.text,
            ground_truth=ground_truth.ground_truth or "",
            row=row,
        )
        if not isinstance(result, bool):
            raise TypeError(
                f"evaluate_row must return bool, got {type(result).__name__}: {result!r}"
            )
        return self._make_evaluation(answer, ground_truth, result)

    def evaluate_batch(
        self, answers: List[Answer], ground_truths: List[Question]
    ) -> List[Evaluation]:
        if self.batch_fn is not None:
            results = self.batch_fn(
                answers=[a.text for a in answers],
                ground_truths=[q.ground_truth or "" for q in ground_truths],
                rows=[q.metadata or {} for q in ground_truths],
            )
            if len(results) != len(answers):
                raise ValueError(
                    f"evaluate_batch returned {len(results)} results for {len(answers)} inputs"
                )
            if results and not isinstance(results[0], bool):
                raise TypeError(
                    f"evaluate_batch must return list[bool], got {type(results[0]).__name__}"
                )
            return [
                self._make_evaluation(ans, gt, r)
                for ans, gt, r in zip(answers, ground_truths, results)
            ]
        # No batch function — fall back to sequential
        return [self.evaluate_single(ans, gt) for ans, gt in zip(answers, ground_truths)]

    def _make_evaluation(
        self, answer: Answer, ground_truth: Question, result: bool
    ) -> Evaluation:
        return Evaluation(
            answer_idx=0,
            question_idx=answer.question_idx,
            model=answer.model,
            evaluator=self.name,
            label=EvaluationLabel.CORRECT if result else EvaluationLabel.INCORRECT,
            original_question=answer.original_question,
            paraphrased_question=answer.paraphrased_question,
            answer_text=answer.text,
            ground_truth=ground_truth.ground_truth,
        )


def load_from_file(path: str) -> FunctionEvaluator:
    """
    Load a FunctionEvaluator from a Python file.

    Required: evaluate_row(answer, ground_truth, row) -> bool
    Optional: evaluate_batch(answers, ground_truths, rows) -> list[bool]
    """
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Evaluator file not found: {path!r}")

    spec = importlib.util.spec_from_file_location("_user_evaluator", resolved)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "evaluate_row"):
        raise AttributeError(
            f"{path!r} must define: evaluate_row(answer: str, ground_truth: str, row: dict) -> bool"
        )

    batch_fn = getattr(module, "evaluate_batch", None)
    if batch_fn is not None:
        print(f"[FunctionEvaluator] Found evaluate_batch — will use batched evaluation.")
    else:
        print(f"[FunctionEvaluator] No evaluate_batch found — falling back to sequential evaluate_row.")

    return FunctionEvaluator(fn=module.evaluate_row, batch_fn=batch_fn, name=resolved.stem)


def dry_run(
    evaluator: FunctionEvaluator,
    answers_df,
    questions: dict,
    n: int = 5,
) -> None:
    """
    Run the evaluator on n random rows and print results.
    Tests both evaluate_single and evaluate_batch (if defined).
    Raises immediately on wrong return type or any exception.
    """
    from access_eval.data.schema import Answer as _Answer

    sample = answers_df.sample(min(n, len(answers_df)), random_state=42)
    sample_answers = [_Answer.from_series(row) for _, row in sample.iterrows()]
    sample_gts = [questions.get(a.question_idx) for a in sample_answers]
    pairs = [(a, gt) for a, gt in zip(sample_answers, sample_gts) if gt is not None]

    if not pairs:
        print("[dry_run] warning: no matching ground truths found in sample — skipping")
        return

    print(f"\n[dry_run] Testing '{evaluator.name}' on {len(pairs)} rows ...")

    # Test evaluate_single (always)
    for ans, gt in pairs:
        result = evaluator.evaluate_single(ans, gt)
        print(
            f"  idx={ans.question_idx}"
            f" | answer={ans.text[:50]!r}"
            f" | ground_truth={str(gt.ground_truth)[:35]!r}"
            f" | verdict={result.label.value}"
        )

    # Also test evaluate_batch if the user defined it
    if evaluator.batch_fn is not None:
        print(f"[dry_run] Testing evaluate_batch with {len(pairs)} rows ...")
        batch_ans = [a for a, _ in pairs]
        batch_gts = [gt for _, gt in pairs]
        batch_results = evaluator.evaluate_batch(batch_ans, batch_gts)
        single_labels = [evaluator.evaluate_single(a, gt).label for a, gt in pairs]
        batch_labels = [r.label for r in batch_results]
        mismatches = sum(s != b for s, b in zip(single_labels, batch_labels))
        if mismatches:
            print(f"  [dry_run] warning: {mismatches}/{len(pairs)} rows differ between "
                  f"evaluate_row and evaluate_batch — verify your batch logic.")
        else:
            print(f"  [dry_run] evaluate_batch matches evaluate_row on all {len(pairs)} rows.")

    print("[dry_run] passed\n")
