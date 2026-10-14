# ============================================================================
# File: src/access_eval/evaluators/math_answer.py
# ============================================================================
"""
Deterministic evaluator for math datasets (GSM8K, MATH-500): the paper_v1 grading policy (math_verify + fallback
extraction, evaluators/math_fallback.py), so math answers go through the same evaluation stage, resume logic and
grade files as the LLM-judged QA datasets. No judge model is used; run it with judge name "rule".

A per-answer SIGALRM bound (ROW_TIMEOUT_S) interrupts a hanging math_verify call, as the paper's grading did. A
timed-out answer is recorded as NOT_ATTEMPTED with raw_response GRADING_TIMEOUT, so analysis can exclude it (the
paper excluded such rows) instead of counting it as a wrong answer.
"""
import hashlib
import inspect
import signal
from typing import List

from access_eval.data.schema import Answer, Evaluation, EvaluationLabel, Question
from access_eval.evaluators import math_fallback
from access_eval.evaluators.base import Evaluator
from access_eval.evaluators.registry import register_evaluator

ROW_TIMEOUT_S = 5
GRADING_TIMEOUT = "GRADING_TIMEOUT"


class MathAnswerEvaluator(Evaluator):
    def __init__(self, judge_model: str = "rule", policy: str = "fallback", **kwargs):
        super().__init__(name="math_verify_fallback")
        self.policy = policy
        # Stands in for a judge prompt in evaluation.py's eval_config_hash: any edit to the grading code or policy
        # is a new config bucket, never aliased with grades produced by an older grader.
        src = inspect.getsource(math_fallback) + inspect.getsource(MathAnswerEvaluator)
        self.prompt_template = f"math_verify_fallback policy={policy} src={hashlib.sha256(src.encode()).hexdigest()[:16]}"

    def _grade(self, text: str, gold: str):
        def _alarm(*_):
            raise TimeoutError

        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(ROW_TIMEOUT_S)
        try:
            return math_fallback.grade_answer(text or "", str(gold), self.policy)
        except Exception:
            return None
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    def evaluate_single(self, answer: Answer, ground_truth: Question) -> Evaluation:
        ok = self._grade(answer.text, ground_truth.ground_truth)
        label = (EvaluationLabel.NOT_ATTEMPTED if ok is None
                 else EvaluationLabel.CORRECT if ok else EvaluationLabel.INCORRECT)
        return Evaluation(
            answer_idx=0,
            question_idx=answer.question_idx,
            model=answer.model,
            evaluator=self.name,
            label=label,
            raw_response=GRADING_TIMEOUT if ok is None else None,
            original_question=answer.original_question,
            paraphrased_question=answer.paraphrased_question,
            answer_text=answer.text,
            ground_truth=ground_truth.ground_truth,
        )

    def evaluate_batch(self, answers: List[Answer], ground_truths: List[Question]) -> List[Evaluation]:
        return [self.evaluate_single(a, g) for a, g in zip(answers, ground_truths)]


HARD_TIMEOUT_S = 60   # wall-clock bound per answer: catches hangs inside C code that SIGALRM cannot interrupt


def _grade_one(args):
    """Worker: one (text, gold) pair, with the same per-answer SIGALRM bound as MathAnswerEvaluator._grade."""
    policy, text, gold = args
    ev = MathAnswerEvaluator.__new__(MathAnswerEvaluator)
    ev.policy = policy
    return ev._grade(text, gold)


def _grade_parallel(rows, policy, n_workers, hard_timeout=HARD_TIMEOUT_S):
    """Grade rows on a fork pool. Results are consumed in order; if the next answer is not back within hard_timeout
    seconds (a hang SIGALRM cannot break), the pool is killed, that answer is recorded as None (= GRADING_TIMEOUT)
    and grading restarts from the following answer."""
    import multiprocessing as mp
    out, start = [], 0
    while start < len(rows):
        pool = mp.get_context("fork").Pool(n_workers)
        it = pool.imap(_grade_one, [(policy, t, g) for t, g in rows[start:]], chunksize=1)
        try:
            for _ in range(start, len(rows)):
                out.append(it.next(timeout=hard_timeout))
            start = len(rows)
            pool.close()
        except mp.TimeoutError:
            print(f"[math_answer] answer {len(out)} exceeded {hard_timeout}s -> GRADING_TIMEOUT; restarting the pool")
            out.append(None)
            start = len(out)
            pool.terminate()
        finally:
            pool.join()
    return out


class ParallelMathAnswerEvaluator(MathAnswerEvaluator):
    """MathAnswerEvaluator with the batch graded by worker processes (one per available core) and a hard wall-clock
    bound per answer. Same grading code, same name and the same eval_config_hash (the hash covers math_fallback +
    MathAnswerEvaluator only), so it shares the grade cache with the sequential evaluator."""

    def evaluate_batch(self, answers: List[Answer], ground_truths: List[Question]) -> List[Evaluation]:
        import os
        n = max(1, len(os.sched_getaffinity(0)))
        rows = [(a.text, g.ground_truth) for a, g in zip(answers, ground_truths)]
        if len(rows) < 2:      # a pool even with one core: it is what enforces the hard per-answer bound
            return super().evaluate_batch(answers, ground_truths)
        oks = _grade_parallel(rows, self.policy, n)
        out = []
        for a, g, ok in zip(answers, ground_truths, oks):
            label = (EvaluationLabel.NOT_ATTEMPTED if ok is None
                     else EvaluationLabel.CORRECT if ok else EvaluationLabel.INCORRECT)
            out.append(Evaluation(answer_idx=0, question_idx=a.question_idx, model=a.model, evaluator=self.name,
                                  label=label, raw_response=GRADING_TIMEOUT if ok is None else None,
                                  original_question=a.original_question, paraphrased_question=a.paraphrased_question,
                                  answer_text=a.text, ground_truth=g.ground_truth))
        return out


for _name in ("gsm8k_test", "math_500_test", "math_answer"):
    register_evaluator(_name, ParallelMathAnswerEvaluator)
