"""
Tests for FunctionEvaluator, load_from_file, dry_run, and evaluator utils.

These tests are self-contained: no LLM calls, no API keys, no GPU required.
LLM-dependent paths (call_llm_judge / call_llm_judge_batch) are tested via mocks.
"""

import textwrap
import tempfile
import os
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from access_eval.evaluators.function_evaluator import (
    FunctionEvaluator,
    load_from_file,
    dry_run,
)
from access_eval.evaluators.utils import _parse_yes_no
from access_eval.data.schema import Answer, Question, EvaluationLabel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_answer(text="Paris", idx=0, model="gpt-test"):
    return Answer(
        question_idx=idx,
        original_question="What is the capital of France?",
        paraphrased_question="What is the capital of France?",
        model=model,
        text=text,
    )


def _make_question(ground_truth="Paris", idx=0):
    return Question(
        idx=idx,
        text="What is the capital of France?",
        dataset="test",
        ground_truth=ground_truth,
        metadata={"question": "What is the capital of France?", "category": "geography"},
    )


def _write_eval_file(content: str) -> str:
    """Write content to a temp .py file and return its path."""
    f = tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False)
    f.write(textwrap.dedent(content))
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# FunctionEvaluator.evaluate_single
# ---------------------------------------------------------------------------

class TestEvaluateSingle:

    def test_correct_verdict(self):
        def fn(answer, ground_truth, row): return ground_truth.lower() in answer.lower()
        ev = FunctionEvaluator(fn=fn)
        result = ev.evaluate_single(_make_answer("Paris"), _make_question("Paris"))
        assert result.label == EvaluationLabel.CORRECT

    def test_incorrect_verdict(self):
        def fn(answer, ground_truth, row): return ground_truth.lower() in answer.lower()
        ev = FunctionEvaluator(fn=fn)
        result = ev.evaluate_single(_make_answer("London"), _make_question("Paris"))
        assert result.label == EvaluationLabel.INCORRECT

    def test_row_dict_accessible(self):
        """evaluate_row receives the full row dict from question.metadata."""
        received = {}
        def fn(answer, ground_truth, row):
            received.update(row)
            return True
        ev = FunctionEvaluator(fn=fn)
        ev.evaluate_single(_make_answer(), _make_question())
        assert "category" in received
        assert received["category"] == "geography"

    def test_non_bool_return_raises(self):
        def fn(answer, ground_truth, row): return "yes"   # str, not bool
        ev = FunctionEvaluator(fn=fn)
        with pytest.raises(TypeError, match="must return bool"):
            ev.evaluate_single(_make_answer(), _make_question())

    def test_none_return_raises(self):
        def fn(answer, ground_truth, row): return None
        ev = FunctionEvaluator(fn=fn)
        with pytest.raises(TypeError, match="must return bool"):
            ev.evaluate_single(_make_answer(), _make_question())

    def test_evaluation_fields_populated(self):
        def fn(answer, ground_truth, row): return True
        ev = FunctionEvaluator(fn=fn, name="my_eval")
        result = ev.evaluate_single(_make_answer("Paris", idx=7), _make_question("Paris", idx=7))
        assert result.question_idx == 7
        assert result.evaluator == "my_eval"
        assert result.answer_text == "Paris"
        assert result.ground_truth == "Paris"


# ---------------------------------------------------------------------------
# FunctionEvaluator.evaluate_batch
# ---------------------------------------------------------------------------

class TestEvaluateBatch:

    def test_uses_batch_fn_when_provided(self):
        def single_fn(answer, ground_truth, row): return True
        batch_calls = []
        def batch_fn(answers, ground_truths, rows):
            batch_calls.append(len(answers))
            return [True] * len(answers)
        ev = FunctionEvaluator(fn=single_fn, batch_fn=batch_fn)
        answers   = [_make_answer(idx=i) for i in range(3)]
        questions = [_make_question(idx=i) for i in range(3)]
        results = ev.evaluate_batch(answers, questions)
        assert len(results) == 3
        assert batch_calls == [3]   # batch_fn was called once with all 3

    def test_falls_back_to_single_without_batch_fn(self):
        call_count = [0]
        def fn(answer, ground_truth, row):
            call_count[0] += 1
            return True
        ev = FunctionEvaluator(fn=fn, batch_fn=None)
        answers   = [_make_answer(idx=i) for i in range(4)]
        questions = [_make_question(idx=i) for i in range(4)]
        results = ev.evaluate_batch(answers, questions)
        assert len(results) == 4
        assert call_count[0] == 4  # called once per row

    def test_batch_wrong_length_raises(self):
        def fn(answer, ground_truth, row): return True
        def batch_fn(answers, ground_truths, rows): return [True]  # 1 item for N inputs
        ev = FunctionEvaluator(fn=fn, batch_fn=batch_fn)
        answers   = [_make_answer(idx=i) for i in range(3)]
        questions = [_make_question(idx=i) for i in range(3)]
        with pytest.raises(ValueError, match="returned 1 results for 3 inputs"):
            ev.evaluate_batch(answers, questions)

    def test_batch_non_bool_raises(self):
        def fn(answer, ground_truth, row): return True
        def batch_fn(answers, ground_truths, rows): return ["yes"] * len(answers)
        ev = FunctionEvaluator(fn=fn, batch_fn=batch_fn)
        answers   = [_make_answer(idx=i) for i in range(2)]
        questions = [_make_question(idx=i) for i in range(2)]
        with pytest.raises(TypeError, match=r"must return list\[bool\]"):
            ev.evaluate_batch(answers, questions)


# ---------------------------------------------------------------------------
# load_from_file
# ---------------------------------------------------------------------------

class TestLoadFromFile:

    def test_loads_evaluate_row(self):
        path = _write_eval_file("""
            def evaluate_row(answer, ground_truth, row):
                return answer == ground_truth
        """)
        try:
            ev = load_from_file(path)
            assert isinstance(ev, FunctionEvaluator)
            assert ev.batch_fn is None
        finally:
            os.unlink(path)

    def test_loads_both_functions(self):
        path = _write_eval_file("""
            def evaluate_row(answer, ground_truth, row):
                return True
            def evaluate_batch(answers, ground_truths, rows):
                return [True] * len(answers)
        """)
        try:
            ev = load_from_file(path)
            assert ev.batch_fn is not None
        finally:
            os.unlink(path)

    def test_name_is_file_stem(self):
        path = _write_eval_file("def evaluate_row(a, g, r): return True")
        try:
            ev = load_from_file(path)
            assert ev.name == os.path.splitext(os.path.basename(path))[0]
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_from_file("/nonexistent/path/my_eval.py")

    def test_missing_evaluate_row_raises(self):
        path = _write_eval_file("x = 1  # no evaluate_row defined")
        try:
            with pytest.raises(AttributeError, match="must define"):
                load_from_file(path)
        finally:
            os.unlink(path)

    def test_evaluate_row_actually_works(self):
        path = _write_eval_file("""
            def evaluate_row(answer, ground_truth, row):
                return ground_truth.lower() in answer.lower()
        """)
        try:
            ev = load_from_file(path)
            assert ev.evaluate_single(_make_answer("Paris"), _make_question("paris")).label \
                   == EvaluationLabel.CORRECT
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# dry_run
# ---------------------------------------------------------------------------

class TestDryRun:

    def _make_answers_df(self, n=5):
        rows = []
        for i in range(n):
            rows.append({
                "idx": i,
                "original_question": "What is the capital?",
                "paraphrased_question": "What is the capital?",
                "model": "gpt-test",
                "answer": "Paris" if i % 2 == 0 else "London",
            })
        return pd.DataFrame(rows)

    def _make_questions(self, n=5):
        return {i: _make_question(ground_truth="Paris", idx=i) for i in range(n)}

    def test_passes_on_valid_evaluator(self):
        def fn(answer, ground_truth, row): return True
        ev = FunctionEvaluator(fn=fn)
        dry_run(ev, self._make_answers_df(), self._make_questions())   # should not raise

    def test_raises_on_wrong_return_type(self):
        def fn(answer, ground_truth, row): return "yes"   # wrong type
        ev = FunctionEvaluator(fn=fn)
        with pytest.raises(TypeError):
            dry_run(ev, self._make_answers_df(), self._make_questions())

    def test_runs_batch_check_when_batch_fn_present(self, capsys):
        def fn(answer, ground_truth, row): return True
        def batch_fn(answers, ground_truths, rows): return [True] * len(answers)
        ev = FunctionEvaluator(fn=fn, batch_fn=batch_fn)
        dry_run(ev, self._make_answers_df(), self._make_questions())
        captured = capsys.readouterr()
        assert "evaluate_batch matches" in captured.out

    def test_warns_on_batch_single_mismatch(self, capsys):
        def fn(answer, ground_truth, row): return True
        def batch_fn(answers, ground_truths, rows): return [False] * len(answers)
        ev = FunctionEvaluator(fn=fn, batch_fn=batch_fn)
        dry_run(ev, self._make_answers_df(), self._make_questions())
        captured = capsys.readouterr()
        assert "differ between" in captured.out   # warning printed, not raised


# ---------------------------------------------------------------------------
# _parse_yes_no
# ---------------------------------------------------------------------------

class TestParseYesNo:

    @pytest.mark.parametrize("response,expected", [
        ("YES",        True),
        ("Yes",        True),
        ("yes",        True),
        ("YES.",       True),
        ("YES, it is", True),
        ("NO",         False),
        ("No",         False),
        ("no.",        False),
        ("NOT correct",False),
        ("",           False),
        ("MAYBE",      False),
        # startswith("YES") short-circuits — practical LLM responses always
        # lead with their verdict, so "YES but also NO" is treated as YES.
        ("YES but also NO", True),
    ])
    def test_parse(self, response, expected):
        assert _parse_yes_no(response) == expected


# ---------------------------------------------------------------------------
# call_llm_judge / call_llm_judge_batch — mocked LLM
# ---------------------------------------------------------------------------

class TestCallLlmJudge:

    def test_raises_when_judge_not_set(self):
        import access_eval.evaluators.utils as utils_mod
        original = utils_mod._judge_llm
        utils_mod._judge_llm = None
        try:
            from access_eval.evaluators.utils import call_llm_judge
            with pytest.raises(RuntimeError, match="No judge LLM"):
                call_llm_judge("Is {answer} correct? YES/NO", answer="Paris")
        finally:
            utils_mod._judge_llm = original

    def test_correct_when_judge_says_yes(self):
        import access_eval.evaluators.utils as utils_mod
        mock_llm = MagicMock()
        mock_llm.single.return_value = "YES"
        utils_mod._judge_llm = mock_llm
        try:
            from access_eval.evaluators.utils import call_llm_judge
            result = call_llm_judge("Answer: {answer} correct?", answer="Paris")
            assert result is True
            mock_llm.single.assert_called_once()
        finally:
            utils_mod._judge_llm = None

    def test_incorrect_when_judge_says_no(self):
        import access_eval.evaluators.utils as utils_mod
        mock_llm = MagicMock()
        mock_llm.single.return_value = "NO"
        utils_mod._judge_llm = mock_llm
        try:
            from access_eval.evaluators.utils import call_llm_judge
            result = call_llm_judge("Answer: {answer} correct?", answer="London")
            assert result is False
        finally:
            utils_mod._judge_llm = None

    def test_batch_raises_when_judge_not_set(self):
        import access_eval.evaluators.utils as utils_mod
        utils_mod._judge_llm = None
        from access_eval.evaluators.utils import call_llm_judge_batch
        with pytest.raises(RuntimeError, match="No judge LLM"):
            call_llm_judge_batch("tmpl", ["a"], ["g"], [{}])

    def test_batch_length_mismatch_raises(self):
        import access_eval.evaluators.utils as utils_mod
        utils_mod._judge_llm = MagicMock()
        try:
            from access_eval.evaluators.utils import call_llm_judge_batch
            with pytest.raises(ValueError, match="same length"):
                call_llm_judge_batch("tmpl", ["a", "b"], ["g"], [{}, {}])
        finally:
            utils_mod._judge_llm = None

    def test_batch_returns_correct_bools(self):
        import access_eval.evaluators.utils as utils_mod
        mock_llm = MagicMock()
        mock_llm.batch.return_value = ["YES", "NO", "YES"]
        utils_mod._judge_llm = mock_llm
        try:
            from access_eval.evaluators.utils import call_llm_judge_batch
            results = call_llm_judge_batch(
                "Answer: {answer}, GT: {ground_truth}. Correct?",
                answers=["Paris", "London", "Berlin"],
                ground_truths=["Paris", "Paris", "Berlin"],
                rows=[{}, {}, {}],
            )
            assert results == [True, False, True]
        finally:
            utils_mod._judge_llm = None

    def test_batch_template_fills_row_values(self):
        """Extra row columns are injected into the template."""
        import access_eval.evaluators.utils as utils_mod
        mock_llm = MagicMock()
        mock_llm.batch.return_value = ["YES"]
        utils_mod._judge_llm = mock_llm
        try:
            from access_eval.evaluators.utils import call_llm_judge_batch
            call_llm_judge_batch(
                "Category: {category}. Answer: {answer}. GT: {ground_truth}.",
                answers=["Paris"],
                ground_truths=["Paris"],
                rows=[{"category": "geography"}],
            )
            prompt_used = mock_llm.batch.call_args[0][0][0]
            assert "geography" in prompt_used
        finally:
            utils_mod._judge_llm = None
