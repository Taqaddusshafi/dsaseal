"""
Unit tests for the code execution and test-case scoring in the DSA Evaluator.

Tests the new scoring features:
  - _extract_code: code extraction from markdown/raw text
  - _score_code_execution: syntax + runtime checking
  - _score_code_with_tests: actual correctness verification
  - _score_code: integrated scoring with all dimensions
"""

import pytest

from seal_dsa.modules.evaluator import DSAEvaluator
from seal_dsa.modules.question_generator import GeneratedQuestion
from seal_dsa.modules.answer_generator import GeneratedAnswer
from seal_dsa.config import SEALDSAConfig


@pytest.fixture
def evaluator():
    config = SEALDSAConfig()
    return DSAEvaluator(config)


@pytest.fixture
def coding_question():
    return GeneratedQuestion(
        question="Write a function to find the maximum element in an array.",
        topic="arrays_strings",
        subtopic="arrays",
        difficulty="easy",
        question_type="coding",
        expected_concepts=["array", "max", "iterate"],
    )


def make_answer(question, text, confidence=0.7):
    return GeneratedAnswer(
        question=question,
        answer=text,
        confidence=confidence,
        generation_tokens=len(text.split()),
    )


# ── Test Code Extraction ─────────────────────────────────────────────

class TestCodeExtraction:
    """Test _extract_code method."""

    def test_extract_from_python_block(self, evaluator):
        """Should extract code from ```python ... ``` blocks."""
        answer = '''Here is the solution:
```python
def foo(x):
    return x + 1
```
'''
        code = evaluator._extract_code(answer.lower())
        assert code is not None
        assert "def foo" in code

    def test_extract_from_generic_block(self, evaluator):
        """Should extract from ``` ... ``` blocks when no language tag."""
        answer = '''Solution:
```
def bar(x):
    return x * 2
```
'''
        code = evaluator._extract_code(answer.lower())
        assert code is not None
        assert "def bar" in code

    def test_extract_raw_function(self, evaluator):
        """Should extract raw function definitions."""
        answer = "def solution(arr):\n    return max(arr)"
        code = evaluator._extract_code(answer.lower())
        assert code is not None
        assert "def solution" in code

    def test_no_code_returns_none(self, evaluator):
        """Should return None when no code is found."""
        answer = "the answer is to iterate through the array and find the max"
        code = evaluator._extract_code(answer.lower())
        assert code is None


# ── Test Code Execution Scoring ──────────────────────────────────────

class TestCodeExecution:
    """Test _score_code_execution method."""

    def test_valid_code_runs(self, evaluator):
        """Code that runs without errors should score 1.0."""
        answer = '''```python
def add(a, b):
    return a + b
```'''
        score = evaluator._score_code_execution(answer.lower())
        assert score == 1.0

    def test_syntax_error_scores_zero(self, evaluator):
        """Code with syntax errors should score 0.0."""
        answer = '''```python
def broken(
    return ++x
```'''
        score = evaluator._score_code_execution(answer.lower())
        assert score == 0.0

    def test_runtime_error_scores_half(self, evaluator):
        """Code that parses but fails at runtime should score 0.5."""
        answer = '''```python
x = 1 / 0
```'''
        score = evaluator._score_code_execution(answer.lower())
        assert score == 0.5

    def test_no_code_scores_zero(self, evaluator):
        """No code in the answer should score 0.0."""
        answer = "just use a loop to find the maximum"
        score = evaluator._score_code_execution(answer.lower())
        assert score == 0.0

    def test_sandbox_blocks_imports(self, evaluator):
        """Sandboxed execution should not allow dangerous imports."""
        answer = '''```python
import os
os.system("echo hacked")
```'''
        # Should fail because import is not in safe builtins
        score = evaluator._score_code_execution(answer.lower())
        assert score <= 0.5  # Either syntax issue or runtime error


# ── Test Code With Test Cases ─────────────────────────────────────────

class TestCodeWithTests:
    """Test _score_code_with_tests method."""

    def test_correct_code_passes_tests(self, evaluator):
        """Code producing correct results should score 1.0."""
        answer = '''```python
def find_max(arr):
    return max(arr)
```'''
        test_cases = [
            {"function": "find_max", "input": "([1, 3, 2],)", "expected": 3},
            {"function": "find_max", "input": "([5],)", "expected": 5},
            {"function": "find_max", "input": "([-1, -5, -2],)", "expected": -1},
        ]
        score = evaluator._score_code_with_tests(answer.lower(), test_cases)
        assert score == 1.0

    def test_buggy_code_fails_tests(self, evaluator):
        """Buggy code that runs but gives wrong results should score < 1.0."""
        answer = '''```python
def find_max(arr):
    return arr[0]
```'''
        test_cases = [
            {"function": "find_max", "input": "([1, 3, 2],)", "expected": 3},
            {"function": "find_max", "input": "([5],)", "expected": 5},
            {"function": "find_max", "input": "([-1, -5, -2],)", "expected": -1},
        ]
        score = evaluator._score_code_with_tests(answer.lower(), test_cases)
        # Only the second case (single element) would pass
        assert 0.0 < score < 1.0

    def test_no_test_cases_returns_negative(self, evaluator):
        """No test cases should return -1.0 (sentinel)."""
        answer = '''```python
def foo(): pass
```'''
        score = evaluator._score_code_with_tests(answer.lower(), [])
        assert score == -1.0

    def test_no_code_with_tests_scores_zero(self, evaluator):
        """No extractable code should score 0.0 even with test cases."""
        answer = "just iterate through the array"
        test_cases = [
            {"function": "find_max", "input": "([1, 2, 3],)", "expected": 3},
        ]
        score = evaluator._score_code_with_tests(answer.lower(), test_cases)
        assert score == 0.0

    def test_auto_detect_function_name(self, evaluator):
        """Should auto-detect function name if not specified in test case."""
        answer = '''```python
def my_sum(arr):
    total = 0
    for x in arr:
        total += x
    return total
```'''
        test_cases = [
            {"function": "", "input": "([1, 2, 3],)", "expected": 6},
            {"function": "", "input": "([],)", "expected": 0},
        ]
        score = evaluator._score_code_with_tests(answer.lower(), test_cases)
        assert score == 1.0


# ── Test Integrated _score_code ──────────────────────────────────────

class TestIntegratedCodeScoring:
    """Test the full _score_code method with all dimensions."""

    def test_good_code_with_tests_scores_high(self, evaluator):
        """Clean, correct code with passing tests should score high."""
        answer = '''```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```'''
        test_cases = [
            {"function": "binary_search", "input": "([1, 3, 5, 7, 9], 5)", "expected": 2},
            {"function": "binary_search", "input": "([1, 3, 5, 7, 9], 4)", "expected": -1},
        ]
        score = evaluator._score_code(answer.lower(), "coding", test_cases)
        assert score > 0.7

    def test_non_coding_question_gets_neutral(self, evaluator):
        """Non-coding questions should get 0.7 regardless."""
        score = evaluator._score_code("some answer", "conceptual")
        assert score == 0.7

    def test_code_without_tests_uses_fallback(self, evaluator):
        """Without test cases, should use structural + execution scoring."""
        answer = '''```python
def foo(x):
    if x > 0:
        return x
    return -x
```'''
        score_no_tests = evaluator._score_code(answer.lower(), "coding")
        score_with_tests = evaluator._score_code(
            answer.lower(), "coding",
            [{"function": "foo", "input": "(-5,)", "expected": 5}],
        )
        # Both should produce reasonable scores
        assert 0.0 <= score_no_tests <= 1.0
        assert 0.0 <= score_with_tests <= 1.0

    def test_score_always_in_range(self, evaluator):
        """Score should always be between 0 and 1."""
        test_texts = [
            "", "no code here", "def broken(: syntax error",
            "```python\nx = 1\n```",
            "```python\ndef f(): return 42\n```",
        ]
        for text in test_texts:
            score = evaluator._score_code(text.lower(), "coding")
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for: '{text[:50]}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
