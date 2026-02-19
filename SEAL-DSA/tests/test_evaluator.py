"""
Unit tests for the rule-based DSA Evaluator.
Tests scoring dimensions, edge cases, and evaluation consistency.
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
def sample_question():
    return GeneratedQuestion(
        question="What is the time complexity of binary search?",
        topic="sorting_searching",
        subtopic="binary_search",
        difficulty="easy",
        question_type="conceptual",
        expected_concepts=["binary search", "O(log n)", "sorted"],
    )


def make_answer(question, text, confidence=0.7):
    return GeneratedAnswer(
        question=question,
        answer=text,
        confidence=confidence,
        generation_tokens=len(text.split()),
    )


class TestEvaluatorScoring:
    """Test individual scoring dimensions."""

    def test_high_quality_answer(self, evaluator, sample_question):
        """A comprehensive correct answer should score high."""
        answer = make_answer(
            sample_question,
            """Binary search has a time complexity of O(log n) where n is the 
            number of elements in the sorted array. It works by repeatedly 
            dividing the search interval in half. At each step, we compare 
            the target with the middle element. If equal, we found it. If 
            the target is smaller, we search the left half. If larger, we 
            search the right half. The space complexity is O(1) for the 
            iterative version and O(log n) for the recursive version due 
            to the call stack.
            
            ```python
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
            ```"""
        )

        result = evaluator.evaluate(answer)
        assert result.overall_score > 0.6, (
            f"High-quality answer scored too low: {result.overall_score}"
        )

    def test_empty_answer_scores_low(self, evaluator, sample_question):
        """An empty answer should score very low."""
        answer = make_answer(sample_question, "")
        result = evaluator.evaluate(answer)
        assert result.overall_score < 0.2

    def test_irrelevant_answer_scores_low(self, evaluator, sample_question):
        """A completely irrelevant answer should score low."""
        answer = make_answer(
            sample_question,
            "The weather is nice today. I like pizza and football."
        )
        result = evaluator.evaluate(answer)
        assert result.overall_score < 0.3

    def test_partial_answer(self, evaluator, sample_question):
        """A partial but correct answer should score moderately."""
        answer = make_answer(
            sample_question,
            "Binary search has O(log n) time complexity because it divides "
            "the array in half each time."
        )
        result = evaluator.evaluate(answer)
        assert 0.2 < result.overall_score < 0.8

    def test_answer_with_complexity_analysis(self, evaluator, sample_question):
        """Presence of Big-O should boost the complexity dimension."""
        with_complexity = make_answer(
            sample_question,
            "The time complexity is O(log n) and space complexity is O(1)."
        )
        without_complexity = make_answer(
            sample_question,
            "It searches by dividing the array in half repeatedly."
        )

        score_with = evaluator.evaluate(with_complexity).overall_score
        score_without = evaluator.evaluate(without_complexity).overall_score

        assert score_with > score_without


class TestEvaluatorBatch:
    """Test batch evaluation."""

    def test_batch_evaluation(self, evaluator, sample_question):
        """Batch evaluation should process all answers."""
        answers = [
            make_answer(sample_question, "O(log n) binary search sorted array"),
            make_answer(sample_question, "I don't know"),
            make_answer(sample_question, "Binary search is O(log n)."),
        ]

        results = evaluator.evaluate_batch(answers)
        assert len(results) == 3

    def test_batch_preserves_order(self, evaluator, sample_question):
        """Results should correspond to input order."""
        good_answer = make_answer(
            sample_question,
            "Binary search O(log n) sorted array divide and conquer"
        )
        bad_answer = make_answer(sample_question, "")

        results = evaluator.evaluate_batch([good_answer, bad_answer])
        assert results[0].overall_score > results[1].overall_score


class TestEvaluatorEdgeCases:
    """Test edge cases and robustness."""

    def test_very_long_answer(self, evaluator, sample_question):
        """Evaluator should handle very long answers."""
        long_text = "Binary search O(log n) sorted. " * 500
        answer = make_answer(sample_question, long_text)
        result = evaluator.evaluate(answer)
        assert 0.0 <= result.overall_score <= 1.0

    def test_special_characters(self, evaluator, sample_question):
        """Evaluator should handle special characters."""
        answer = make_answer(
            sample_question,
            "O(log n) — θ(log₂n) ≤ O(n) ∈ Ω(1) → binary search 🔍"
        )
        result = evaluator.evaluate(answer)
        assert 0.0 <= result.overall_score <= 1.0

    def test_code_only_answer(self, evaluator, sample_question):
        answer = make_answer(
            sample_question,
            """```python
            def binary_search(arr, target):
                lo, hi = 0, len(arr) - 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if arr[mid] == target: return mid
                    elif arr[mid] < target: lo = mid + 1
                    else: hi = mid - 1
                return -1
            ```"""
        )
        result = evaluator.evaluate(answer)
        # Code-only answer should still get a reasonable score
        assert result.overall_score > 0.1

    def test_score_always_in_range(self, evaluator, sample_question):
        """Score should always be between 0 and 1."""
        test_texts = [
            "", " ", "a", "x" * 10000,
            "O(n) O(log n) O(1) O(n²) O(n log n)",
            "binary search linked list hash map tree graph",
        ]
        for text in test_texts:
            answer = make_answer(sample_question, text)
            result = evaluator.evaluate(answer)
            assert 0.0 <= result.overall_score <= 1.0, (
                f"Score {result.overall_score} out of range for: '{text[:50]}'"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
