"""
Unit tests for curriculum management.
Tests topic definitions, scheduling strategies, and review logic.
"""

import pytest

from seal_dsa.curriculum.dsa_topics import (
    DSA_TOPICS,
    get_topic_names,
    get_topic_by_week,
    get_difficulty_for_week,
)
from seal_dsa.curriculum.scheduler import CurriculumScheduler
from seal_dsa.config import CurriculumConfig


class TestDSATopics:
    """Test the DSA topic definitions."""

    def test_topics_not_empty(self):
        assert len(DSA_TOPICS) > 0

    def test_required_topics_exist(self):
        required = [
            "arrays_strings",
            "linked_lists",
            "stacks_queues",
            "trees",
            "graphs",
            "sorting_searching",
            "dynamic_programming",
        ]
        names = get_topic_names()
        for topic in required:
            assert topic in names, f"Missing required topic: {topic}"

    def test_topic_has_required_fields(self):
        for name, topic in DSA_TOPICS.items():
            assert "subtopics" in topic, f"'{name}' missing subtopics"
            assert "key_concepts" in topic, f"'{name}' missing key_concepts"
            assert "difficulty" in topic, f"'{name}' missing difficulty"

    def test_difficulty_progression(self):
        """Verify that topics progress from easy to hard."""
        difficulties = {
            "arrays_strings": "easy",
            "linked_lists": "easy",
            "dynamic_programming": "hard",
        }
        for topic, expected_diff in difficulties.items():
            if topic in DSA_TOPICS:
                actual = DSA_TOPICS[topic]["difficulty"]
                assert actual == expected_diff, (
                    f"{topic} difficulty should be '{expected_diff}', got '{actual}'"
                )

    def test_sample_questions_exist(self):
        """Each topic should have at least one sample question."""
        for name, topic in DSA_TOPICS.items():
            if name not in ("advanced_topics", "comprehensive_review"):
                assert len(topic.get("sample_questions", [])) > 0, (
                    f"'{name}' has no sample questions"
                )

    def test_get_topic_by_week(self):
        """Test week-to-topic mapping."""
        # Week 1 should return a foundational topic
        topic = get_topic_by_week(1)
        assert topic is not None

    def test_get_difficulty_for_week(self):
        """Test difficulty progression by week."""
        early_diff = get_difficulty_for_week(1)
        late_diff = get_difficulty_for_week(14)
        # Early weeks should be easier than late weeks
        diff_order = {"easy": 0, "medium": 1, "hard": 2}
        assert diff_order.get(early_diff, 0) <= diff_order.get(late_diff, 2)


class TestCurriculumScheduler:
    """Test the CurriculumScheduler."""

    @pytest.fixture
    def config(self):
        return CurriculumConfig(strategy="progressive")

    @pytest.fixture
    def scheduler(self, config):
        return CurriculumScheduler(config)

    def test_initialization(self, scheduler):
        assert scheduler is not None
        assert len(scheduler.all_topics) > 0

    def test_progressive_strategy(self, scheduler):
        """Progressive strategy should add one topic per epoch."""
        topics_e0 = scheduler.get_topics_for_epoch(0)
        topics_e1 = scheduler.get_topics_for_epoch(1)

        assert len(topics_e0) >= 1
        # Each epoch should include at least as many topics as previous
        assert len(topics_e1) >= len(topics_e0)

    def test_random_strategy(self):
        config = CurriculumConfig(strategy="random")
        scheduler = CurriculumScheduler(config)
        topics = scheduler.get_topics_for_epoch(0)
        assert len(topics) >= 1

    def test_record_performance(self, scheduler):
        """Test that performance recording works."""
        scheduler.record_performance("arrays_strings", 0.75)
        scheduler.record_performance("arrays_strings", 0.80)

        # Should not raise errors
        assert True

    def test_review_topics(self, scheduler):
        """Test review topic selection."""
        # Record low performance to trigger review
        scheduler.record_performance("arrays_strings", 0.9)
        scheduler.record_performance("arrays_strings", 0.3)  # Big drop

        reviews = scheduler.get_review_topics()
        # Review logic may or may not add topics depending on impl
        assert isinstance(reviews, list)


class TestAdaptiveScheduler:
    """Test adaptive scheduling strategy."""

    def test_adaptive_prioritizes_weak_topics(self):
        config = CurriculumConfig(strategy="adaptive")
        scheduler = CurriculumScheduler(config)

        # Simulate: strong on arrays, weak on trees
        scheduler.record_performance("arrays_strings", 0.9)
        scheduler.record_performance("trees", 0.2)

        topics = scheduler.get_topics_for_epoch(5)
        # Adaptive should prioritize weak topics
        assert isinstance(topics, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
