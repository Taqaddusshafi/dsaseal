"""
Unit tests for the MetricsTracker.
Tests recording, summaries, and learning curves.
"""

import os
import json
import tempfile
import pytest

from seal_dsa.evaluation.metrics import MetricsTracker


@pytest.fixture
def tracker():
    return MetricsTracker()


@pytest.fixture
def populated_tracker():
    t = MetricsTracker()
    # Simulate 3 epochs, 2 topics each
    for epoch in range(3):
        for topic in ["arrays_strings", "linked_lists"]:
            score = 0.3 + epoch * 0.1 + (0.05 if topic == "arrays_strings" else 0)
            t.record(
                epoch=epoch,
                topic=topic,
                avg_score=score,
                correct_ratio=score * 0.8,
                loss=1.0 - score,
                grad_norm=0.5,
                lr=2e-4 / (epoch + 1),
            )
    return t


class TestMetricsRecording:
    """Test basic recording."""

    def test_record_single_entry(self, tracker):
        tracker.record(
            epoch=0, topic="arrays_strings",
            avg_score=0.5, correct_ratio=0.4,
            loss=0.8, grad_norm=0.3, lr=2e-4,
        )
        assert len(tracker.records) == 1

    def test_record_multiple_entries(self, tracker):
        for i in range(10):
            tracker.record(
                epoch=i // 3, topic=f"topic_{i % 3}",
                avg_score=0.5, correct_ratio=0.4,
                loss=0.8,
            )
        assert len(tracker.records) == 10

    def test_records_have_timestamps(self, tracker):
        tracker.record(
            epoch=0, topic="test",
            avg_score=0.5, correct_ratio=0.4, loss=0.8,
        )
        assert "timestamp" in tracker.records[0]


class TestMetricsSummary:
    """Test summary generation."""

    def test_empty_summary(self, tracker):
        summary = tracker.get_summary()
        assert summary.get("status") == "no data"

    def test_populated_summary(self, populated_tracker):
        summary = populated_tracker.get_summary()
        assert summary["total_steps"] == 6
        assert summary["total_epochs"] == 3
        assert summary["topics_trained"] == 2
        assert 0.0 <= summary["avg_score"] <= 1.0

    def test_improvement_tracking(self, populated_tracker):
        summary = populated_tracker.get_summary()
        # Scores increase over epochs, so improvement should be positive
        assert summary["improvement"] > 0

    def test_topic_summary(self, populated_tracker):
        topic_summary = populated_tracker.get_topic_summary()
        assert "arrays_strings" in topic_summary
        assert "linked_lists" in topic_summary
        assert topic_summary["arrays_strings"]["num_iterations"] == 3

    def test_learning_curve(self, populated_tracker):
        curves = populated_tracker.get_learning_curve()
        assert "overall_score" in curves
        assert "overall_loss" in curves
        assert len(curves["overall_score"]) == 6


class TestMetricsPersistence:
    """Test save and load."""

    def test_save_and_load(self, populated_tracker):
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode='w'
        ) as f:
            temp_path = f.name

        try:
            populated_tracker.save(temp_path)
            assert os.path.exists(temp_path)

            # Verify JSON structure
            with open(temp_path, 'r') as f:
                data = json.load(f)
            assert "summary" in data
            assert "records" in data
            assert len(data["records"]) == 6

            # Test loading
            new_tracker = MetricsTracker()
            new_tracker.load(temp_path)
            assert len(new_tracker.records) == 6

        finally:
            os.unlink(temp_path)


class TestMetricsEdgeCases:
    """Test edge cases."""

    def test_single_record_summary(self, tracker):
        tracker.record(
            epoch=0, topic="test",
            avg_score=0.5, correct_ratio=0.4, loss=0.8,
        )
        summary = tracker.get_summary()
        assert summary["total_steps"] == 1
        assert summary["avg_score"] == 0.5

    def test_extra_kwargs_stored(self, tracker):
        tracker.record(
            epoch=0, topic="test",
            avg_score=0.5, correct_ratio=0.4, loss=0.8,
            custom_metric=42,
        )
        assert tracker.records[0]["custom_metric"] == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
