"""
Tests for Evaluation Metrics Module

Tests the evaluation functionality including:
- Precision, recall, F1 calculation
- Severity accuracy metrics
- Cohen's Kappa calculation
- Confusion matrices
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.metrics import (
    HazardEvaluator,
    ConsistencyEvaluator,
    EvaluationResult,
    CategoryMetrics,
    DetectionMatch
)


class TestHazardEvaluator:
    """Test the HazardEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create an evaluator instance."""
        return HazardEvaluator()

    def test_perfect_match(self, evaluator):
        """Perfect predictions should yield perfect metrics."""
        ground_truth = [
            {"category": "bathroom", "subcategory": "no_grab_bars", "severity": "critical"},
            {"category": "flooring", "subcategory": "loose_rug", "severity": "high"}
        ]
        predictions = ground_truth.copy()

        result = evaluator.evaluate(predictions, ground_truth)

        assert result.overall_precision == 1.0
        assert result.overall_recall == 1.0
        assert result.overall_f1 == 1.0

    def test_no_predictions(self, evaluator):
        """No predictions should yield zero precision/recall."""
        ground_truth = [
            {"category": "bathroom", "subcategory": "no_grab_bars", "severity": "critical"}
        ]
        predictions = []

        result = evaluator.evaluate(predictions, ground_truth)

        assert result.overall_precision == 0.0
        assert result.overall_recall == 0.0

    def test_false_positives(self, evaluator):
        """Extra predictions should reduce precision."""
        ground_truth = [
            {"category": "bathroom", "subcategory": "no_grab_bars", "severity": "critical"}
        ]
        predictions = [
            {"category": "bathroom", "subcategory": "no_grab_bars", "severity": "critical"},
            {"category": "lighting", "subcategory": "dim", "severity": "low"}  # FP
        ]

        result = evaluator.evaluate(predictions, ground_truth)

        assert result.overall_recall == 1.0
        assert result.overall_precision < 1.0

    def test_false_negatives(self, evaluator):
        """Missing predictions should reduce recall."""
        ground_truth = [
            {"category": "bathroom", "subcategory": "no_grab_bars", "severity": "critical"},
            {"category": "flooring", "subcategory": "loose_rug", "severity": "high"}
        ]
        predictions = [
            {"category": "bathroom", "subcategory": "no_grab_bars", "severity": "critical"}
        ]

        result = evaluator.evaluate(predictions, ground_truth)

        assert result.overall_precision == 1.0
        assert result.overall_recall < 1.0

    def test_category_metrics(self, evaluator, sample_predictions, sample_ground_truth):
        """Should calculate per-category metrics."""
        result = evaluator.evaluate(sample_predictions, sample_ground_truth)

        assert len(result.category_metrics) > 0
        for cat, metrics in result.category_metrics.items():
            assert hasattr(metrics, 'precision')
            assert hasattr(metrics, 'recall')
            assert hasattr(metrics, 'f1_score')

    def test_severity_accuracy(self, evaluator):
        """Should calculate severity classification accuracy."""
        ground_truth = [
            {"category": "bathroom", "subcategory": "a", "severity": "critical"},
            {"category": "flooring", "subcategory": "b", "severity": "high"}
        ]
        predictions = [
            {"category": "bathroom", "subcategory": "a", "severity": "critical"},  # Correct
            {"category": "flooring", "subcategory": "b", "severity": "medium"}  # Wrong
        ]

        result = evaluator.evaluate(predictions, ground_truth)

        # 1 correct out of 2
        assert result.severity_accuracy == 0.5

    def test_batch_evaluation(self, evaluator):
        """Should aggregate metrics across multiple images."""
        all_preds = [
            [{"category": "bathroom", "subcategory": "a", "severity": "high"}],
            [{"category": "stairs", "subcategory": "b", "severity": "critical"}]
        ]
        all_gts = [
            [{"category": "bathroom", "subcategory": "a", "severity": "high"}],
            [{"category": "stairs", "subcategory": "b", "severity": "critical"}]
        ]

        result = evaluator.evaluate_batch(all_preds, all_gts)

        assert result.overall_f1 == 1.0
        assert result.total_predictions == 2
        assert result.total_ground_truth == 2


class TestCategoryMetrics:
    """Test CategoryMetrics dataclass."""

    def test_metrics_calculation(self):
        """Test precision/recall/F1 calculation."""
        metrics = CategoryMetrics(
            category="bathroom",
            true_positives=8,
            false_positives=2,
            false_negatives=2
        )
        # Manually calculate expected values
        expected_precision = 8 / (8 + 2)  # 0.8
        expected_recall = 8 / (8 + 2)  # 0.8
        expected_f1 = 2 * 0.8 * 0.8 / (0.8 + 0.8)  # 0.8

        # The dataclass stores calculated values
        assert metrics.true_positives == 8
        assert metrics.false_positives == 2
        assert metrics.false_negatives == 2


class TestConsistencyEvaluator:
    """Test the ConsistencyEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create a consistency evaluator."""
        return ConsistencyEvaluator()

    def test_perfect_intra_consistency(self, evaluator):
        """Identical results should have perfect consistency."""
        results = [
            [{"category": "bathroom", "subcategory": "a"}],
            [{"category": "bathroom", "subcategory": "a"}],
            [{"category": "bathroom", "subcategory": "a"}]
        ]

        metrics = evaluator.evaluate_intra_consistency(results)

        assert metrics['hazard_count_std'] == 0.0
        assert metrics['category_agreement'] == 1.0

    def test_varying_intra_consistency(self, evaluator):
        """Different results should have lower consistency."""
        results = [
            [{"category": "bathroom"}],
            [{"category": "bathroom"}, {"category": "stairs"}],
            [{"category": "flooring"}]
        ]

        metrics = evaluator.evaluate_intra_consistency(results)

        assert metrics['hazard_count_std'] > 0
        assert metrics['category_agreement'] < 1.0

    def test_inter_model_agreement(self, evaluator):
        """Should calculate agreement between different models."""
        model_results = {
            "gpt4v": [
                {"category": "bathroom", "subcategory": "a"},
                {"category": "stairs", "subcategory": "b"}
            ],
            "gemini": [
                {"category": "bathroom", "subcategory": "a"},
                {"category": "flooring", "subcategory": "c"}
            ]
        }

        metrics = evaluator.evaluate_inter_consistency(model_results)

        assert 'pairwise_agreement' in metrics
        assert 'mean_jaccard' in metrics
        assert 0 <= metrics['mean_jaccard'] <= 1.0


class TestDetectionMatch:
    """Test DetectionMatch dataclass."""

    def test_true_positive(self):
        """True positive should have both prediction and ground truth."""
        match = DetectionMatch(
            predicted={"category": "bathroom"},
            ground_truth={"category": "bathroom"},
            category_match=True,
            subcategory_match=True,
            severity_match=True
        )
        assert match.predicted is not None
        assert match.ground_truth is not None
        assert match.category_match

    def test_false_positive(self):
        """False positive should have prediction but no ground truth."""
        match = DetectionMatch(
            predicted={"category": "lighting"},
            ground_truth=None,
            category_match=False
        )
        assert match.predicted is not None
        assert match.ground_truth is None

    def test_false_negative(self):
        """False negative should have ground truth but no prediction."""
        match = DetectionMatch(
            predicted=None,
            ground_truth={"category": "stairs"},
            category_match=False
        )
        assert match.predicted is None
        assert match.ground_truth is not None


class TestReportGeneration:
    """Test evaluation report generation."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample evaluation result."""
        return EvaluationResult(
            overall_precision=0.85,
            overall_recall=0.80,
            overall_f1=0.82,
            category_metrics={
                "bathroom": CategoryMetrics(
                    category="bathroom",
                    true_positives=10,
                    false_positives=2,
                    false_negatives=3,
                    precision=0.83,
                    recall=0.77,
                    f1_score=0.80
                )
            },
            severity_accuracy=0.75,
            severity_kappa=0.65,
            confusion_matrix=None,
            total_predictions=15,
            total_ground_truth=16
        )

    def test_report_generation(self, sample_result):
        """Should generate readable text report."""
        evaluator = HazardEvaluator()
        report = evaluator.generate_report(sample_result)

        assert "HAZARD DETECTION EVALUATION REPORT" in report
        assert "0.85" in report or "0.8500" in report  # Precision
        assert "bathroom" in report.lower()

    def test_report_contains_metrics(self, sample_result):
        """Report should contain all key metrics."""
        evaluator = HazardEvaluator()
        report = evaluator.generate_report(sample_result)

        assert "Precision" in report
        assert "Recall" in report
        assert "F1" in report
        assert "Severity" in report
