"""
Tests for Risk Scoring Module

Tests the risk scoring functionality including:
- Score calculation
- Category weights application
- Risk level classification
- Recommendation generation
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from risk_scoring.weights import (
    CATEGORY_WEIGHTS,
    SEVERITY_MULTIPLIERS,
    HAZARD_BASE_WEIGHTS,
    HazardCategory,
)
from risk_scoring.scorer import RiskScorer, RiskLevel


class TestCategoryWeights:
    """Test category weight definitions."""

    def test_all_categories_have_weights(self):
        """Ensure all main categories have weights defined."""
        for cat in HazardCategory:
            assert cat in CATEGORY_WEIGHTS, f"Missing weight for {cat}"

    def test_weights_are_valid(self):
        """Ensure weights are between 0 and 1."""
        for cat, weight in CATEGORY_WEIGHTS.items():
            assert 0 < weight.base_weight <= 1, f"Invalid weight for {cat}: {weight}"

    def test_stairs_highest_weight(self):
        """Stairs should have highest risk weight."""
        stairs_weight = CATEGORY_WEIGHTS[HazardCategory.STAIRS].base_weight
        for cat, weight in CATEGORY_WEIGHTS.items():
            assert stairs_weight >= weight.base_weight, f"{cat} has higher weight than stairs"


class TestSeverityMultipliers:
    """Test severity multiplier definitions."""

    def test_all_severities_defined(self):
        """Ensure all severity levels have multipliers."""
        expected = ["low", "medium", "high", "critical"]
        for sev in expected:
            assert sev in SEVERITY_MULTIPLIERS, f"Missing multiplier for {sev}"

    def test_severity_ordering(self):
        """Severity multipliers should increase with severity."""
        assert SEVERITY_MULTIPLIERS["low"] < SEVERITY_MULTIPLIERS["medium"]
        assert SEVERITY_MULTIPLIERS["medium"] < SEVERITY_MULTIPLIERS["high"]
        assert SEVERITY_MULTIPLIERS["high"] < SEVERITY_MULTIPLIERS["critical"]


class TestRiskLevel:
    """Test risk level enum."""

    def test_risk_levels_exist(self):
        """Risk levels should be defined."""
        assert RiskLevel.LOW
        assert RiskLevel.MODERATE
        assert RiskLevel.HIGH
        assert RiskLevel.CRITICAL

    def test_risk_level_values(self):
        """Risk level values should be strings."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.CRITICAL.value == "critical"


class TestRiskScorer:
    """Test the RiskScorer class."""

    @pytest.fixture
    def scorer(self):
        """Create a RiskScorer instance."""
        return RiskScorer()

    def test_empty_hazards(self, scorer):
        """No hazards should result in zero risk."""
        result = scorer.calculate_risk([])
        assert result.overall_score == 0
        assert result.risk_level == RiskLevel.LOW

    def test_single_hazard(self, scorer, sample_hazard):
        """Single hazard should produce non-zero score."""
        result = scorer.calculate_risk([sample_hazard])
        assert result.overall_score > 0
        assert len(result.category_scores) >= 1

    def test_multiple_hazards(self, scorer, sample_hazard_list):
        """Multiple hazards should accumulate risk."""
        result = scorer.calculate_risk(sample_hazard_list)
        assert result.overall_score > 0
        assert result.total_hazards == len(sample_hazard_list)

    def test_critical_hazard_high_score(self, scorer):
        """Critical hazards should contribute significant risk."""
        critical_hazard = {
            "category": "stairs",
            "subcategory": "no_handrails",
            "severity": "critical"
        }
        result = scorer.calculate_risk([critical_hazard])
        # Stairs + critical should be high risk
        assert result.overall_score >= 20

    def test_low_hazard_low_score(self, scorer):
        """Low severity hazards should contribute minimal risk."""
        low_hazard = {
            "category": "general",
            "subcategory": "general_clutter",
            "severity": "low"
        }
        result = scorer.calculate_risk([low_hazard])
        # Low severity general hazard should be minimal
        assert result.overall_score < 20

    def test_category_breakdown(self, scorer, sample_hazard_list):
        """Results should include category-level breakdown."""
        result = scorer.calculate_risk(sample_hazard_list)
        assert hasattr(result, 'category_scores')
        assert len(result.category_scores) > 0

    def test_top_hazards_limited(self, scorer):
        """Top hazards list should be limited in size."""
        # Create many hazards
        hazards = [
            {"category": "flooring", "subcategory": f"issue_{i}", "severity": "medium"}
            for i in range(20)
        ]
        result = scorer.calculate_risk(hazards)
        assert len(result.top_hazards) <= 10  # Should be limited

    def test_recommendations_generated(self, scorer, sample_hazard_list):
        """Recommendations should be generated for hazards."""
        result = scorer.calculate_risk(sample_hazard_list)
        assert hasattr(result, 'recommendations')
        assert len(result.recommendations) > 0

    def test_score_normalized_to_100(self, scorer):
        """Score should be normalized to 0-100 range."""
        # Create extreme case with many critical hazards
        hazards = [
            {"category": cat, "subcategory": "critical_issue", "severity": "critical"}
            for cat in ["stairs", "bathroom", "flooring", "lighting", "obstacles"]
        ]
        result = scorer.calculate_risk(hazards)
        assert 0 <= result.overall_score <= 100

    def test_diminishing_returns(self, scorer):
        """Multiple hazards in same category should have diminishing effect."""
        single_hazard = [{"category": "flooring", "subcategory": "loose_rug", "severity": "high"}]
        double_hazard = single_hazard * 2
        triple_hazard = single_hazard * 3

        single_score = scorer.calculate_risk(single_hazard).overall_score
        double_score = scorer.calculate_risk(double_hazard).overall_score
        triple_score = scorer.calculate_risk(triple_hazard).overall_score

        # Score should increase but not linearly
        assert double_score > single_score
        assert triple_score > double_score
        assert double_score < single_score * 2  # Diminishing returns


class TestHazardWeights:
    """Test individual hazard weight definitions."""

    def test_hazard_weights_exist(self):
        """Verify hazard weights are defined."""
        assert len(HAZARD_BASE_WEIGHTS) > 0

    def test_stairs_hazards_high_weight(self):
        """Stair-related hazards should have high base weights."""
        stair_hazards = [k for k in HAZARD_BASE_WEIGHTS.keys() if k.startswith("stairs_")]
        for hazard in stair_hazards:
            assert HAZARD_BASE_WEIGHTS[hazard] >= 0.7, f"{hazard} should have high weight"

    def test_bathroom_grab_bars_critical(self):
        """Missing grab bars should have high weight."""
        grab_bar_hazards = [
            k for k in HAZARD_BASE_WEIGHTS.keys()
            if "grab_bar" in k or "no_grab" in k
        ]
        for hazard in grab_bar_hazards:
            assert HAZARD_BASE_WEIGHTS.get(hazard, 0) >= 0.8
