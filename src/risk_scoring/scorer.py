"""
Risk Scoring Engine for Fall Hazard Assessment

Calculates weighted risk scores from detected hazards, providing
quantified risk levels aligned with clinical assessment standards.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from ..models.base_model import DetectedHazard, HazardDetectionResult
from .weights import (
    HazardCategory,
    CategoryWeight,
    CATEGORY_WEIGHTS,
    SEVERITY_MULTIPLIERS,
    HAZARD_BASE_WEIGHTS,
    get_hazard_weight,
    get_category_weight,
)


class RiskLevel(Enum):
    """Overall risk level classification."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CategoryScore:
    """
    Score breakdown for a single hazard category.

    Attributes:
        category: The hazard category
        raw_score: Unweighted sum of hazard scores
        weighted_score: Score after applying category weight
        hazard_count: Number of hazards in this category
        max_severity: Highest severity level in this category
        contributing_hazards: List of hazard IDs contributing to score
    """
    category: HazardCategory
    raw_score: float
    weighted_score: float
    hazard_count: int
    max_severity: str
    contributing_hazards: List[str]


@dataclass
class RiskScoreResult:
    """
    Complete risk scoring result.

    Attributes:
        total_score: Normalized risk score (0-100)
        risk_level: Classified risk level
        category_breakdown: Score details per category
        confidence: Average detection confidence
        top_hazards: Most critical hazards identified
        recommendations_summary: Prioritized action items
        explanation: Human-readable risk explanation
    """
    total_score: float
    risk_level: RiskLevel
    category_breakdown: Dict[str, CategoryScore]
    confidence: float
    top_hazards: List[DetectedHazard]
    recommendations_summary: List[str]
    explanation: str


class RiskScorer:
    """
    Calculate weighted risk scores from detected hazards.

    Scoring methodology:
    1. Each hazard gets a base score from its clinical severity weight
    2. Severity multiplier applied (low=0.25 to critical=1.0)
    3. Confidence weighting applied
    4. Category scores aggregated with diminishing returns
    5. Final score normalized to 0-100 scale

    Attributes:
        category_weights: Weight configuration per category
        severity_multipliers: Multipliers per severity level
    """

    def __init__(
        self,
        category_weights: Optional[Dict[HazardCategory, CategoryWeight]] = None,
        severity_multipliers: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the risk scorer.

        Args:
            category_weights: Custom category weights (uses defaults if None)
            severity_multipliers: Custom severity multipliers (uses defaults if None)
        """
        self.category_weights = category_weights or CATEGORY_WEIGHTS
        self.severity_multipliers = severity_multipliers or SEVERITY_MULTIPLIERS

    def calculate_score(
        self,
        detection_result: HazardDetectionResult
    ) -> RiskScoreResult:
        """
        Calculate comprehensive risk score from detection result.

        Args:
            detection_result: HazardDetectionResult from vision model

        Returns:
            RiskScoreResult with complete scoring breakdown
        """
        # Group hazards by category
        hazards_by_category = self._group_by_category(detection_result.hazards)

        # Calculate category scores
        category_scores: Dict[str, CategoryScore] = {}
        total_weighted_score = 0.0

        for category, hazards in hazards_by_category.items():
            cat_score = self._calculate_category_score(category, hazards)
            category_scores[category.value] = cat_score
            total_weighted_score += cat_score.weighted_score

        # Normalize to 0-100 scale
        max_possible = sum(
            w.max_contribution for w in self.category_weights.values()
        )
        normalized_score = min(100, (total_weighted_score / max_possible) * 100)

        # Determine risk level
        risk_level = self._get_risk_level(normalized_score)

        # Get top hazards by priority
        top_hazards = self._get_top_hazards(detection_result.hazards, n=5)

        # Generate recommendations summary
        recommendations = self._generate_recommendations(top_hazards, category_scores)

        # Generate explanation
        explanation = self._generate_explanation(
            normalized_score,
            risk_level,
            category_scores,
            detection_result.room_type,
            len(detection_result.hazards)
        )

        return RiskScoreResult(
            total_score=round(normalized_score, 1),
            risk_level=risk_level,
            category_breakdown=category_scores,
            confidence=detection_result.overall_confidence,
            top_hazards=top_hazards,
            recommendations_summary=recommendations,
            explanation=explanation
        )

    def _group_by_category(
        self,
        hazards: List[DetectedHazard]
    ) -> Dict[HazardCategory, List[DetectedHazard]]:
        """Group hazards by their category."""
        grouped: Dict[HazardCategory, List[DetectedHazard]] = defaultdict(list)

        for hazard in hazards:
            try:
                category = HazardCategory(hazard.category.lower())
            except ValueError:
                category = HazardCategory.GENERAL
            grouped[category].append(hazard)

        return grouped

    def _calculate_category_score(
        self,
        category: HazardCategory,
        hazards: List[DetectedHazard]
    ) -> CategoryScore:
        """
        Calculate weighted score for a single category.

        Implements diminishing returns for multiple hazards in same category.
        """
        cat_weight = self.category_weights.get(
            category,
            CategoryWeight(0.5, 0.1, 10.0)
        )

        raw_scores = []
        severities = []
        hazard_ids = []

        for i, hazard in enumerate(hazards):
            # Get hazard base weight
            base_weight = get_hazard_weight(hazard.subcategory)

            # Apply severity multiplier
            severity_mult = self.severity_multipliers.get(
                hazard.severity.lower(), 0.5
            )

            # Apply confidence weighting
            confidence_weight = hazard.confidence

            # Calculate individual hazard score
            hazard_score = base_weight * severity_mult * confidence_weight

            # Apply diminishing returns for multiple hazards
            # First hazard: full contribution
            # Subsequent: reduced by cumulative factor
            if i == 0:
                position_factor = 1.0
            else:
                position_factor = cat_weight.cumulative_factor

            raw_scores.append(hazard_score * position_factor)
            severities.append(hazard.severity.lower())
            hazard_ids.append(hazard.subcategory)

        # Sum raw scores
        raw_score = sum(raw_scores)

        # Apply category weight and scale
        weighted_score = min(
            raw_score * cat_weight.base_weight * 10,
            cat_weight.max_contribution
        )

        # Determine max severity
        severity_order = ["low", "medium", "high", "critical"]
        max_severity = max(
            severities,
            key=lambda s: severity_order.index(s) if s in severity_order else 0
        ) if severities else "low"

        return CategoryScore(
            category=category,
            raw_score=round(raw_score, 3),
            weighted_score=round(weighted_score, 2),
            hazard_count=len(hazards),
            max_severity=max_severity,
            contributing_hazards=hazard_ids
        )

    def _get_risk_level(self, score: float) -> RiskLevel:
        """Determine risk level from normalized score."""
        if score <= 25:
            return RiskLevel.LOW
        elif score <= 50:
            return RiskLevel.MODERATE
        elif score <= 75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _get_top_hazards(
        self,
        hazards: List[DetectedHazard],
        n: int = 5
    ) -> List[DetectedHazard]:
        """Get the N most critical hazards by priority score."""
        return sorted(
            hazards,
            key=lambda h: self._hazard_priority_score(h),
            reverse=True
        )[:n]

    def _hazard_priority_score(self, hazard: DetectedHazard) -> float:
        """Calculate priority score for ranking hazards."""
        severity_scores = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        severity_value = severity_scores.get(hazard.severity.lower(), 1)

        try:
            category = HazardCategory(hazard.category.lower())
            cat_weight = self.category_weights.get(
                category,
                CategoryWeight(0.5, 0.1, 10.0)
            )
        except ValueError:
            cat_weight = CategoryWeight(0.5, 0.1, 10.0)

        return severity_value * hazard.confidence * cat_weight.base_weight

    def _generate_recommendations(
        self,
        top_hazards: List[DetectedHazard],
        category_scores: Dict[str, CategoryScore]
    ) -> List[str]:
        """Generate prioritized recommendations list."""
        recommendations = []
        seen = set()

        # Collect recommendations from top hazards
        for hazard in top_hazards:
            for rec in hazard.recommendations:
                if rec not in seen:
                    recommendations.append(rec)
                    seen.add(rec)
                if len(recommendations) >= 10:
                    break
            if len(recommendations) >= 10:
                break

        return recommendations

    def _generate_explanation(
        self,
        score: float,
        risk_level: RiskLevel,
        category_scores: Dict[str, CategoryScore],
        room_type: str,
        total_hazards: int
    ) -> str:
        """Generate human-readable explanation of the risk score."""
        # Find top contributing categories
        sorted_categories = sorted(
            category_scores.items(),
            key=lambda x: x[1].weighted_score,
            reverse=True
        )

        top_categories = [
            f"{cat} ({cs.hazard_count} hazard{'s' if cs.hazard_count > 1 else ''}, "
            f"max severity: {cs.max_severity})"
            for cat, cs in sorted_categories[:3]
            if cs.hazard_count > 0
        ]

        level_descriptions = {
            RiskLevel.LOW: "relatively safe with minor concerns that can be addressed at convenience",
            RiskLevel.MODERATE: "presenting moderate fall risks that should be addressed within 2 weeks",
            RiskLevel.HIGH: "presenting significant fall risks requiring prompt attention within 48 hours",
            RiskLevel.CRITICAL: "presenting critical fall risks requiring immediate action before continued use"
        }

        level_desc = level_descriptions.get(
            risk_level,
            "requiring assessment"
        )

        if top_categories:
            categories_text = f"Primary areas of concern: {', '.join(top_categories)}."
        else:
            categories_text = "No significant hazards identified."

        explanation = f"""This {room_type} environment has been assessed with a fall risk score of {score:.1f} out of 100, classified as {risk_level.value.upper()} risk.

The environment is {level_desc}.

{categories_text}

Total hazards identified: {total_hazards}

This assessment is based on the Westmead Home Safety Assessment framework and should be used to guide fall prevention interventions for elderly individuals."""

        return explanation.strip()

    def calculate_quick_score(
        self,
        hazards: List[DetectedHazard]
    ) -> float:
        """
        Calculate a quick risk score without full breakdown.

        Useful for comparing multiple assessments quickly.

        Args:
            hazards: List of detected hazards

        Returns:
            Normalized risk score (0-100)
        """
        total_score = 0.0

        for hazard in hazards:
            base_weight = get_hazard_weight(hazard.subcategory)
            severity_mult = self.severity_multipliers.get(hazard.severity.lower(), 0.5)
            total_score += base_weight * severity_mult * hazard.confidence

        # Simple normalization
        max_expected = len(hazards) * 1.0 if hazards else 1.0
        normalized = min(100, (total_score / max_expected) * 100)

        return round(normalized, 1)
