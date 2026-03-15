"""
Tests for Hazard Detection Module

Tests the hazard detection functionality including:
- Hazard category definitions
- Detection result parsing
- Multi-pass detection strategies
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hazard_detection.categories import (
    HazardCategory,
    HazardSubcategory,
    HAZARD_DEFINITIONS,
    get_hazards_by_category,
    get_room_priority_categories,
    get_all_detection_keywords,
    get_category_weights,
)


class TestHazardCategories:
    """Test hazard category definitions."""

    def test_all_categories_defined(self):
        """Ensure all expected categories are defined."""
        expected = [
            "flooring", "lighting", "obstacles", "furniture",
            "stairs", "bathroom", "kitchen", "bedroom",
            "external", "general"
        ]
        category_names = [cat.value for cat in HazardCategory]
        for exp in expected:
            assert exp in category_names, f"Missing category: {exp}"

    def test_category_count(self):
        """Verify total number of categories."""
        assert len(HazardCategory) == 10

    def test_categories_have_hazards(self):
        """Each category should have at least one hazard defined."""
        for category in HazardCategory:
            hazards = get_hazards_by_category(category)
            assert len(hazards) > 0, f"No hazards for {category.name}"


class TestHazardDefinitions:
    """Test individual hazard definitions."""

    def test_hazard_structure(self):
        """Each hazard should have required fields."""
        for hazard_id, hazard in HAZARD_DEFINITIONS.items():
            assert hasattr(hazard, 'id'), f"Missing id in {hazard_id}"
            assert hasattr(hazard, 'name'), f"Missing name in {hazard_id}"
            assert hasattr(hazard, 'category'), f"Missing category in {hazard_id}"
            assert hasattr(hazard, 'clinical_severity_weight'), f"Missing weight in {hazard_id}"
            assert hasattr(hazard, 'westmead_section'), f"Missing westmead_section in {hazard_id}"
            assert hasattr(hazard, 'examples'), f"Missing examples in {hazard_id}"

    def test_severity_weights_valid(self):
        """Severity weights should be between 0 and 1."""
        for hazard_id, hazard in HAZARD_DEFINITIONS.items():
            weight = hazard.clinical_severity_weight
            assert 0 < weight <= 1, \
                f"Invalid weight for {hazard_id}: {weight}"

    def test_westmead_references(self):
        """Hazards should reference Westmead sections."""
        for hazard_id, hazard in HAZARD_DEFINITIONS.items():
            ref = hazard.westmead_section
            assert ref, f"Missing Westmead ref for {hazard_id}"


class TestSubcategories:
    """Test subcategory retrieval."""

    def test_flooring_subcategories(self):
        """Flooring should have expected subcategories."""
        hazards = get_hazards_by_category(HazardCategory.FLOORING)
        hazard_names = [h.name.lower() for h in hazards]
        # Should have rug or floor related hazards
        assert any('rug' in name or 'floor' in name for name in hazard_names), \
            "Flooring missing expected hazards"

    def test_bathroom_subcategories(self):
        """Bathroom should have grab bar related subcategories."""
        hazards = get_hazards_by_category(HazardCategory.BATHROOM)
        hazard_names = [h.name.lower() for h in hazards]
        # Should have grab bar hazards
        assert any('grab' in name for name in hazard_names), \
            "Bathroom missing grab bar hazards"

    def test_stairs_subcategories(self):
        """Stairs should have handrail subcategories."""
        hazards = get_hazards_by_category(HazardCategory.STAIRS)
        hazard_names = [h.name.lower() for h in hazards]
        assert any('handrail' in name or 'rail' in name for name in hazard_names), \
            "Stairs missing handrail hazards"


class TestDetectionKeywords:
    """Test keyword retrieval for detection prompts."""

    def test_keywords_returned(self):
        """Keywords should be returned."""
        keywords = get_all_detection_keywords()
        assert len(keywords) > 0, "No keywords returned"

    def test_keywords_dict_structure(self):
        """Keywords should be a dictionary."""
        keywords = get_all_detection_keywords()
        assert isinstance(keywords, dict), "Keywords should be a dict"


class TestHazardCategoryEnum:
    """Test HazardCategory enum functionality."""

    def test_enum_values(self):
        """Enum values should be accessible."""
        assert HazardCategory.FLOORING.value == "flooring"
        assert HazardCategory.STAIRS.value == "stairs"
        assert HazardCategory.BATHROOM.value == "bathroom"

    def test_enum_from_string(self):
        """Should be able to create enum from string."""
        cat = HazardCategory("flooring")
        assert cat == HazardCategory.FLOORING

    def test_invalid_category(self):
        """Invalid category string should raise ValueError."""
        with pytest.raises(ValueError):
            HazardCategory("invalid_category")


class TestCategoryWeights:
    """Test that categories have appropriate clinical weights."""

    def test_category_weights_exist(self):
        """Category weights should be defined."""
        weights = get_category_weights()
        assert len(weights) > 0, "No category weights defined"

    def test_stairs_has_weight(self):
        """Stairs category should have a weight."""
        weights = get_category_weights()
        assert HazardCategory.STAIRS in weights, "Stairs missing from weights"

    def test_bathroom_has_weight(self):
        """Bathroom category should have a weight."""
        weights = get_category_weights()
        assert HazardCategory.BATHROOM in weights, "Bathroom missing from weights"


class TestRoomPriority:
    """Test room priority categories."""

    def test_bathroom_priority(self):
        """Bathroom room should prioritize bathroom hazards."""
        priorities = get_room_priority_categories("bathroom")
        assert HazardCategory.BATHROOM in priorities, "Bathroom not prioritized for bathroom room"

    def test_kitchen_priority(self):
        """Kitchen room should prioritize kitchen hazards."""
        priorities = get_room_priority_categories("kitchen")
        assert HazardCategory.KITCHEN in priorities, "Kitchen not prioritized for kitchen room"

    def test_stairs_priority(self):
        """Stairs room should prioritize stairs hazards."""
        priorities = get_room_priority_categories("stairs")
        assert HazardCategory.STAIRS in priorities, "Stairs not prioritized for stairs room"


class TestHazardExamples:
    """Test that hazards have useful examples."""

    def test_examples_are_lists(self):
        """Examples should be lists of strings."""
        for hazard_id, hazard in HAZARD_DEFINITIONS.items():
            examples = hazard.examples
            assert isinstance(examples, list), \
                f"Examples not a list for {hazard_id}"

    def test_examples_not_empty(self):
        """Each hazard should have at least one example."""
        for hazard_id, hazard in HAZARD_DEFINITIONS.items():
            examples = hazard.examples
            assert len(examples) > 0, \
                f"No examples for {hazard_id}"

    def test_examples_are_strings(self):
        """Examples should be strings."""
        for hazard_id, hazard in HAZARD_DEFINITIONS.items():
            for example in hazard.examples:
                assert isinstance(example, str), \
                    f"Non-string example in {hazard_id}"
