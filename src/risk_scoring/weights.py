"""
Clinical Risk Weights for Fall Hazard Assessment

Defines category weights and severity multipliers based on clinical
evidence from fall prevention literature and the Westmead Home Safety
Assessment validation studies.

References:
- Clemson L. et al. (1997). Westmead Home Safety Assessment
- Rubenstein LZ. (2006). Falls in older people: epidemiology, risk factors
- CDC STEADI Framework (2023)
"""

from dataclasses import dataclass
from typing import Dict
from enum import Enum


class HazardCategory(Enum):
    """Hazard categories aligned with Westmead Assessment."""
    FLOORING = "flooring"
    LIGHTING = "lighting"
    OBSTACLES = "obstacles"
    FURNITURE = "furniture"
    STAIRS = "stairs"
    BATHROOM = "bathroom"
    KITCHEN = "kitchen"
    BEDROOM = "bedroom"
    EXTERNAL = "external"
    GENERAL = "general"


@dataclass
class CategoryWeight:
    """
    Weight configuration for a hazard category.

    Attributes:
        base_weight: Clinical importance factor (0-1)
                    Higher values indicate categories with stronger
                    correlation to fall incidents
        cumulative_factor: Multiplier for additional hazards in same category
                          Controls diminishing returns for multiple hazards
        max_contribution: Maximum score this category can contribute
                         Prevents any single category from dominating
    """
    base_weight: float
    cumulative_factor: float
    max_contribution: float


# =============================================================================
# CLINICAL EVIDENCE-BASED CATEGORY WEIGHTS
# =============================================================================

CATEGORY_WEIGHTS: Dict[HazardCategory, CategoryWeight] = {
    # STAIRS: Highest risk category
    # Stair falls result in most severe injuries (hip fractures, head trauma)
    # Studies show stairs are involved in 10% of fatal falls in elderly
    HazardCategory.STAIRS: CategoryWeight(
        base_weight=1.0,
        cumulative_factor=0.3,
        max_contribution=30.0
    ),

    # BATHROOM: Very high risk
    # Wet surfaces + transfers + hard surfaces = severe injury potential
    # 80% of bathroom falls occur during transfers (toilet, tub, shower)
    HazardCategory.BATHROOM: CategoryWeight(
        base_weight=0.95,
        cumulative_factor=0.25,
        max_contribution=25.0
    ),

    # FLOORING: High risk, most common hazard type
    # Rugs, slippery surfaces, uneven floors cause majority of trips
    # Easy to remediate but frequently overlooked
    HazardCategory.FLOORING: CategoryWeight(
        base_weight=0.85,
        cumulative_factor=0.2,
        max_contribution=20.0
    ),

    # OBSTACLES: High risk for trips
    # Cords, clutter, blocked pathways directly cause falls
    # Particularly dangerous at night or with impaired vision
    HazardCategory.OBSTACLES: CategoryWeight(
        base_weight=0.85,
        cumulative_factor=0.2,
        max_contribution=20.0
    ),

    # BEDROOM: Elevated risk for nighttime falls
    # Path to bathroom at night is high-risk period
    # Bed transfers and nighttime navigation critical
    HazardCategory.BEDROOM: CategoryWeight(
        base_weight=0.75,
        cumulative_factor=0.15,
        max_contribution=15.0
    ),

    # LIGHTING: Important but indirect risk factor
    # Poor lighting contributes to falls by reducing hazard visibility
    # Particularly important in combination with other hazards
    HazardCategory.LIGHTING: CategoryWeight(
        base_weight=0.75,
        cumulative_factor=0.15,
        max_contribution=15.0
    ),

    # FURNITURE: Moderate direct risk
    # Unstable furniture used for support can cause falls
    # Seating height affects transfer safety
    HazardCategory.FURNITURE: CategoryWeight(
        base_weight=0.70,
        cumulative_factor=0.15,
        max_contribution=15.0
    ),

    # KITCHEN: Moderate risk
    # Spills, reaching, multiple activities increase risk
    # Generally well-lit and frequently used
    HazardCategory.KITCHEN: CategoryWeight(
        base_weight=0.70,
        cumulative_factor=0.15,
        max_contribution=15.0
    ),

    # EXTERNAL: Moderate to high risk
    # Weather, lighting, surface conditions variable
    # Often less familiar than indoor environment
    HazardCategory.EXTERNAL: CategoryWeight(
        base_weight=0.80,
        cumulative_factor=0.2,
        max_contribution=20.0
    ),

    # GENERAL: Catch-all for miscellaneous hazards
    # Footwear, pets, mobility aids
    HazardCategory.GENERAL: CategoryWeight(
        base_weight=0.50,
        cumulative_factor=0.1,
        max_contribution=10.0
    ),
}


# =============================================================================
# SEVERITY MULTIPLIERS
# =============================================================================

SEVERITY_MULTIPLIERS: Dict[str, float] = {
    # LOW: Minor hazard, slight increase in fall risk
    # Example: Slightly dim lighting, minor clutter in corner
    "low": 0.25,

    # MEDIUM: Moderate hazard, noticeable increase in fall risk
    # Example: Rug without grip tape, chair difficult to rise from
    "medium": 0.50,

    # HIGH: Significant hazard, substantially increases fall risk
    # Example: Cord across walkway, wet floor surface
    "high": 0.75,

    # CRITICAL: Severe hazard, immediate fall risk
    # Example: Missing stair handrail, no grab bars in shower
    "critical": 1.00,
}


# =============================================================================
# RISK LEVEL THRESHOLDS
# =============================================================================

RISK_LEVEL_THRESHOLDS: Dict[str, tuple] = {
    # Score range, label, description, action urgency
    "low": (0, 25, "Low risk environment with minor concerns"),
    "moderate": (26, 50, "Moderate risk requiring attention within 2 weeks"),
    "high": (51, 75, "High risk requiring prompt attention within 48 hours"),
    "critical": (76, 100, "Critical risk requiring immediate action"),
}


# =============================================================================
# HAZARD BASE WEIGHTS
# =============================================================================

# Individual hazard severity weights from clinical guidelines
# These supplement the category weights for more granular scoring

HAZARD_BASE_WEIGHTS: Dict[str, float] = {
    # Flooring hazards
    "floor_loose_rug": 0.85,
    "floor_slippery": 0.90,
    "floor_uneven": 0.80,
    "floor_worn": 0.75,

    # Lighting hazards
    "light_insufficient": 0.75,
    "light_night_absent": 0.80,
    "light_glare": 0.60,
    "light_switch_access": 0.65,

    # Obstacle hazards
    "obstacle_cord": 0.90,
    "obstacle_clutter": 0.85,
    "obstacle_pathway": 0.85,
    "obstacle_spill": 0.85,

    # Furniture hazards
    "furniture_unstable": 0.85,
    "furniture_sharp_edges": 0.60,
    "furniture_seat_height": 0.70,
    "furniture_support_used": 0.80,

    # Stairs hazards
    "stairs_no_handrail": 0.95,
    "stairs_handrail_poor": 0.85,
    "stairs_slippery": 0.90,
    "stairs_visibility": 0.80,
    "stairs_dimensions": 0.75,

    # Bathroom hazards
    "bath_no_grab_bars": 0.95,
    "bath_grab_bars_poor": 0.80,
    "bath_floor_slippery": 0.90,
    "bath_tub_entry": 0.85,
    "bath_toilet_height": 0.70,
    "bath_floor_covering": 0.80,

    # Bedroom hazards
    "bedroom_bed_height": 0.70,
    "bedroom_path_bathroom": 0.85,
    "bedroom_lighting": 0.70,
    "bedroom_trailing": 0.65,

    # Kitchen hazards
    "kitchen_high_storage": 0.70,
    "kitchen_spill_areas": 0.75,
    "kitchen_floor": 0.80,

    # External hazards
    "external_pathway": 0.80,
    "external_steps": 0.85,
    "external_lighting": 0.75,
    "external_doormat": 0.70,

    # General hazards
    "general_mobility_aid": 0.75,
    "general_reaching": 0.75,
    "general_pets": 0.60,
    "general_footwear": 0.70,
}


def get_hazard_weight(hazard_id: str) -> float:
    """
    Get the clinical weight for a specific hazard type.

    Args:
        hazard_id: Hazard subcategory identifier

    Returns:
        Weight value (0-1), defaults to 0.5 if not found
    """
    return HAZARD_BASE_WEIGHTS.get(hazard_id, 0.5)


def get_category_weight(category: str) -> CategoryWeight:
    """
    Get the weight configuration for a category.

    Args:
        category: Category name or HazardCategory enum

    Returns:
        CategoryWeight configuration
    """
    if isinstance(category, str):
        try:
            category = HazardCategory(category.lower())
        except ValueError:
            # Return default weights for unknown category
            return CategoryWeight(0.5, 0.1, 10.0)

    return CATEGORY_WEIGHTS.get(
        category,
        CategoryWeight(0.5, 0.1, 10.0)
    )
