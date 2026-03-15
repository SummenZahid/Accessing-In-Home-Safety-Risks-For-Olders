"""Hazard detection module with Westmead-aligned categories."""

from .categories import (
    HazardCategory,
    HazardSubcategory,
    HAZARD_DEFINITIONS,
    get_hazards_by_category,
    get_room_priority_categories,
)

__all__ = [
    "HazardCategory",
    "HazardSubcategory",
    "HAZARD_DEFINITIONS",
    "get_hazards_by_category",
    "get_room_priority_categories",
]
