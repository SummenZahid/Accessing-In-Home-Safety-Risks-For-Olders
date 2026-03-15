"""Vision-Language Model integrations for hazard detection."""

from .base_model import (
    BaseVisionModel,
    ImageInput,
    DetectedHazard,
    HazardDetectionResult,
)

__all__ = [
    "BaseVisionModel",
    "ImageInput",
    "DetectedHazard",
    "HazardDetectionResult",
]
