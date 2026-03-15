"""
Base Vision-Language Model Interface

Provides abstract base class and Pydantic schemas for structured output
from vision-language models used in fall hazard detection.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import base64
import time

from pydantic import BaseModel, Field, field_validator


@dataclass
class ImageInput:
    """
    Standardized image input format for vision models.

    Supports multiple input types:
    - File path (local image)
    - URL (remote image)
    - Base64 encoded data

    Attributes:
        path: Local file path to the image
        url: Remote URL of the image
        base64_data: Pre-encoded base64 image data
        mime_type: MIME type of the image (default: image/jpeg)
    """
    path: Optional[str] = None
    url: Optional[str] = None
    base64_data: Optional[str] = None
    mime_type: str = "image/jpeg"

    def __post_init__(self):
        """Validate that at least one image source is provided."""
        if not any([self.path, self.url, self.base64_data]):
            raise ValueError("Must provide at least one of: path, url, or base64_data")

        # Auto-detect mime type from path extension
        if self.path and self.mime_type == "image/jpeg":
            ext = Path(self.path).suffix.lower()
            mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }
            self.mime_type = mime_map.get(ext, "image/jpeg")

    def to_base64(self) -> str:
        """
        Convert image to base64 string.

        Returns:
            Base64 encoded string of the image

        Raises:
            ValueError: If no valid image source is available
            FileNotFoundError: If the specified file path doesn't exist
        """
        if self.base64_data:
            return self.base64_data

        if self.path:
            path = Path(self.path)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {self.path}")

            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        raise ValueError("Cannot convert URL to base64 without downloading first")

    def get_data_uri(self) -> str:
        """
        Get image as data URI for API requests.

        Returns:
            Data URI string (data:mime_type;base64,...)
        """
        b64 = self.to_base64()
        return f"data:{self.mime_type};base64,{b64}"


class BoundingBox(BaseModel):
    """
    Normalized bounding box coordinates (0-1 range).

    Coordinates are relative to image dimensions for portability.
    """
    x: float = Field(..., ge=0.0, le=1.0, description="Left edge (0-1)")
    y: float = Field(..., ge=0.0, le=1.0, description="Top edge (0-1)")
    width: float = Field(..., ge=0.0, le=1.0, description="Box width (0-1)")
    height: float = Field(..., ge=0.0, le=1.0, description="Box height (0-1)")

    def to_pixels(self, img_width: int, img_height: int) -> Dict[str, int]:
        """Convert normalized coordinates to pixel values."""
        return {
            "x": int(self.x * img_width),
            "y": int(self.y * img_height),
            "width": int(self.width * img_width),
            "height": int(self.height * img_height),
        }


class DetectedHazard(BaseModel):
    """
    Individual hazard detected in an image.

    Aligned with Westmead Home Safety Assessment categories.

    Attributes:
        category: High-level hazard category (e.g., "flooring", "lighting")
        subcategory: Specific hazard type (e.g., "loose_rug", "dim_lighting")
        description: Human-readable description of the hazard
        severity: Risk severity level (low, medium, high, critical)
        confidence: Model's confidence in the detection (0.0 to 1.0)
        location_description: Where in the image the hazard is located
        bounding_box: Optional normalized coordinates for visualization
        recommendations: Suggested remediation actions
    """
    category: str = Field(
        ...,
        description="Hazard category: flooring, lighting, obstacles, furniture, stairs, bathroom, kitchen, bedroom, external, general"
    )
    subcategory: str = Field(
        ...,
        description="Specific hazard type identifier"
    )
    description: str = Field(
        ...,
        min_length=10,
        description="Detailed description of the identified hazard"
    )
    severity: str = Field(
        ...,
        description="Severity level: low, medium, high, or critical"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score (0-1)"
    )
    location_description: str = Field(
        ...,
        description="Where in the image the hazard is located"
    )
    bounding_box: Optional[BoundingBox] = Field(
        None,
        description="Normalized bounding box coordinates"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Suggested actions to mitigate the hazard"
    )

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Ensure severity is a valid level."""
        valid = {"low", "medium", "high", "critical"}
        v_lower = v.lower()
        if v_lower not in valid:
            raise ValueError(f"Severity must be one of: {valid}")
        return v_lower

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        """Ensure category is a valid Westmead-aligned category."""
        valid = {
            "flooring", "lighting", "obstacles", "furniture",
            "stairs", "bathroom", "kitchen", "bedroom",
            "external", "general"
        }
        v_lower = v.lower()
        if v_lower not in valid:
            raise ValueError(f"Category must be one of: {valid}")
        return v_lower


class HazardDetectionResult(BaseModel):
    """
    Complete result from hazard detection analysis.

    Contains all detected hazards, metadata, and model information.

    Attributes:
        room_type: Identified room type from the image
        hazards: List of all detected hazards
        overall_confidence: Model's overall confidence in the analysis
        raw_response: Optional raw text response from the model
        model_name: Name/ID of the model used
        processing_time_ms: Time taken for analysis in milliseconds
    """
    room_type: str = Field(
        ...,
        description="Detected room type: living_room, bedroom, bathroom, kitchen, hallway, stairs, external, other"
    )
    hazards: List[DetectedHazard] = Field(
        default_factory=list,
        description="List of all detected hazards"
    )
    overall_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the analysis"
    )
    raw_response: Optional[str] = Field(
        None,
        description="Raw text response from the model"
    )
    model_name: str = Field(
        default="unknown",
        description="Model identifier used for detection"
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds"
    )

    @property
    def hazard_count(self) -> int:
        """Return total number of detected hazards."""
        return len(self.hazards)

    @property
    def has_critical_hazards(self) -> bool:
        """Check if any critical hazards were detected."""
        return any(h.severity == "critical" for h in self.hazards)

    @property
    def hazards_by_category(self) -> Dict[str, List[DetectedHazard]]:
        """Group hazards by category."""
        grouped: Dict[str, List[DetectedHazard]] = {}
        for hazard in self.hazards:
            if hazard.category not in grouped:
                grouped[hazard.category] = []
            grouped[hazard.category].append(hazard)
        return grouped

    @property
    def hazards_by_severity(self) -> Dict[str, List[DetectedHazard]]:
        """Group hazards by severity level."""
        grouped: Dict[str, List[DetectedHazard]] = {}
        for hazard in self.hazards:
            if hazard.severity not in grouped:
                grouped[hazard.severity] = []
            grouped[hazard.severity].append(hazard)
        return grouped

    def get_high_priority_hazards(self, min_severity: str = "high") -> List[DetectedHazard]:
        """
        Get hazards at or above a minimum severity level.

        Args:
            min_severity: Minimum severity to include (default: high)

        Returns:
            List of hazards meeting the severity threshold
        """
        severity_order = ["low", "medium", "high", "critical"]
        min_idx = severity_order.index(min_severity)

        return [
            h for h in self.hazards
            if severity_order.index(h.severity) >= min_idx
        ]


class BaseVisionModel(ABC):
    """
    Abstract base class for vision-language model integration.

    Provides common interface for different VLM backends
    (OpenAI GPT-4V, Google Gemini, local models like LLaVA).

    Subclasses must implement:
    - analyze_image(): Raw text analysis
    - analyze_image_structured(): Structured output using Pydantic
    - health_check(): Model availability verification
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vision model.

        Args:
            config: Model configuration dictionary containing:
                - api_key: API key for the service
                - model_name: Specific model identifier
                - Additional model-specific settings
        """
        self.config = config
        self.model_name = config.get("model_name", "unknown")
        self._initialized = False

    @abstractmethod
    def analyze_image(
        self,
        image: ImageInput,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4096
    ) -> str:
        """
        Analyze an image and return raw text response.

        Args:
            image: Image to analyze
            prompt: Analysis prompt/instructions
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum response tokens

        Returns:
            Raw text response from the model
        """
        pass

    @abstractmethod
    def analyze_image_structured(
        self,
        image: ImageInput,
        prompt: str,
        response_schema: type[BaseModel],
        temperature: float = 0.1
    ) -> BaseModel:
        """
        Analyze an image and return structured output.

        Args:
            image: Image to analyze
            prompt: Analysis prompt/instructions
            response_schema: Pydantic model for response validation
            temperature: Sampling temperature (0-1)

        Returns:
            Validated Pydantic model instance
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Verify model connectivity and availability.

        Returns:
            True if model is accessible, False otherwise
        """
        pass

    def preprocess_prompt(self, template: str, **kwargs) -> str:
        """
        Fill prompt template with dynamic values.

        Args:
            template: Prompt template string with placeholders
            **kwargs: Values to substitute in template

        Returns:
            Filled prompt string
        """
        return template.format(**kwargs)

    def _get_system_prompt(self) -> str:
        """
        Get the default system prompt for hazard detection.

        Returns:
            System prompt establishing the AI's role
        """
        return """You are an expert occupational therapist with 20 years of experience
in geriatric fall prevention. You specialize in home safety assessments using
the Westmead Home Safety Assessment framework and CDC STEADI guidelines.

Your role is to analyze images of home environments to identify potential
fall hazards that could endanger elderly individuals (65+ years old).

Consider the perspective of someone with:
- Reduced vision and depth perception
- Balance and mobility challenges
- Slower reaction times
- Possible use of walking aids (cane, walker, wheelchair)

Be thorough, specific, and evidence-based in your assessments."""
