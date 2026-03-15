"""
Hazard Detection Orchestrator

Main orchestrator for detecting fall hazards in home environment images.
Implements multi-pass detection strategy for improved accuracy and coverage.
"""

import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import yaml

from ..models.base_model import (
    BaseVisionModel,
    ImageInput,
    HazardDetectionResult,
    DetectedHazard,
)
from .categories import (
    HazardCategory,
    HAZARD_DEFINITIONS,
    get_room_priority_categories,
)


@dataclass
class DetectionConfig:
    """
    Configuration for hazard detection.

    Attributes:
        use_chain_of_thought: Enable step-by-step reasoning in prompts
        multi_pass: Use multiple focused passes for better coverage
        confidence_threshold: Minimum confidence to include hazard
        focus_categories: Specific categories to focus on (None = all)
        room_type_hint: Pre-specified room type (None = auto-detect)
        temperature: Model sampling temperature
    """
    use_chain_of_thought: bool = True
    multi_pass: bool = True
    confidence_threshold: float = 0.5
    focus_categories: Optional[List[HazardCategory]] = None
    room_type_hint: Optional[str] = None
    temperature: float = 0.1


class HazardDetector:
    """
    Main orchestrator for hazard detection using vision-language models.

    Implements a multi-pass detection strategy:
    1. General comprehensive scan to identify room type and major hazards
    2. Category-specific focused passes on high-priority areas
    3. Merge and deduplicate results

    Attributes:
        model: Vision-language model for image analysis
        prompts_dir: Directory containing prompt templates
    """

    def __init__(
        self,
        vision_model: BaseVisionModel,
        prompts_dir: str = "configs/prompts"
    ):
        """
        Initialize the hazard detector.

        Args:
            vision_model: Configured vision-language model
            prompts_dir: Path to prompt template directory
        """
        self.model = vision_model
        self.prompts_dir = Path(prompts_dir)
        self._prompts_cache: Dict[str, Any] = {}

    def detect_hazards(
        self,
        image: ImageInput,
        config: Optional[DetectionConfig] = None
    ) -> HazardDetectionResult:
        """
        Run comprehensive hazard detection on an image.

        Args:
            image: Image to analyze
            config: Detection configuration (uses defaults if None)

        Returns:
            HazardDetectionResult with all detected hazards
        """
        config = config or DetectionConfig()
        start_time = time.time()

        if config.multi_pass:
            result = self._multi_pass_detection(image, config)
        else:
            result = self._single_pass_detection(image, config)

        result.processing_time_ms = (time.time() - start_time) * 1000
        result.model_name = self.model.model_name

        return result

    def _single_pass_detection(
        self,
        image: ImageInput,
        config: DetectionConfig
    ) -> HazardDetectionResult:
        """
        Single comprehensive detection pass.

        Analyzes the entire image in one API call.
        """
        prompt = self._build_detection_prompt(
            room_type=config.room_type_hint,
            focus_categories=config.focus_categories,
            use_chain_of_thought=config.use_chain_of_thought,
        )

        result = self.model.analyze_image_structured(
            image=image,
            prompt=prompt,
            response_schema=HazardDetectionResult,
            temperature=config.temperature,
        )

        # Filter by confidence threshold
        result.hazards = [
            h for h in result.hazards
            if h.confidence >= config.confidence_threshold
        ]

        return result

    def _multi_pass_detection(
        self,
        image: ImageInput,
        config: DetectionConfig
    ) -> HazardDetectionResult:
        """
        Multi-pass detection strategy for improved coverage.

        Pass 1: General scan to identify room type and major hazards
        Pass 2+: Focused passes on high-priority categories for the room type
        Final: Merge and deduplicate all results
        """
        # Pass 1: General comprehensive scan
        general_prompt = self._build_detection_prompt(
            use_chain_of_thought=config.use_chain_of_thought,
        )

        general_result = self.model.analyze_image_structured(
            image=image,
            prompt=general_prompt,
            response_schema=HazardDetectionResult,
            temperature=config.temperature,
        )

        all_hazards = list(general_result.hazards)
        room_type = general_result.room_type

        # Determine which categories to focus on
        if config.focus_categories:
            priority_categories = config.focus_categories
        else:
            priority_categories = get_room_priority_categories(room_type)

        # Pass 2-N: Category-specific focused passes
        for category in priority_categories[:2]:  # Limit to top 2 for efficiency
            focused_prompt = self._build_focused_prompt(
                room_type=room_type,
                category=category,
                use_chain_of_thought=config.use_chain_of_thought,
            )

            try:
                focused_result = self.model.analyze_image_structured(
                    image=image,
                    prompt=focused_prompt,
                    response_schema=HazardDetectionResult,
                    temperature=config.temperature + 0.05,  # Slightly higher for diversity
                )
                all_hazards.extend(focused_result.hazards)
            except Exception:
                # Continue if focused pass fails
                pass

        # Merge and deduplicate
        merged_hazards = self._merge_hazards(all_hazards)

        # Filter by confidence threshold
        filtered_hazards = [
            h for h in merged_hazards
            if h.confidence >= config.confidence_threshold
        ]

        return HazardDetectionResult(
            room_type=room_type,
            hazards=filtered_hazards,
            overall_confidence=general_result.overall_confidence,
            raw_response=None,
            model_name=self.model.model_name,
            processing_time_ms=0,  # Will be set by caller
        )

    def _build_detection_prompt(
        self,
        room_type: Optional[str] = None,
        focus_categories: Optional[List[HazardCategory]] = None,
        use_chain_of_thought: bool = True,
    ) -> str:
        """
        Build a comprehensive hazard detection prompt.

        Args:
            room_type: Pre-specified room type (None = auto-detect)
            focus_categories: Categories to focus on
            use_chain_of_thought: Include reasoning steps

        Returns:
            Complete prompt string
        """
        parts = []

        # Task description
        parts.append(self._get_task_description(room_type))

        # Chain-of-thought reasoning
        if use_chain_of_thought:
            parts.append(self._get_chain_of_thought_instructions())

        # Category guidance
        parts.append(self._get_category_guidance(focus_categories))

        # Severity guidelines
        parts.append(self._get_severity_guidelines())

        # Output format
        parts.append(self._get_output_format())

        return "\n\n".join(parts)

    def _build_focused_prompt(
        self,
        room_type: str,
        category: HazardCategory,
        use_chain_of_thought: bool = True,
    ) -> str:
        """
        Build a category-focused detection prompt.

        Args:
            room_type: Identified room type
            category: Category to focus on
            use_chain_of_thought: Include reasoning steps

        Returns:
            Focused prompt string
        """
        parts = []

        # Focused task
        parts.append(f"""## Focused Analysis: {category.value.upper()} Hazards

You previously identified this as a {room_type}. Now focus SPECIFICALLY on {category.value} hazards.

Look for ANY hazards related to {category.value} that may have been missed in the initial scan.
Be thorough and check for subtle hazards that could be easily overlooked.""")

        # Category-specific guidance
        category_hazards = [
            h for h in HAZARD_DEFINITIONS.values()
            if h.category == category
        ]

        if category_hazards:
            parts.append(f"## {category.value.upper()} Hazards to Look For:")
            for hazard in category_hazards:
                examples = ", ".join(hazard.examples[:3])
                parts.append(f"- **{hazard.name}**: {hazard.description}")
                parts.append(f"  Examples: {examples}")

        # Output format
        parts.append(self._get_output_format())

        return "\n\n".join(parts)

    def _get_task_description(self, room_type: Optional[str] = None) -> str:
        """Get the main task description."""
        room_context = (
            f"You are analyzing a **{room_type}**."
            if room_type
            else "First, identify what type of room or area this is."
        )

        return f"""## Task: Fall Hazard Detection for Elderly Safety

Analyze this home environment image to identify ALL potential fall hazards
that could endanger an elderly person (65+ years old).

{room_context}

Consider the perspective of someone with:
- Reduced vision and depth perception
- Balance and mobility challenges
- Slower reaction times
- Possible use of walking aids (cane, walker, wheelchair)

Your assessment will be used to improve home safety and prevent falls.
Be thorough, specific, and evidence-based."""

    def _get_chain_of_thought_instructions(self) -> str:
        """Get chain-of-thought reasoning instructions."""
        return """## Analysis Process (Think Step-by-Step)

Before providing your final assessment, work through these steps:

1. **ROOM TYPE**: Identify what type of room or area this is
2. **SYSTEMATIC SCAN**: Look at the entire image from left to right, top to bottom
3. **FLOOR CHECK**: Examine floor surfaces, rugs, cords, clutter, transitions
4. **VERTICAL CHECK**: Look at furniture, railings, lighting, obstacles at height
5. **LIGHTING**: Assess light levels, shadows, glare sources
6. **PATHWAYS**: Trace likely walking routes and identify obstructions
7. **SUPPORT**: Note what objects might be used for support while walking
8. **PRIORITIZE**: Rank hazards by severity and likelihood

Document your reasoning for each identified hazard."""

    def _get_category_guidance(
        self,
        focus_categories: Optional[List[HazardCategory]] = None
    ) -> str:
        """Get category-specific guidance text."""
        categories = focus_categories or list(HazardCategory)

        parts = ["## Hazard Categories to Evaluate"]

        category_examples = {
            HazardCategory.FLOORING: [
                "Loose/unsecured rugs", "Slippery surfaces",
                "Uneven floors", "Worn coverings"
            ],
            HazardCategory.LIGHTING: [
                "Dim/dark areas", "No night lights",
                "Glare", "Inaccessible switches"
            ],
            HazardCategory.OBSTACLES: [
                "Trailing cords", "Floor clutter",
                "Blocked pathways", "Spills"
            ],
            HazardCategory.FURNITURE: [
                "Unstable furniture", "Sharp edges",
                "Wrong seat height", "Used for support"
            ],
            HazardCategory.STAIRS: [
                "Missing handrails", "Slippery treads",
                "Poor visibility", "Uneven steps"
            ],
            HazardCategory.BATHROOM: [
                "No grab bars", "Slippery floors",
                "High tub entry", "Toilet height"
            ],
            HazardCategory.BEDROOM: [
                "Bed height", "Path to bathroom",
                "Lighting", "Trailing items"
            ],
            HazardCategory.KITCHEN: [
                "High storage", "Spill areas",
                "Slippery floors"
            ],
            HazardCategory.EXTERNAL: [
                "Pathway hazards", "Steps",
                "Lighting", "Doormats"
            ],
            HazardCategory.GENERAL: [
                "Mobility aids", "Reaching hazards",
                "Pets", "Footwear"
            ],
        }

        for category in categories:
            examples = category_examples.get(category, [])
            if examples:
                parts.append(f"\n### {category.value.upper()}")
                parts.append(", ".join(examples))

        return "\n".join(parts)

    def _get_severity_guidelines(self) -> str:
        """Get severity classification guidelines."""
        return """## Severity Classification

- **LOW**: Minor hazard, slightly increases fall risk
  - Example: Slightly dim lighting, small secured rug

- **MEDIUM**: Moderate hazard, noticeable increase in fall risk
  - Example: Rug without non-slip backing, cluttered area

- **HIGH**: Significant hazard, substantially increases fall risk
  - Example: Cord across walkway, wet floor

- **CRITICAL**: Severe hazard, immediate fall risk
  - Example: Missing stair handrail, no grab bars in shower"""

    def _get_output_format(self) -> str:
        """Get output format specification."""
        return """## Required Output Format

Respond with valid JSON in this exact format:

{
  "room_type": "living_room|bedroom|bathroom|kitchen|hallway|stairs|external|other",
  "overall_confidence": 0.0-1.0,
  "hazards": [
    {
      "category": "flooring|lighting|obstacles|furniture|stairs|bathroom|kitchen|bedroom|external|general",
      "subcategory": "specific_hazard_type",
      "description": "Detailed description of the hazard",
      "severity": "low|medium|high|critical",
      "confidence": 0.0-1.0,
      "location_description": "Where in the image",
      "bounding_box": null,
      "recommendations": ["Action 1", "Action 2"]
    }
  ]
}

Include ALL hazards you identify, even if confidence is moderate.
Be specific in descriptions - reference actual visible objects."""

    def _merge_hazards(
        self,
        hazards: List[DetectedHazard]
    ) -> List[DetectedHazard]:
        """
        Merge and deduplicate hazards from multiple passes.

        Uses category + subcategory as key for deduplication.
        Takes highest confidence version of duplicates.
        Merges recommendations from all versions.
        """
        # Group by category + subcategory
        grouped: Dict[str, List[DetectedHazard]] = {}

        for hazard in hazards:
            key = f"{hazard.category}_{hazard.subcategory}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(hazard)

        # Merge each group
        merged = []
        for key, group in grouped.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Take highest confidence detection
                best = max(group, key=lambda h: h.confidence)

                # Merge all recommendations
                all_recommendations = set()
                for h in group:
                    all_recommendations.update(h.recommendations)

                # Create new hazard with merged data
                merged_hazard = DetectedHazard(
                    category=best.category,
                    subcategory=best.subcategory,
                    description=best.description,
                    severity=best.severity,
                    confidence=best.confidence,
                    location_description=best.location_description,
                    bounding_box=best.bounding_box,
                    recommendations=list(all_recommendations)[:5]
                )
                merged.append(merged_hazard)

        return merged

    def detect_quick(
        self,
        image: ImageInput,
        room_type: Optional[str] = None
    ) -> HazardDetectionResult:
        """
        Quick detection with single pass and no chain-of-thought.

        Faster but potentially less accurate than full detection.

        Args:
            image: Image to analyze
            room_type: Optional room type hint

        Returns:
            HazardDetectionResult
        """
        config = DetectionConfig(
            use_chain_of_thought=False,
            multi_pass=False,
            confidence_threshold=0.4,
            room_type_hint=room_type,
            temperature=0.2,
        )
        return self.detect_hazards(image, config)
