"""
OpenAI GPT-4 Vision Model Integration

Implements the BaseVisionModel interface for OpenAI's GPT-4 Vision API,
supporting both raw text and structured output responses.
"""

import os
import time
import json
from typing import Any, Dict, Optional

from openai import OpenAI
from pydantic import BaseModel

from .base_model import BaseVisionModel, ImageInput, HazardDetectionResult


class GPT4VisionModel(BaseVisionModel):
    """
    GPT-4 Vision API implementation for hazard detection.

    Supports:
    - Standard chat completions with images
    - Structured output using response_format
    - Multiple image detail levels (low, high, auto)

    Attributes:
        client: OpenAI client instance
        model_id: Model identifier (e.g., "gpt-4o", "gpt-4-vision-preview")
        detail_level: Image processing detail ("low", "high", "auto")
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GPT-4 Vision model.

        Args:
            config: Configuration dictionary containing:
                - api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
                - model_id: Model to use (default: "gpt-4o")
                - detail: Image detail level (default: "high")
                - organization: Optional organization ID
        """
        super().__init__(config)

        # Get API key from config or environment
        api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Provide in config or set OPENAI_API_KEY env var."
            )

        # Initialize client
        self.client = OpenAI(
            api_key=api_key,
            organization=config.get("organization"),
        )

        self.model_id = config.get("model_id", "gpt-4o")
        self.detail_level = config.get("detail", "high")
        self.model_name = self.model_id
        self._initialized = True

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
            image: ImageInput object with the image to analyze
            prompt: Analysis prompt/instructions
            temperature: Sampling temperature (0-1, lower = more deterministic)
            max_tokens: Maximum response tokens

        Returns:
            Raw text response from the model

        Raises:
            ValueError: If image format is invalid
            OpenAI API errors on connection/rate limit issues
        """
        # Build image content
        image_content = self._build_image_content(image)

        # Make API call
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "system",
                    "content": self._get_system_prompt()
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        image_content
                    ]
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    def analyze_image_structured(
        self,
        image: ImageInput,
        prompt: str,
        response_schema: type[BaseModel],
        temperature: float = 0.1
    ) -> BaseModel:
        """
        Analyze an image and return structured output.

        Uses OpenAI's structured output feature to return a validated
        Pydantic model instance.

        Args:
            image: ImageInput object with the image to analyze
            prompt: Analysis prompt/instructions
            response_schema: Pydantic model class for response validation
            temperature: Sampling temperature (0-1)

        Returns:
            Validated Pydantic model instance matching response_schema

        Note:
            For models that don't support structured output natively,
            falls back to JSON parsing from text response.
        """
        # Build image content
        image_content = self._build_image_content(image)

        start_time = time.time()

        try:
            # Try using the parse method for structured output (OpenAI SDK v1.x)
            completion = self.client.beta.chat.completions.parse(
                model=self.model_id,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            image_content
                        ]
                    }
                ],
                response_format=response_schema,
                temperature=temperature,
            )

            processing_time = (time.time() - start_time) * 1000

            result = completion.choices[0].message.parsed

            # Add metadata if the schema supports it
            if hasattr(result, 'processing_time_ms'):
                result.processing_time_ms = processing_time
            if hasattr(result, 'model_name'):
                result.model_name = self.model_id

            return result

        except Exception as e:
            # Fallback: Get JSON response and parse manually
            return self._fallback_structured_output(
                image, prompt, response_schema, temperature, start_time
            )

    def _fallback_structured_output(
        self,
        image: ImageInput,
        prompt: str,
        response_schema: type[BaseModel],
        temperature: float,
        start_time: float
    ) -> BaseModel:
        """
        Fallback method for structured output when parse() is not available.

        Adds JSON schema to prompt and parses response manually.
        """
        # Enhance prompt with JSON schema requirement
        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        enhanced_prompt = f"""{prompt}

IMPORTANT: You MUST respond with valid JSON that exactly matches this schema:
{schema_json}

Do not include any text before or after the JSON. Start directly with {{ and end with }}."""

        # Get raw response
        raw_response = self.analyze_image(image, enhanced_prompt, temperature)

        processing_time = (time.time() - start_time) * 1000

        # Extract and parse JSON
        json_str = self._extract_json(raw_response)
        data = json.loads(json_str)

        # Validate with Pydantic
        result = response_schema.model_validate(data)

        # Add metadata
        if hasattr(result, 'processing_time_ms'):
            result.processing_time_ms = processing_time
        if hasattr(result, 'model_name'):
            result.model_name = self.model_id
        if hasattr(result, 'raw_response'):
            result.raw_response = raw_response

        return result

    def _build_image_content(self, image: ImageInput) -> Dict[str, Any]:
        """
        Build the image content dictionary for the API request.

        Args:
            image: ImageInput object

        Returns:
            Dictionary formatted for OpenAI API image content
        """
        if image.url:
            return {
                "type": "image_url",
                "image_url": {
                    "url": image.url,
                    "detail": self.detail_level
                }
            }
        else:
            # Convert to base64 data URI
            data_uri = image.get_data_uri()
            return {
                "type": "image_url",
                "image_url": {
                    "url": data_uri,
                    "detail": self.detail_level
                }
            }

    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from model response.

        Handles cases where JSON is wrapped in markdown code blocks
        or has surrounding text.

        Args:
            text: Raw model response

        Returns:
            Extracted JSON string

        Raises:
            ValueError: If no valid JSON can be extracted
        """
        # Try to find JSON in code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip()

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            candidate = text[start:end].strip()
            if candidate.startswith("{"):
                return candidate

        # Try to find raw JSON
        if "{" in text:
            start = text.find("{")
            # Find matching closing brace
            depth = 0
            for i, char in enumerate(text[start:], start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]

        raise ValueError("Could not extract JSON from response")

    def health_check(self) -> bool:
        """
        Verify model connectivity and availability.

        Returns:
            True if the API is accessible, False otherwise
        """
        try:
            # Simple API call to verify connectivity
            self.client.models.list()
            return True
        except Exception:
            return False

    def get_token_usage(self, response) -> Dict[str, int]:
        """
        Extract token usage from API response.

        Args:
            response: OpenAI API response object

        Returns:
            Dictionary with prompt_tokens, completion_tokens, total_tokens
        """
        if hasattr(response, 'usage') and response.usage:
            return {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def create_gpt4v_model(
    api_key: Optional[str] = None,
    model_id: str = "gpt-4o",
    detail: str = "high"
) -> GPT4VisionModel:
    """
    Factory function to create GPT-4 Vision model instance.

    Args:
        api_key: OpenAI API key (uses env var if not provided)
        model_id: Model to use
        detail: Image detail level

    Returns:
        Configured GPT4VisionModel instance
    """
    config = {
        "api_key": api_key,
        "model_id": model_id,
        "detail": detail,
    }
    return GPT4VisionModel(config)
