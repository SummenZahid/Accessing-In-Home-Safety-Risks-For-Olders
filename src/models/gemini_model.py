"""
Google Gemini Vision Model Integration

Implements the BaseVisionModel interface for Google's Gemini API,
supporting both raw text and structured output responses.
"""

import os
import time
import json
from typing import Any, Dict, Optional
from pathlib import Path

from pydantic import BaseModel

# Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from .base_model import BaseVisionModel, ImageInput, HazardDetectionResult


class GeminiVisionModel(BaseVisionModel):
    """
    Google Gemini Vision API implementation for hazard detection.

    Supports:
    - Gemini 2.0 Flash and Pro models
    - Image analysis with text prompts
    - Structured JSON output
    - File uploads for larger images

    Attributes:
        client: Google GenAI client instance
        model_id: Model identifier (e.g., "gemini-2.0-flash", "gemini-2.0-pro")
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Gemini Vision model.

        Args:
            config: Configuration dictionary containing:
                - api_key: Google API key (or uses GOOGLE_API_KEY env var)
                - model_id: Model to use (default: "gemini-2.0-flash")

        Raises:
            ImportError: If google-genai package is not installed
            ValueError: If API key is not provided
        """
        super().__init__(config)

        if not GENAI_AVAILABLE:
            raise ImportError(
                "Google GenAI package not installed. "
                "Install with: pip install google-genai"
            )

        # Get API key from config or environment
        api_key = config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key required. Provide in config or set GOOGLE_API_KEY env var."
            )

        # Initialize client
        self.client = genai.Client(api_key=api_key)

        self.model_id = config.get("model_id", "gemini-2.0-flash")
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
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum response tokens

        Returns:
            Raw text response from the model
        """
        # Build content list
        contents = self._build_contents(image, prompt)

        # Make API call
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=self._get_system_prompt(),
            )
        )

        return response.text

    def analyze_image_structured(
        self,
        image: ImageInput,
        prompt: str,
        response_schema: type[BaseModel],
        temperature: float = 0.1
    ) -> BaseModel:
        """
        Analyze an image and return structured JSON output.

        Uses Gemini's native JSON mode with schema enforcement.

        Args:
            image: ImageInput object with the image to analyze
            prompt: Analysis prompt/instructions
            response_schema: Pydantic model class for response validation
            temperature: Sampling temperature (0-1)

        Returns:
            Validated Pydantic model instance matching response_schema
        """
        # Build content list
        contents = self._build_contents(image, prompt)

        start_time = time.time()

        try:
            # Use Gemini's structured output with JSON schema
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type='application/json',
                    response_schema=response_schema,
                    system_instruction=self._get_system_prompt(),
                )
            )

            processing_time = (time.time() - start_time) * 1000

            # Parse the response
            if hasattr(response, 'parsed') and response.parsed:
                result = response.parsed
            else:
                # Manual parsing if parsed attribute not available
                json_str = response.text
                data = json.loads(json_str)
                result = response_schema.model_validate(data)

            # Add metadata
            if hasattr(result, 'processing_time_ms'):
                result.processing_time_ms = processing_time
            if hasattr(result, 'model_name'):
                result.model_name = self.model_id

            return result

        except Exception as e:
            # Fallback to manual JSON parsing
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
        Fallback method for structured output.

        Adds JSON schema requirement to prompt and parses manually.
        """
        # Enhance prompt with JSON schema
        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        enhanced_prompt = f"""{prompt}

CRITICAL: Respond ONLY with valid JSON matching this exact schema:
{schema_json}

Start directly with {{ and end with }}. No other text."""

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

    def _build_contents(self, image: ImageInput, prompt: str) -> list:
        """
        Build the contents list for the API request.

        Args:
            image: ImageInput object
            prompt: Text prompt

        Returns:
            List of content parts for the API
        """
        contents = []

        # Add image
        if image.path:
            # For local files, we can upload or embed inline
            path = Path(image.path)
            if path.exists():
                # Read and encode the image
                with open(path, "rb") as f:
                    image_data = f.read()

                contents.append(
                    types.Part.from_bytes(
                        data=image_data,
                        mime_type=image.mime_type,
                    )
                )
        elif image.base64_data:
            import base64
            image_bytes = base64.b64decode(image.base64_data)
            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=image.mime_type,
                )
            )
        elif image.url:
            # For URLs, include as URI reference
            contents.append(
                types.Part.from_uri(
                    file_uri=image.url,
                    mime_type=image.mime_type,
                )
            )

        # Add text prompt
        contents.append(prompt)

        return contents

    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from model response.

        Args:
            text: Raw model response

        Returns:
            Extracted JSON string
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
            # Simple test to verify API connectivity
            models = self.client.models.list()
            return True
        except Exception:
            return False

    def upload_file(self, file_path: str) -> str:
        """
        Upload a file to Google's servers for use in generation.

        Useful for larger files or when reusing the same image multiple times.

        Args:
            file_path: Path to the file to upload

        Returns:
            URI of the uploaded file
        """
        uploaded = self.client.files.upload(file=file_path)
        return uploaded.uri


def create_gemini_model(
    api_key: Optional[str] = None,
    model_id: str = "gemini-2.0-flash"
) -> GeminiVisionModel:
    """
    Factory function to create Gemini Vision model instance.

    Args:
        api_key: Google API key (uses env var if not provided)
        model_id: Model to use

    Returns:
        Configured GeminiVisionModel instance
    """
    config = {
        "api_key": api_key,
        "model_id": model_id,
    }
    return GeminiVisionModel(config)
