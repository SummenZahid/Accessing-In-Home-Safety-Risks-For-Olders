"""
Vision Model Factory

Factory pattern implementation for creating vision-language model instances.
Supports multiple backends (OpenAI, Google Gemini, local models).
"""

from enum import Enum
from typing import Dict, Any, Optional

from .base_model import BaseVisionModel


class ModelType(Enum):
    """Supported vision model types."""
    OPENAI = "openai"
    GPT4V = "gpt4v"
    GEMINI = "gemini"
    LOCAL = "local"


class VisionModelFactory:
    """
    Factory for creating vision model instances.

    Provides a unified interface for instantiating different vision model
    backends based on configuration or user preference.

    Usage:
        # Create using enum
        model = VisionModelFactory.create(ModelType.OPENAI, config)

        # Create using string
        model = VisionModelFactory.create_from_string("openai", config)

        # Auto-detect from config
        model = VisionModelFactory.create_auto(config)
    """

    @classmethod
    def create(
        cls,
        model_type: ModelType,
        config: Dict[str, Any]
    ) -> BaseVisionModel:
        """
        Create a vision model instance based on type.

        Args:
            model_type: Type of model to create
            config: Model configuration dictionary

        Returns:
            Configured vision model instance

        Raises:
            ValueError: If model type is not supported
            ImportError: If required dependencies are missing
        """
        if model_type in (ModelType.OPENAI, ModelType.GPT4V):
            from .gpt4v_model import GPT4VisionModel
            return GPT4VisionModel(config)

        elif model_type == ModelType.GEMINI:
            from .gemini_model import GeminiVisionModel
            return GeminiVisionModel(config)

        elif model_type == ModelType.LOCAL:
            raise NotImplementedError(
                "Local model support not yet implemented. "
                "Consider using LLaVA via Ollama for local inference."
            )

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @classmethod
    def create_from_string(
        cls,
        model_type: str,
        config: Dict[str, Any]
    ) -> BaseVisionModel:
        """
        Create a vision model instance from string type name.

        Args:
            model_type: String name of model type (e.g., "openai", "gemini")
            config: Model configuration dictionary

        Returns:
            Configured vision model instance
        """
        try:
            enum_type = ModelType(model_type.lower())
        except ValueError:
            # Try mapping common aliases
            aliases = {
                "gpt4": ModelType.OPENAI,
                "gpt-4": ModelType.OPENAI,
                "gpt4v": ModelType.GPT4V,
                "gpt-4v": ModelType.GPT4V,
                "gpt4-vision": ModelType.OPENAI,
                "gpt-4o": ModelType.OPENAI,
                "google": ModelType.GEMINI,
                "gemini-flash": ModelType.GEMINI,
                "gemini-pro": ModelType.GEMINI,
                "llava": ModelType.LOCAL,
                "ollama": ModelType.LOCAL,
            }
            enum_type = aliases.get(model_type.lower())
            if enum_type is None:
                raise ValueError(
                    f"Unknown model type: {model_type}. "
                    f"Supported types: {[m.value for m in ModelType]}"
                )

        return cls.create(enum_type, config)

    @classmethod
    def create_auto(cls, config: Dict[str, Any]) -> BaseVisionModel:
        """
        Auto-detect and create the appropriate model from config.

        Checks for API keys in config or environment variables to
        determine which model to use.

        Args:
            config: Configuration dictionary that may contain:
                - model_type: Explicit model type preference
                - api_key / openai_api_key: OpenAI API key
                - google_api_key: Google API key

        Returns:
            Configured vision model instance

        Raises:
            ValueError: If no suitable model can be determined
        """
        import os

        # Check for explicit model type preference
        if "model_type" in config:
            return cls.create_from_string(config["model_type"], config)

        # Auto-detect based on available API keys
        openai_key = (
            config.get("api_key") or
            config.get("openai_api_key") or
            os.getenv("OPENAI_API_KEY")
        )
        google_key = (
            config.get("google_api_key") or
            os.getenv("GOOGLE_API_KEY")
        )

        if openai_key:
            config["api_key"] = openai_key
            return cls.create(ModelType.OPENAI, config)

        elif google_key:
            config["api_key"] = google_key
            return cls.create(ModelType.GEMINI, config)

        else:
            raise ValueError(
                "No API key found. Please provide either:\n"
                "- OPENAI_API_KEY environment variable or 'api_key' in config\n"
                "- GOOGLE_API_KEY environment variable or 'google_api_key' in config"
            )

    @classmethod
    def available_models(cls) -> list:
        """
        List available model types.

        Returns:
            List of ModelType enum values
        """
        return list(ModelType)

    @classmethod
    def get_default_config(cls, model_type: ModelType) -> Dict[str, Any]:
        """
        Get default configuration for a model type.

        Args:
            model_type: Type of model

        Returns:
            Default configuration dictionary
        """
        defaults = {
            ModelType.OPENAI: {
                "model_id": "gpt-4o",
                "detail": "high",
            },
            ModelType.GPT4V: {
                "model_id": "gpt-4o",
                "detail": "high",
            },
            ModelType.GEMINI: {
                "model_id": "gemini-2.0-flash",
            },
            ModelType.LOCAL: {
                "model_id": "llava:13b",
                "backend": "ollama",
                "base_url": "http://localhost:11434",
            },
        }
        return defaults.get(model_type, {})
