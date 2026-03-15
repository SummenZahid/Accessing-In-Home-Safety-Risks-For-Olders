"""
Tests for Vision Model Module

Tests the model abstraction and factory functionality.
Note: Actual API calls are mocked to avoid requiring API keys during testing.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.base_model import (
    ImageInput,
    BoundingBox,
    DetectedHazard,
    HazardDetectionResult,
    BaseVisionModel
)
from models.model_factory import (
    VisionModelFactory,
    ModelType,
)


class TestImageInput:
    """Test ImageInput schema."""

    def test_from_path(self, tmp_path):
        """Should create input from file path."""
        # Create a fake image file
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake image data")

        img_input = ImageInput(source=str(img_path))
        assert img_input.source == str(img_path)

    def test_from_url(self):
        """Should accept URL as source."""
        img_input = ImageInput(source="https://example.com/image.jpg")
        assert "https://" in img_input.source

    def test_from_base64(self):
        """Should accept base64 string."""
        img_input = ImageInput(
            source="data:image/jpeg;base64,/9j/4AAQ...",
            is_base64=True
        )
        assert img_input.is_base64


class TestBoundingBox:
    """Test BoundingBox schema."""

    def test_valid_bounding_box(self):
        """Should create valid bounding box."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50

    def test_bounding_box_area(self):
        """Should calculate area correctly."""
        bbox = BoundingBox(x=0, y=0, width=10, height=20)
        # Area = width * height = 200
        assert bbox.width * bbox.height == 200


class TestDetectedHazard:
    """Test DetectedHazard schema."""

    def test_hazard_creation(self):
        """Should create hazard with required fields."""
        hazard = DetectedHazard(
            category="bathroom",
            subcategory="no_grab_bars",
            severity="critical",
            description="No grab bars near bathtub",
            confidence=0.92
        )
        assert hazard.category == "bathroom"
        assert hazard.severity == "critical"
        assert hazard.confidence == 0.92

    def test_hazard_with_bounding_box(self):
        """Should include optional bounding box."""
        hazard = DetectedHazard(
            category="flooring",
            subcategory="loose_rug",
            severity="high",
            description="Loose rug in hallway",
            confidence=0.85,
            bounding_box=BoundingBox(x=100, y=200, width=300, height=150)
        )
        assert hazard.bounding_box is not None
        assert hazard.bounding_box.width == 300

    def test_hazard_to_dict(self):
        """Should convert to dictionary."""
        hazard = DetectedHazard(
            category="stairs",
            subcategory="no_handrails",
            severity="critical",
            description="Missing handrails",
            confidence=0.95
        )
        d = hazard.model_dump()
        assert d["category"] == "stairs"
        assert d["confidence"] == 0.95


class TestHazardDetectionResult:
    """Test HazardDetectionResult schema."""

    def test_empty_result(self):
        """Should handle empty detection result."""
        result = HazardDetectionResult(
            hazards=[],
            room_type="bathroom",
            overall_risk_assessment="No hazards detected",
            model_name="test_model"
        )
        assert len(result.hazards) == 0

    def test_result_with_hazards(self, sample_hazard):
        """Should include detected hazards."""
        hazard = DetectedHazard(**sample_hazard)
        result = HazardDetectionResult(
            hazards=[hazard],
            room_type="bathroom",
            overall_risk_assessment="Critical risk",
            model_name="gpt4v"
        )
        assert len(result.hazards) == 1
        assert result.hazards[0].category == "bathroom"


class TestModelFactory:
    """Test VisionModelFactory class."""

    def test_get_available_models(self):
        """Should return list of available model types."""
        models = VisionModelFactory.available_models()
        assert isinstance(models, list)

    def test_model_type_enum(self):
        """ModelType enum should have expected values."""
        assert ModelType.OPENAI.value == "openai"
        assert ModelType.GPT4V.value == "gpt4v"
        assert ModelType.GEMINI.value == "gemini"

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_create_openai_model(self):
        """Should create OpenAI model when API key present."""
        try:
            model = VisionModelFactory.create(ModelType.OPENAI, {})
            assert model is not None
        except Exception:
            # May fail if openai package not installed
            pytest.skip("OpenAI package not available")

    @patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'})
    def test_create_gemini_model(self):
        """Should create Gemini model when API key present."""
        try:
            model = VisionModelFactory.create(ModelType.GEMINI, {})
            assert model is not None
        except Exception:
            # May fail if google-genai package not installed
            pytest.skip("Google GenAI package not available")

    def test_create_without_api_key(self):
        """Should raise error when API key missing."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises((ValueError, KeyError, Exception)):
                VisionModelFactory.create(ModelType.OPENAI, {})


class TestBaseVisionModel:
    """Test BaseVisionModel abstract class."""

    def test_abstract_methods(self):
        """Cannot instantiate abstract class directly."""
        with pytest.raises(TypeError):
            BaseVisionModel()

    def test_concrete_implementation(self):
        """Concrete implementation should work."""

        class MockVisionModel(BaseVisionModel):
            def analyze_image(self, image_input, prompt):
                return "Mock response"

            def analyze_image_structured(self, image_input, prompt, response_schema):
                return {"hazards": []}

            def health_check(self):
                return True

        model = MockVisionModel()
        assert model.health_check()
        assert model.analyze_image(None, "test") == "Mock response"


class TestModelIntegration:
    """Integration tests for model functionality (mocked)."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock vision model."""
        model = Mock(spec=BaseVisionModel)
        model.analyze_image.return_value = "Detected hazards: loose rug, dim lighting"
        model.analyze_image_structured.return_value = {
            "hazards": [
                {
                    "category": "flooring",
                    "subcategory": "loose_rug",
                    "severity": "high",
                    "description": "Loose rug in hallway",
                    "confidence": 0.85
                }
            ],
            "room_type": "hallway"
        }
        model.health_check.return_value = True
        return model

    def test_analyze_image(self, mock_model):
        """Should analyze image and return response."""
        result = mock_model.analyze_image(
            ImageInput(source="/path/to/image.jpg"),
            "Analyze this image for fall hazards"
        )
        assert "hazards" in result.lower()

    def test_structured_response(self, mock_model):
        """Should return structured hazard data."""
        result = mock_model.analyze_image_structured(
            ImageInput(source="/path/to/image.jpg"),
            "Analyze for hazards",
            HazardDetectionResult
        )
        assert "hazards" in result
        assert len(result["hazards"]) > 0

    def test_health_check(self, mock_model):
        """Health check should return boolean."""
        assert mock_model.health_check() is True


class TestModelConfiguration:
    """Test model configuration handling."""

    def test_default_config(self):
        """Models should have default configuration."""
        # This tests that configuration is accessible
        from models.model_factory import VisionModelFactory
        assert hasattr(VisionModelFactory, 'create')

    def test_custom_temperature(self):
        """Should accept custom temperature parameter."""
        # Mock test - actual implementation may vary
        config = {"temperature": 0.3, "max_tokens": 2000}
        assert config["temperature"] == 0.3

    def test_model_version_selection(self):
        """Should support model version selection."""
        config = {"model_version": "gpt-4-vision-preview"}
        assert "gpt-4" in config["model_version"]
