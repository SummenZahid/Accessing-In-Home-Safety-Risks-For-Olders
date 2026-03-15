"""
Tests for Image Preprocessing Module

Tests the image preprocessing functionality including:
- Image quality assessment
- Resolution handling
- Brightness/contrast adjustments
- Batch processing
"""

import pytest
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Try to import preprocessing - may fail without OpenCV
try:
    from preprocessing.image_processor import (
        ImageProcessor,
        ImageQuality,
        ProcessingResult,
        BatchProcessor
    )
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False


@pytest.mark.skipif(not PREPROCESSING_AVAILABLE, reason="OpenCV not available")
class TestImageProcessor:
    """Test ImageProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a processor instance."""
        return ImageProcessor()

    @pytest.fixture
    def test_image(self, tmp_path):
        """Create a test image file."""
        import cv2
        import numpy as np

        # Create a simple test image
        img = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), img)
        return img_path

    def test_process_valid_image(self, processor, test_image):
        """Should process a valid image successfully."""
        result = processor.process(test_image)

        assert result is not None
        assert result.success
        assert result.quality != ImageQuality.REJECTED

    def test_process_nonexistent_image(self, processor):
        """Should handle nonexistent image gracefully."""
        result = processor.process(Path("/nonexistent/image.jpg"))

        assert not result.success

    def test_quality_assessment(self, processor, test_image):
        """Should assess image quality."""
        result = processor.validate_only(test_image)

        assert result is not None
        assert hasattr(result, 'quality')
        assert result.quality in list(ImageQuality)

    def test_resize_large_image(self, processor, tmp_path):
        """Large images should be resized."""
        import cv2
        import numpy as np

        # Create large image
        large_img = np.random.randint(0, 255, (3000, 4000, 3), dtype=np.uint8)
        img_path = tmp_path / "large.jpg"
        cv2.imwrite(str(img_path), large_img)

        result = processor.process(img_path, max_dimension=1024)

        # Result should be resized
        assert result.processed_dimensions[0] <= 1024
        assert result.processed_dimensions[1] <= 1024

    def test_base64_encoding(self, processor, test_image):
        """Should encode image to base64."""
        result = processor.process(test_image)
        base64_str = processor.to_base64(result.processed_path or test_image)

        assert base64_str is not None
        assert len(base64_str) > 0
        # Base64 strings contain alphanumeric chars, + / =
        assert all(c.isalnum() or c in '+/=' for c in base64_str)


@pytest.mark.skipif(not PREPROCESSING_AVAILABLE, reason="OpenCV not available")
class TestImageQuality:
    """Test image quality assessment."""

    def test_quality_enum_values(self):
        """Quality enum should have expected values."""
        assert ImageQuality.EXCELLENT.value > ImageQuality.GOOD.value
        assert ImageQuality.GOOD.value > ImageQuality.ACCEPTABLE.value
        assert ImageQuality.ACCEPTABLE.value > ImageQuality.POOR.value
        assert ImageQuality.POOR.value > ImageQuality.REJECTED.value

    def test_quality_comparison(self):
        """Quality levels should be comparable."""
        assert ImageQuality.EXCELLENT != ImageQuality.REJECTED


@pytest.mark.skipif(not PREPROCESSING_AVAILABLE, reason="OpenCV not available")
class TestBatchProcessor:
    """Test batch processing functionality."""

    @pytest.fixture
    def batch_processor(self):
        """Create a batch processor."""
        return BatchProcessor()

    @pytest.fixture
    def test_images(self, tmp_path):
        """Create multiple test images."""
        import cv2
        import numpy as np

        paths = []
        for i in range(3):
            img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            img_path = tmp_path / f"test_{i}.jpg"
            cv2.imwrite(str(img_path), img)
            paths.append(img_path)
        return paths

    def test_batch_process(self, batch_processor, test_images):
        """Should process multiple images."""
        results = batch_processor.process_batch(test_images)

        assert len(results) == len(test_images)
        assert all(r.success for r in results)

    def test_batch_with_invalid(self, batch_processor, test_images):
        """Should handle mixed valid/invalid images."""
        # Add invalid path
        paths = test_images + [Path("/invalid/path.jpg")]
        results = batch_processor.process_batch(paths)

        assert len(results) == len(paths)
        # Last one should fail
        assert not results[-1].success


class TestDataLoader:
    """Test data loader functionality."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary data directory structure."""
        # Create subdirectories
        (tmp_path / "sample").mkdir()
        (tmp_path / "annotations").mkdir()
        return tmp_path

    def test_loader_import(self):
        """Data loader should be importable."""
        from preprocessing.data_loader import DataLoader
        assert DataLoader is not None

    def test_loader_initialization(self, temp_data_dir):
        """Should initialize with data directory."""
        from preprocessing.data_loader import DataLoader

        loader = DataLoader(temp_data_dir)
        assert loader is not None

    def test_empty_dataset(self, temp_data_dir):
        """Should handle empty dataset."""
        from preprocessing.data_loader import DataLoader

        loader = DataLoader(temp_data_dir)
        assert len(loader) == 0

    def test_dataset_stats(self, temp_data_dir):
        """Should calculate dataset statistics."""
        from preprocessing.data_loader import DataLoader

        loader = DataLoader(temp_data_dir)
        stats = loader.get_stats()

        assert hasattr(stats, 'total_images')
        assert stats.total_images == 0


class TestAnnotationValidation:
    """Test annotation validation."""

    def test_validate_valid_annotation(self):
        """Valid annotation should pass validation."""
        from preprocessing.data_loader import validate_annotation

        annotation = {
            "image_id": "test_001",
            "hazards": [
                {
                    "category": "bathroom",
                    "subcategory": "no_grab_bars",
                    "severity": "critical"
                }
            ]
        }

        is_valid, errors = validate_annotation(annotation)
        assert is_valid
        assert len(errors) == 0

    def test_validate_missing_fields(self):
        """Missing required fields should fail validation."""
        from preprocessing.data_loader import validate_annotation

        annotation = {"hazards": []}  # Missing image_id

        is_valid, errors = validate_annotation(annotation)
        # Should have error about missing image_id
        assert not is_valid or any("image_id" in e for e in errors)

    def test_validate_invalid_category(self):
        """Invalid category should fail validation."""
        from preprocessing.data_loader import validate_annotation

        annotation = {
            "image_id": "test_001",
            "hazards": [
                {
                    "category": "invalid_category",
                    "severity": "high"
                }
            ]
        }

        is_valid, errors = validate_annotation(annotation)
        assert not is_valid
        assert any("category" in e.lower() for e in errors)

    def test_validate_invalid_severity(self):
        """Invalid severity should fail validation."""
        from preprocessing.data_loader import validate_annotation

        annotation = {
            "image_id": "test_001",
            "hazards": [
                {
                    "category": "bathroom",
                    "severity": "invalid_severity"
                }
            ]
        }

        is_valid, errors = validate_annotation(annotation)
        assert not is_valid
        assert any("severity" in e.lower() for e in errors)


class TestAnnotationTemplate:
    """Test annotation template generation."""

    def test_create_template(self):
        """Should create annotation template."""
        from preprocessing.data_loader import create_annotation_template

        template = create_annotation_template()

        assert "image_id" in template
        assert "room_type" in template
        assert "hazards" in template
        assert isinstance(template["hazards"], list)
