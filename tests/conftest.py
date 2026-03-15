"""
Pytest configuration and shared fixtures for Fall Risk Detection tests.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_hazard():
    """Sample detected hazard for testing."""
    return {
        "category": "bathroom",
        "subcategory": "bath_no_grab_bars",
        "severity": "critical",
        "description": "No grab bars near bathtub",
        "location": "bathtub area",
        "confidence": 0.92,
        "westmead_reference": "2.1"
    }


@pytest.fixture
def sample_hazard_list():
    """List of sample hazards for testing."""
    return [
        {
            "category": "bathroom",
            "subcategory": "bath_no_grab_bars",
            "severity": "critical",
            "description": "No grab bars near bathtub",
            "confidence": 0.92
        },
        {
            "category": "flooring",
            "subcategory": "loose_rug",
            "severity": "high",
            "description": "Loose rug near bathroom entrance",
            "confidence": 0.85
        },
        {
            "category": "lighting",
            "subcategory": "dim_lighting",
            "severity": "medium",
            "description": "Insufficient lighting in hallway",
            "confidence": 0.78
        }
    ]


@pytest.fixture
def sample_ground_truth():
    """Sample ground truth annotations for testing."""
    return [
        {
            "category": "bathroom",
            "subcategory": "bath_no_grab_bars",
            "severity": "critical",
            "location": "bathtub area"
        },
        {
            "category": "flooring",
            "subcategory": "loose_rug",
            "severity": "high",
            "location": "bathroom entrance"
        },
        {
            "category": "stairs",
            "subcategory": "no_handrails",
            "severity": "critical",
            "location": "main staircase"
        }
    ]


@pytest.fixture
def sample_predictions():
    """Sample model predictions for testing."""
    return [
        {
            "category": "bathroom",
            "subcategory": "bath_no_grab_bars",
            "severity": "critical",
            "confidence": 0.92
        },
        {
            "category": "flooring",
            "subcategory": "loose_rug",
            "severity": "medium",  # Wrong severity
            "confidence": 0.85
        },
        {
            "category": "obstacles",  # False positive
            "subcategory": "floor_clutter",
            "severity": "low",
            "confidence": 0.65
        }
    ]


@pytest.fixture
def sample_annotation():
    """Sample annotation dictionary for testing."""
    return {
        "image_id": "bathroom_001",
        "room_type": "bathroom",
        "source": "staged",
        "hazards": [
            {
                "category": "bathroom",
                "subcategory": "bath_no_grab_bars",
                "severity": "critical",
                "location": "bathtub area",
                "annotator": "expert_1"
            },
            {
                "category": "flooring",
                "subcategory": "slippery_surface",
                "severity": "medium",
                "location": "bathroom floor",
                "annotator": "expert_1"
            }
        ]
    }


@pytest.fixture
def mock_image_path(tmp_path):
    """Create a mock image file for testing."""
    import numpy as np

    # Try to create actual image if OpenCV available
    try:
        import cv2
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path = tmp_path / "test_image.jpg"
        cv2.imwrite(str(img_path), img)
        return img_path
    except ImportError:
        # Create placeholder file
        img_path = tmp_path / "test_image.jpg"
        img_path.write_bytes(b"fake image data")
        return img_path
