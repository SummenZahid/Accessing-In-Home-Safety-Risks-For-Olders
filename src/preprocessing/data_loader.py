"""
Data Loader for Fall Hazard Detection System

Provides utilities for loading images, annotations, and managing
datasets for training and evaluation.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import random


class DatasetSplit(Enum):
    """Dataset split types."""
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"
    ALL = "all"


@dataclass
class ImageSample:
    """
    Represents a single image sample with its metadata and annotations.

    Attributes:
        image_id: Unique identifier for the image
        image_path: Path to the image file
        room_type: Type of room (bathroom, kitchen, etc.)
        source: Data source (staged, public_dataset, synthetic)
        annotations: List of ground truth hazard annotations
        metadata: Additional metadata about the image
    """
    image_id: str
    image_path: Path
    room_type: Optional[str] = None
    source: Optional[str] = None
    annotations: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_annotations(self) -> bool:
        """Check if sample has ground truth annotations."""
        return len(self.annotations) > 0

    def get_categories(self) -> List[str]:
        """Get list of hazard categories in annotations."""
        return list(set(
            ann.get('category', 'unknown')
            for ann in self.annotations
        ))

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'image_id': self.image_id,
            'image_path': str(self.image_path),
            'room_type': self.room_type,
            'source': self.source,
            'annotations': self.annotations,
            'metadata': self.metadata
        }


@dataclass
class DatasetStats:
    """Statistics about the dataset."""
    total_images: int = 0
    annotated_images: int = 0
    total_hazards: int = 0
    hazards_by_category: Dict[str, int] = field(default_factory=dict)
    hazards_by_severity: Dict[str, int] = field(default_factory=dict)
    images_by_room: Dict[str, int] = field(default_factory=dict)
    images_by_source: Dict[str, int] = field(default_factory=dict)


class AnnotationLoader:
    """
    Load and parse ground truth annotations from various formats.

    Supports:
    - JSON annotation files
    - COCO-style annotations
    - Simple directory-based labels
    """

    SUPPORTED_FORMATS = ['json', 'coco', 'directory']

    def __init__(self, annotation_dir: Union[str, Path]):
        """
        Initialize the annotation loader.

        Args:
            annotation_dir: Directory containing annotation files
        """
        self.annotation_dir = Path(annotation_dir)
        self._cache: Dict[str, List[Dict]] = {}

    def load(self, image_id: str) -> List[Dict]:
        """
        Load annotations for a specific image.

        Args:
            image_id: The image identifier

        Returns:
            List of hazard annotation dictionaries
        """
        if image_id in self._cache:
            return self._cache[image_id]

        # Try JSON file first
        json_path = self.annotation_dir / f"{image_id}.json"
        if json_path.exists():
            annotations = self._load_json(json_path)
            self._cache[image_id] = annotations
            return annotations

        # Try loading from combined annotations file
        combined_path = self.annotation_dir / "annotations.json"
        if combined_path.exists():
            self._load_combined_annotations(combined_path)
            return self._cache.get(image_id, [])

        return []

    def _load_json(self, path: Path) -> List[Dict]:
        """Load annotations from individual JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get('hazards', data.get('annotations', []))
        except (json.JSONDecodeError, IOError):
            return []

        return []

    def _load_combined_annotations(self, path: Path) -> None:
        """Load annotations from combined annotations file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            # COCO-style format
            if 'images' in data and 'annotations' in data:
                self._parse_coco_format(data)
            # Simple dict format {image_id: [annotations]}
            elif isinstance(data, dict):
                for image_id, annotations in data.items():
                    if isinstance(annotations, dict):
                        annotations = annotations.get('hazards', [])
                    self._cache[image_id] = annotations
        except (json.JSONDecodeError, IOError):
            pass

    def _parse_coco_format(self, data: Dict) -> None:
        """Parse COCO-style annotation format."""
        # Create image_id mapping
        id_to_name = {}
        for img in data.get('images', []):
            id_to_name[img['id']] = img.get('file_name', '').split('.')[0]

        # Group annotations by image
        for ann in data.get('annotations', []):
            image_id = id_to_name.get(ann.get('image_id'))
            if image_id:
                if image_id not in self._cache:
                    self._cache[image_id] = []

                self._cache[image_id].append({
                    'category': ann.get('category_name', ann.get('category', 'unknown')),
                    'subcategory': ann.get('subcategory', ''),
                    'severity': ann.get('severity', 'medium'),
                    'location': ann.get('location', ''),
                    'bounding_box': ann.get('bbox')
                })

    def load_all(self) -> Dict[str, List[Dict]]:
        """Load all available annotations."""
        # Load combined file if exists
        combined_path = self.annotation_dir / "annotations.json"
        if combined_path.exists():
            self._load_combined_annotations(combined_path)

        # Load individual JSON files
        for json_file in self.annotation_dir.glob("*.json"):
            if json_file.name != "annotations.json":
                image_id = json_file.stem
                if image_id not in self._cache:
                    self._cache[image_id] = self._load_json(json_file)

        return self._cache


class DataLoader:
    """
    Main data loader for the fall hazard detection system.

    Loads images and annotations from the data directory,
    provides iteration and batching capabilities.

    Usage:
        loader = DataLoader('/path/to/data')

        # Iterate through samples
        for sample in loader:
            process(sample)

        # Get batches
        for batch in loader.batches(batch_size=8):
            process_batch(batch)

        # Filter by room type
        bathroom_samples = loader.filter(room_type='bathroom')
    """

    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    def __init__(
        self,
        data_dir: Union[str, Path],
        annotation_dir: Optional[Union[str, Path]] = None,
        split: DatasetSplit = DatasetSplit.ALL
    ):
        """
        Initialize the data loader.

        Args:
            data_dir: Root directory containing image data
            annotation_dir: Directory with annotations (defaults to data_dir/annotations)
            split: Which split to load (train, val, test, all)
        """
        self.data_dir = Path(data_dir)
        self.annotation_dir = Path(annotation_dir) if annotation_dir else self.data_dir / "annotations"
        self.split = split

        self._samples: List[ImageSample] = []
        self._index = 0

        # Initialize annotation loader
        self.annotation_loader = AnnotationLoader(self.annotation_dir)

        # Load samples
        self._load_samples()

    def _load_samples(self) -> None:
        """Discover and load all image samples."""
        self._samples = []

        # Check for sample directory
        sample_dir = self.data_dir / "sample"
        if sample_dir.exists():
            self._scan_directory(sample_dir, source='sample')

        # Check for split directories
        if self.split == DatasetSplit.ALL:
            for split in ['train', 'val', 'test']:
                split_dir = self.data_dir / split
                if split_dir.exists():
                    self._scan_directory(split_dir, source=split)
        else:
            split_dir = self.data_dir / self.split.value
            if split_dir.exists():
                self._scan_directory(split_dir, source=self.split.value)

        # Also scan root data directory for images
        self._scan_directory(self.data_dir, source='root', recursive=False)

    def _scan_directory(
        self,
        directory: Path,
        source: str,
        recursive: bool = True
    ) -> None:
        """Scan directory for image files."""
        pattern = "**/*" if recursive else "*"

        for file_path in directory.glob(pattern):
            if file_path.suffix.lower() in self.SUPPORTED_IMAGE_FORMATS:
                # Skip if already added
                if any(s.image_path == file_path for s in self._samples):
                    continue

                image_id = file_path.stem

                # Try to determine room type from path or filename
                room_type = self._infer_room_type(file_path)

                # Load annotations
                annotations = self.annotation_loader.load(image_id)

                sample = ImageSample(
                    image_id=image_id,
                    image_path=file_path,
                    room_type=room_type,
                    source=source,
                    annotations=annotations,
                    metadata={
                        'file_size': file_path.stat().st_size if file_path.exists() else 0
                    }
                )

                self._samples.append(sample)

    def _infer_room_type(self, path: Path) -> Optional[str]:
        """Infer room type from file path or name."""
        path_str = str(path).lower()

        room_types = {
            'bathroom': ['bathroom', 'bath', 'toilet', 'shower'],
            'kitchen': ['kitchen', 'cooking'],
            'bedroom': ['bedroom', 'bed'],
            'living_room': ['living', 'lounge', 'sitting'],
            'stairs': ['stairs', 'stairway', 'staircase'],
            'hallway': ['hallway', 'corridor', 'hall'],
            'entrance': ['entrance', 'entry', 'doorway', 'porch'],
            'external': ['external', 'outdoor', 'garden', 'yard']
        }

        for room, keywords in room_types.items():
            if any(kw in path_str for kw in keywords):
                return room

        return None

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self._samples)

    def __iter__(self) -> Iterator[ImageSample]:
        """Iterate through samples."""
        self._index = 0
        return self

    def __next__(self) -> ImageSample:
        """Get next sample."""
        if self._index >= len(self._samples):
            raise StopIteration
        sample = self._samples[self._index]
        self._index += 1
        return sample

    def __getitem__(self, idx: int) -> ImageSample:
        """Get sample by index."""
        return self._samples[idx]

    def batches(
        self,
        batch_size: int = 8,
        shuffle: bool = False
    ) -> Iterator[List[ImageSample]]:
        """
        Iterate through samples in batches.

        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle samples

        Yields:
            Lists of ImageSample objects
        """
        samples = self._samples.copy()
        if shuffle:
            random.shuffle(samples)

        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]

    def filter(
        self,
        room_type: Optional[str] = None,
        source: Optional[str] = None,
        has_annotations: Optional[bool] = None,
        min_hazards: Optional[int] = None,
        categories: Optional[List[str]] = None
    ) -> List[ImageSample]:
        """
        Filter samples by criteria.

        Args:
            room_type: Filter by room type
            source: Filter by data source
            has_annotations: Filter by annotation presence
            min_hazards: Minimum number of hazards
            categories: Filter by hazard categories present

        Returns:
            Filtered list of samples
        """
        filtered = self._samples

        if room_type:
            filtered = [s for s in filtered if s.room_type == room_type]

        if source:
            filtered = [s for s in filtered if s.source == source]

        if has_annotations is not None:
            filtered = [s for s in filtered if s.has_annotations() == has_annotations]

        if min_hazards is not None:
            filtered = [s for s in filtered if len(s.annotations) >= min_hazards]

        if categories:
            filtered = [
                s for s in filtered
                if any(cat in s.get_categories() for cat in categories)
            ]

        return filtered

    def get_annotated_samples(self) -> List[ImageSample]:
        """Get only samples with annotations."""
        return self.filter(has_annotations=True)

    def get_stats(self) -> DatasetStats:
        """
        Calculate dataset statistics.

        Returns:
            DatasetStats object with counts and distributions
        """
        stats = DatasetStats()
        stats.total_images = len(self._samples)

        for sample in self._samples:
            # Count annotated
            if sample.has_annotations():
                stats.annotated_images += 1

            # Count by room
            if sample.room_type:
                stats.images_by_room[sample.room_type] = \
                    stats.images_by_room.get(sample.room_type, 0) + 1

            # Count by source
            if sample.source:
                stats.images_by_source[sample.source] = \
                    stats.images_by_source.get(sample.source, 0) + 1

            # Count hazards
            for ann in sample.annotations:
                stats.total_hazards += 1

                category = ann.get('category', 'unknown')
                stats.hazards_by_category[category] = \
                    stats.hazards_by_category.get(category, 0) + 1

                severity = ann.get('severity', 'unknown')
                stats.hazards_by_severity[severity] = \
                    stats.hazards_by_severity.get(severity, 0) + 1

        return stats

    def print_stats(self) -> None:
        """Print dataset statistics to console."""
        stats = self.get_stats()

        print("=" * 50)
        print("DATASET STATISTICS")
        print("=" * 50)
        print(f"\nTotal Images: {stats.total_images}")
        print(f"Annotated Images: {stats.annotated_images}")
        print(f"Total Hazards: {stats.total_hazards}")

        if stats.images_by_room:
            print("\nImages by Room Type:")
            for room, count in sorted(stats.images_by_room.items()):
                print(f"  {room}: {count}")

        if stats.hazards_by_category:
            print("\nHazards by Category:")
            for cat, count in sorted(
                stats.hazards_by_category.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"  {cat}: {count}")

        if stats.hazards_by_severity:
            print("\nHazards by Severity:")
            for sev, count in stats.hazards_by_severity.items():
                print(f"  {sev}: {count}")

        print("=" * 50)

    def export_to_json(self, output_path: Union[str, Path]) -> None:
        """
        Export dataset to JSON format.

        Args:
            output_path: Path for output JSON file
        """
        data = {
            'samples': [s.to_dict() for s in self._samples],
            'stats': {
                'total_images': len(self._samples),
                'annotated_images': len(self.get_annotated_samples())
            }
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def split_dataset(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[List[ImageSample], List[ImageSample], List[ImageSample]]:
        """
        Split dataset into train/val/test sets.

        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001

        random.seed(seed)
        samples = self._samples.copy()
        random.shuffle(samples)

        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        return (
            samples[:train_end],
            samples[train_end:val_end],
            samples[val_end:]
        )


def create_annotation_template() -> Dict:
    """
    Create a template annotation structure.

    Returns:
        Template dictionary for image annotations
    """
    return {
        "image_id": "",
        "room_type": "",  # bathroom, kitchen, bedroom, living_room, stairs, etc.
        "source": "",     # staged, public_dataset, synthetic
        "hazards": [
            {
                "category": "",        # flooring, lighting, obstacles, etc.
                "subcategory": "",     # loose_rug, dim_lighting, etc.
                "severity": "",        # low, medium, high, critical
                "location": "",        # Description of where in the image
                "annotator": "",       # Who annotated this
                "notes": ""            # Additional notes
            }
        ],
        "metadata": {
            "captured_date": "",
            "camera": "",
            "notes": ""
        }
    }


def validate_annotation(annotation: Dict) -> Tuple[bool, List[str]]:
    """
    Validate an annotation structure.

    Args:
        annotation: Annotation dictionary to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    required_fields = ['image_id', 'hazards']
    for field in required_fields:
        if field not in annotation:
            errors.append(f"Missing required field: {field}")

    valid_categories = {
        'flooring', 'lighting', 'obstacles', 'furniture',
        'stairs', 'bathroom', 'kitchen', 'bedroom',
        'external', 'general'
    }

    valid_severities = {'low', 'medium', 'high', 'critical'}

    for i, hazard in enumerate(annotation.get('hazards', [])):
        if not hazard.get('category'):
            errors.append(f"Hazard {i}: Missing category")
        elif hazard.get('category', '').lower() not in valid_categories:
            errors.append(f"Hazard {i}: Invalid category '{hazard.get('category')}'")

        if not hazard.get('severity'):
            errors.append(f"Hazard {i}: Missing severity")
        elif hazard.get('severity', '').lower() not in valid_severities:
            errors.append(f"Hazard {i}: Invalid severity '{hazard.get('severity')}'")

    return len(errors) == 0, errors
