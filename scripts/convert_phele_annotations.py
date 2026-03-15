#!/usr/bin/env python3
"""
PHELE Dataset Annotation Converter

Converts PHELE dataset annotations to the project's JSON format
for use with the Fall Risk Detection System.

The PHELE (Physical Hazards of Elderly Living Environment) dataset
contains images of home environments with labeled fall hazards.

Usage:
    python scripts/convert_phele_annotations.py --input /path/to/phele --output data/annotations/phele
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import csv


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Mapping from PHELE hazard types to project categories
PHELE_TO_PROJECT_CATEGORY = {
    # Flooring hazards
    "loose_rug": ("flooring", "loose_rug"),
    "slippery_floor": ("flooring", "slippery_surface"),
    "uneven_floor": ("flooring", "uneven_floor"),
    "wet_floor": ("flooring", "slippery_surface"),
    "floor_clutter": ("obstacles", "floor_clutter"),
    "threshold": ("flooring", "high_threshold"),

    # Lighting hazards
    "poor_lighting": ("lighting", "dim_lighting"),
    "dim_lighting": ("lighting", "dim_lighting"),
    "no_night_light": ("lighting", "no_night_light"),
    "glare": ("lighting", "glare"),

    # Obstacles
    "clutter": ("obstacles", "floor_clutter"),
    "trailing_cords": ("obstacles", "trailing_cords"),
    "cables": ("obstacles", "trailing_cords"),
    "pet_hazard": ("obstacles", "pet_hazards"),
    "narrow_pathway": ("obstacles", "narrow_pathway"),

    # Furniture
    "unstable_furniture": ("furniture", "unstable_furniture"),
    "low_seating": ("furniture", "low_seating"),
    "no_armrests": ("furniture", "no_armrests"),
    "sharp_corners": ("furniture", "sharp_corners"),

    # Stairs
    "no_handrail": ("stairs", "no_handrails"),
    "single_handrail": ("stairs", "single_handrail"),
    "worn_steps": ("stairs", "worn_treads"),
    "steep_stairs": ("stairs", "steep_stairs"),
    "uneven_steps": ("stairs", "uneven_steps"),

    # Bathroom
    "no_grab_bar": ("bathroom", "bath_no_grab_bars"),
    "slippery_bath": ("bathroom", "slippery_bath"),
    "no_bath_mat": ("bathroom", "no_bath_mat"),
    "high_tub": ("bathroom", "high_bath_edge"),

    # Kitchen
    "high_cabinets": ("kitchen", "high_cabinets"),
    "cluttered_counter": ("kitchen", "cluttered_counters"),

    # Default
    "other": ("general", "general_clutter"),
}

# Severity mapping
SEVERITY_MAP = {
    "low": "low",
    "minor": "low",
    "moderate": "medium",
    "medium": "medium",
    "high": "high",
    "severe": "high",
    "critical": "critical",
    "extreme": "critical",
}


def parse_phele_csv(csv_path: Path) -> Dict[str, List[Dict]]:
    """
    Parse PHELE annotations from CSV format.

    Expected columns: image_id, hazard_type, severity, location, description
    """
    annotations = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_id = row.get('image_id', row.get('filename', '')).strip()
            if not image_id:
                continue

            # Remove file extension if present
            if '.' in image_id:
                image_id = image_id.rsplit('.', 1)[0]

            hazard_type = row.get('hazard_type', row.get('type', 'other')).lower().strip()
            severity = row.get('severity', 'medium').lower().strip()
            location = row.get('location', row.get('region', '')).strip()
            description = row.get('description', row.get('notes', '')).strip()

            # Map to project format
            category, subcategory = PHELE_TO_PROJECT_CATEGORY.get(
                hazard_type,
                ("general", hazard_type)
            )
            mapped_severity = SEVERITY_MAP.get(severity, "medium")

            hazard = {
                "category": category,
                "subcategory": subcategory,
                "severity": mapped_severity,
                "location": location,
                "description": description,
                "original_label": hazard_type,
                "annotator": "PHELE_dataset"
            }

            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(hazard)

    return annotations


def parse_phele_json(json_path: Path) -> Dict[str, List[Dict]]:
    """
    Parse PHELE annotations from JSON format.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotations = {}

    # Handle different JSON structures
    if isinstance(data, list):
        # List of image annotations
        for item in data:
            image_id = item.get('image_id', item.get('filename', ''))
            if not image_id:
                continue
            if '.' in image_id:
                image_id = image_id.rsplit('.', 1)[0]

            hazards = item.get('hazards', item.get('annotations', []))
            annotations[image_id] = convert_hazards(hazards)

    elif isinstance(data, dict):
        # Check for COCO format
        if 'images' in data and 'annotations' in data:
            annotations = parse_coco_format(data)
        else:
            # Simple dict format {image_id: [hazards]}
            for image_id, hazards in data.items():
                if isinstance(hazards, dict):
                    hazards = hazards.get('hazards', [])
                annotations[image_id] = convert_hazards(hazards)

    return annotations


def parse_coco_format(data: Dict) -> Dict[str, List[Dict]]:
    """
    Parse COCO-style annotation format.
    """
    # Build image ID to filename mapping
    id_to_name = {}
    for img in data.get('images', []):
        img_id = img.get('id')
        filename = img.get('file_name', '')
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]
        id_to_name[img_id] = filename

    # Build category ID to name mapping
    cat_id_to_name = {}
    for cat in data.get('categories', []):
        cat_id_to_name[cat.get('id')] = cat.get('name', 'unknown')

    annotations = {}

    for ann in data.get('annotations', []):
        img_id = ann.get('image_id')
        image_name = id_to_name.get(img_id, str(img_id))

        if image_name not in annotations:
            annotations[image_name] = []

        cat_id = ann.get('category_id')
        hazard_type = cat_id_to_name.get(cat_id, 'unknown')

        category, subcategory = PHELE_TO_PROJECT_CATEGORY.get(
            hazard_type.lower(),
            ("general", hazard_type)
        )

        hazard = {
            "category": category,
            "subcategory": subcategory,
            "severity": ann.get('severity', 'medium'),
            "location": ann.get('location', ''),
            "description": ann.get('description', ''),
            "original_label": hazard_type,
            "annotator": "PHELE_dataset"
        }

        # Add bounding box if available
        bbox = ann.get('bbox')
        if bbox and len(bbox) == 4:
            hazard["bounding_box"] = {
                "x": bbox[0],
                "y": bbox[1],
                "width": bbox[2],
                "height": bbox[3]
            }

        annotations[image_name].append(hazard)

    return annotations


def convert_hazards(hazards: List[Dict]) -> List[Dict]:
    """
    Convert hazard list to project format.
    """
    converted = []

    for h in hazards:
        hazard_type = h.get('type', h.get('hazard_type', h.get('category', 'other')))
        hazard_type = hazard_type.lower().strip()

        category, subcategory = PHELE_TO_PROJECT_CATEGORY.get(
            hazard_type,
            ("general", hazard_type)
        )

        severity = h.get('severity', 'medium').lower()
        mapped_severity = SEVERITY_MAP.get(severity, 'medium')

        converted.append({
            "category": category,
            "subcategory": subcategory,
            "severity": mapped_severity,
            "location": h.get('location', h.get('region', '')),
            "description": h.get('description', h.get('notes', '')),
            "original_label": hazard_type,
            "annotator": "PHELE_dataset"
        })

    return converted


def discover_annotation_files(input_dir: Path) -> List[Path]:
    """
    Find all annotation files in the input directory.
    """
    files = []

    for pattern in ['*.csv', '*.json', 'annotations.*', '*_labels.*']:
        files.extend(input_dir.glob(pattern))
        files.extend(input_dir.glob(f'**/{pattern}'))

    return list(set(files))


def scan_images_for_annotations(input_dir: Path) -> Dict[str, List[Dict]]:
    """
    If no annotation files found, scan for images and create placeholder annotations.
    This allows manual annotation later.
    """
    annotations = {}

    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    for split in ['train', 'test', 'val']:
        split_dir = input_dir / split
        if split_dir.exists():
            for img_file in split_dir.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    image_id = img_file.stem
                    annotations[image_id] = []  # Empty - needs manual annotation

    # Also check root directory
    for img_file in input_dir.iterdir():
        if img_file.suffix.lower() in image_extensions:
            image_id = img_file.stem
            if image_id not in annotations:
                annotations[image_id] = []

    return annotations


def main():
    parser = argparse.ArgumentParser(
        description='Convert PHELE dataset annotations to project format'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/raw/phele',
        help='Path to PHELE dataset directory'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/annotations/phele',
        help='Output directory for converted annotations'
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['combined', 'individual'],
        default='combined',
        help='Output format: combined JSON or individual files per image'
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        print(f"Please copy your PHELE dataset to: {input_dir}")
        print("\nExpected structure:")
        print("  data/raw/phele/")
        print("  ├── train/           # Training images")
        print("  ├── test/            # Test images")
        print("  └── annotations.csv  # Or annotations.json")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting PHELE annotations...")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")

    # Find annotation files
    annotation_files = discover_annotation_files(input_dir)

    all_annotations = {}

    if annotation_files:
        print(f"\nFound {len(annotation_files)} annotation file(s):")

        for ann_file in annotation_files:
            print(f"  - {ann_file.name}")

            if ann_file.suffix.lower() == '.csv':
                annotations = parse_phele_csv(ann_file)
            elif ann_file.suffix.lower() == '.json':
                annotations = parse_phele_json(ann_file)
            else:
                continue

            # Merge annotations
            for image_id, hazards in annotations.items():
                if image_id not in all_annotations:
                    all_annotations[image_id] = []
                all_annotations[image_id].extend(hazards)
    else:
        print("\nNo annotation files found. Scanning for images...")
        all_annotations = scan_images_for_annotations(input_dir)

        if all_annotations:
            print(f"Found {len(all_annotations)} images without annotations.")
            print("Creating placeholder annotation file for manual annotation.")
        else:
            print("No images found either. Please check your dataset structure.")
            sys.exit(1)

    # Output annotations
    if args.format == 'combined':
        output_file = output_dir / 'annotations.json'

        output_data = {
            "dataset": "PHELE",
            "description": "Physical Hazards of Elderly Living Environment",
            "converted_by": "convert_phele_annotations.py",
            "total_images": len(all_annotations),
            "total_hazards": sum(len(h) for h in all_annotations.values()),
            "annotations": all_annotations
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nSaved combined annotations to: {output_file}")

    else:
        # Individual files
        for image_id, hazards in all_annotations.items():
            output_file = output_dir / f'{image_id}.json'

            annotation_data = {
                "image_id": image_id,
                "room_type": "",  # To be filled manually if needed
                "source": "PHELE",
                "hazards": hazards,
                "metadata": {
                    "dataset": "PHELE",
                    "converted_by": "convert_phele_annotations.py"
                }
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2)

        print(f"\nSaved {len(all_annotations)} individual annotation files to: {output_dir}")

    # Print summary
    print("\n" + "=" * 50)
    print("CONVERSION SUMMARY")
    print("=" * 50)
    print(f"Total images:  {len(all_annotations)}")
    print(f"Total hazards: {sum(len(h) for h in all_annotations.values())}")

    # Category breakdown
    category_counts = {}
    for hazards in all_annotations.values():
        for h in hazards:
            cat = h.get('category', 'unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1

    if category_counts:
        print("\nHazards by category:")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

    print("=" * 50)


if __name__ == '__main__':
    main()
