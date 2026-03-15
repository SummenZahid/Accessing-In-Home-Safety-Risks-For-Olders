#!/usr/bin/env python3
"""
Training Data Preparation for LLaVA Fine-tuning

Converts PHELE dataset annotations to the format expected by LLaVA fine-tuning.
The output can be used with:
- Unsloth (recommended for Google Colab)
- Official LLaVA fine-tuning scripts
- LLaMA-Factory

Usage:
    python scripts/prepare_training_data.py \
        --input data/raw/phele \
        --annotations data/annotations/phele \
        --output data/training/llava_format

Output Format (LLaVA conversation format):
{
  "id": "unique_id",
  "image": "path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nPrompt text..."},
    {"from": "gpt", "value": "Response with detected hazards..."}
  ]
}
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Detection prompt template
DETECTION_PROMPT = """<image>
Analyze this home environment image and identify all fall hazards that could affect elderly residents.

For each hazard you identify, provide:
1. Category (flooring, lighting, obstacles, furniture, stairs, bathroom, kitchen, bedroom, external, general)
2. Specific type of hazard
3. Severity level (low, medium, high, critical)
4. Brief description

Be thorough but accurate - only report hazards you can clearly see in the image."""


def format_hazards_as_response(hazards: List[Dict]) -> str:
    """
    Format hazard annotations as a natural language response.

    Args:
        hazards: List of hazard annotation dicts

    Returns:
        Formatted response string
    """
    if not hazards:
        return "After careful analysis of this image, I did not identify any significant fall hazards. The environment appears to be relatively safe for elderly residents."

    # Group hazards by severity
    severity_order = ['critical', 'high', 'medium', 'low']
    hazards_by_severity = {s: [] for s in severity_order}

    for h in hazards:
        sev = h.get('severity', 'medium').lower()
        if sev in hazards_by_severity:
            hazards_by_severity[sev].append(h)

    # Build response
    lines = ["I have identified the following fall hazards in this image:\n"]

    hazard_num = 1
    for severity in severity_order:
        for h in hazards_by_severity[severity]:
            category = h.get('category', 'general')
            subcategory = h.get('subcategory', h.get('type', 'unspecified'))
            location = h.get('location', '')
            description = h.get('description', '')

            # Format subcategory nicely
            subcategory_display = subcategory.replace('_', ' ').title()

            line = f"{hazard_num}. **{subcategory_display}** ({category})"
            line += f"\n   - Severity: {severity.upper()}"

            if location:
                line += f"\n   - Location: {location}"

            if description:
                line += f"\n   - {description}"

            lines.append(line)
            hazard_num += 1

    # Add summary
    total = len(hazards)
    critical_count = len(hazards_by_severity['critical'])
    high_count = len(hazards_by_severity['high'])

    lines.append(f"\n**Summary**: Found {total} hazard(s)")
    if critical_count > 0:
        lines.append(f"- {critical_count} critical hazard(s) requiring immediate attention")
    if high_count > 0:
        lines.append(f"- {high_count} high-severity hazard(s) that should be addressed soon")

    return "\n".join(lines)


def format_hazards_as_json_response(hazards: List[Dict]) -> str:
    """
    Format hazard annotations as JSON response (for structured output).

    Args:
        hazards: List of hazard annotation dicts

    Returns:
        JSON-formatted response string
    """
    formatted_hazards = []

    for h in hazards:
        formatted = {
            "category": h.get('category', 'general'),
            "subcategory": h.get('subcategory', h.get('type', 'unspecified')),
            "severity": h.get('severity', 'medium'),
            "confidence": h.get('confidence', 0.8),
            "location": h.get('location', ''),
            "description": h.get('description', '')
        }
        formatted_hazards.append(formatted)

    return json.dumps({"hazards": formatted_hazards}, indent=2)


def load_annotations(annotation_dir: Path) -> Dict[str, List[Dict]]:
    """Load all annotations from annotation directory."""
    annotations = {}

    # Try combined file first
    combined_file = annotation_dir / "annotations.json"
    if combined_file.exists():
        with open(combined_file, 'r') as f:
            data = json.load(f)

        if 'annotations' in data:
            annotations = data['annotations']
        elif isinstance(data, dict):
            # Check if it's the raw format
            for key, value in data.items():
                if isinstance(value, list):
                    annotations[key] = value

    # Load individual files
    for json_file in annotation_dir.glob("*.json"):
        if json_file.name == "annotations.json":
            continue

        image_id = json_file.stem
        if image_id not in annotations:
            with open(json_file, 'r') as f:
                data = json.load(f)
            annotations[image_id] = data.get('hazards', [])

    return annotations


def find_image(image_id: str, data_dir: Path) -> Optional[Path]:
    """Find image file by ID."""
    extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']

    # Check in subdirectories
    for subdir in ['train', 'test', 'val', '']:
        search_dir = data_dir / subdir if subdir else data_dir
        if not search_dir.exists():
            continue

        for ext in extensions:
            img_path = search_dir / f"{image_id}{ext}"
            if img_path.exists():
                return img_path

    return None


def create_training_sample(
    image_id: str,
    image_path: Path,
    hazards: List[Dict],
    response_format: str = "natural"
) -> Dict[str, Any]:
    """
    Create a single training sample in LLaVA format.

    Args:
        image_id: Unique identifier
        image_path: Path to image file
        hazards: List of hazard annotations
        response_format: "natural" for text, "json" for structured

    Returns:
        Training sample dict
    """
    if response_format == "json":
        response = format_hazards_as_json_response(hazards)
    else:
        response = format_hazards_as_response(hazards)

    return {
        "id": image_id,
        "image": str(image_path),
        "conversations": [
            {
                "from": "human",
                "value": DETECTION_PROMPT
            },
            {
                "from": "gpt",
                "value": response
            }
        ]
    }


def prepare_dataset(
    data_dir: Path,
    annotation_dir: Path,
    output_dir: Path,
    response_format: str = "natural",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    copy_images: bool = False,
    seed: int = 42
) -> Dict[str, int]:
    """
    Prepare full training dataset.

    Args:
        data_dir: Directory containing images
        annotation_dir: Directory containing annotations
        output_dir: Output directory
        response_format: Response format (natural/json)
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        copy_images: Whether to copy images to output directory
        seed: Random seed for reproducibility

    Returns:
        Dict with counts of samples per split
    """
    random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    print("Loading annotations...")
    annotations = load_annotations(annotation_dir)
    print(f"  Found {len(annotations)} annotated images")

    # Match images with annotations
    print("Matching images with annotations...")
    samples = []

    for image_id, hazards in annotations.items():
        image_path = find_image(image_id, data_dir)
        if image_path:
            samples.append({
                "id": image_id,
                "image_path": image_path,
                "hazards": hazards
            })
        else:
            print(f"  Warning: Image not found for {image_id}")

    print(f"  Matched {len(samples)} images")

    if not samples:
        print("\nError: No samples found!")
        return {"train": 0, "val": 0, "test": 0}

    # Shuffle and split
    random.shuffle(samples)

    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    splits = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:]
    }

    # Create output for each split
    counts = {}

    for split_name, split_samples in splits.items():
        if not split_samples:
            counts[split_name] = 0
            continue

        print(f"\nProcessing {split_name} split ({len(split_samples)} samples)...")

        training_data = []

        for sample in split_samples:
            # Create training sample
            train_sample = create_training_sample(
                image_id=sample["id"],
                image_path=sample["image_path"],
                hazards=sample["hazards"],
                response_format=response_format
            )

            # Optionally copy image
            if copy_images:
                img_output_dir = output_dir / "images" / split_name
                img_output_dir.mkdir(parents=True, exist_ok=True)

                new_img_path = img_output_dir / sample["image_path"].name
                shutil.copy2(sample["image_path"], new_img_path)
                train_sample["image"] = str(new_img_path.relative_to(output_dir))

            training_data.append(train_sample)

        # Save split data
        output_file = output_dir / f"{split_name}.json"
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)

        print(f"  Saved to: {output_file}")
        counts[split_name] = len(split_samples)

    # Create combined file for convenience
    all_train = []
    train_file = output_dir / "train.json"
    if train_file.exists():
        with open(train_file, 'r') as f:
            all_train = json.load(f)

    combined_file = output_dir / "training_data.json"
    with open(combined_file, 'w') as f:
        json.dump(all_train, f, indent=2)

    print(f"\nCombined training data: {combined_file}")

    return counts


def main():
    parser = argparse.ArgumentParser(
        description='Prepare PHELE dataset for LLaVA fine-tuning'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/raw/phele',
        help='Path to PHELE dataset directory'
    )
    parser.add_argument(
        '--annotations', '-a',
        type=str,
        default='data/annotations/phele',
        help='Path to annotations directory'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/training/llava_format',
        help='Output directory for training data'
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['natural', 'json'],
        default='natural',
        help='Response format: natural language or JSON'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training set ratio (default: 0.8)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--copy-images',
        action='store_true',
        help='Copy images to output directory'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Error: Ratios must sum to 1.0 (got {total_ratio})")
        sys.exit(1)

    print("=" * 60)
    print("LLAVA TRAINING DATA PREPARATION")
    print("=" * 60)
    print(f"Input directory: {args.input}")
    print(f"Annotations: {args.annotations}")
    print(f"Output directory: {args.output}")
    print(f"Response format: {args.format}")
    print(f"Split ratios: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")

    # Prepare dataset
    counts = prepare_dataset(
        data_dir=Path(args.input),
        annotation_dir=Path(args.annotations),
        output_dir=Path(args.output),
        response_format=args.format,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        copy_images=args.copy_images,
        seed=args.seed
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Training samples:   {counts.get('train', 0)}")
    print(f"Validation samples: {counts.get('val', 0)}")
    print(f"Test samples:       {counts.get('test', 0)}")
    print(f"Total:              {sum(counts.values())}")

    print("\nNext steps:")
    print("1. Upload the training data to Google Drive")
    print("2. Open notebooks/finetune_llava_colab.ipynb in Google Colab")
    print("3. Follow the notebook instructions to fine-tune LLaVA")
    print("=" * 60)


if __name__ == '__main__':
    main()
