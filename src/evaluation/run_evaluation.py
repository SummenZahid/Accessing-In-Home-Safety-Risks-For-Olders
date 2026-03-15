#!/usr/bin/env python3
"""
Evaluation Pipeline for Fall Hazard Detection

Runs systematic evaluation of hazard detection models on the PHELE dataset.
Compares predictions against ground truth annotations and generates
comprehensive metrics reports.

Usage:
    python -m src.evaluation.run_evaluation --model llava --split test

    # Compare multiple models
    python -m src.evaluation.run_evaluation --model llava gemini --split test
"""

import os
import sys
import json
import argparse
import base64
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.metrics import HazardEvaluator, EvaluationResult
from src.preprocessing.data_loader import DataLoader, DatasetSplit


@dataclass
class ModelConfig:
    """Configuration for a detection model."""
    name: str
    model_type: str  # 'ollama', 'gemini', 'gpt4v'
    model_id: str
    endpoint: Optional[str] = None
    api_key: Optional[str] = None


# Available model configurations
MODEL_CONFIGS = {
    "llava": ModelConfig(
        name="LLaVA 7B",
        model_type="ollama",
        model_id="llava:7b",
        endpoint="http://localhost:11434/api/generate"
    ),
    "llava-13b": ModelConfig(
        name="LLaVA 13B",
        model_type="ollama",
        model_id="llava:13b",
        endpoint="http://localhost:11434/api/generate"
    ),
    "moondream": ModelConfig(
        name="Moondream2",
        model_type="ollama",
        model_id="moondream",
        endpoint="http://localhost:11434/api/generate"
    ),
    "gemini": ModelConfig(
        name="Gemini 2.0 Flash",
        model_type="gemini",
        model_id="gemini-2.0-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    ),
    "gpt4v": ModelConfig(
        name="GPT-4 Vision",
        model_type="gpt4v",
        model_id="gpt-4-vision-preview",
        api_key=os.getenv("OPENAI_API_KEY")
    ),
}


# Conservative hazard detection prompt
DETECTION_PROMPT = """You are an expert occupational therapist assessing a home environment for fall hazards affecting elderly residents.

Analyze this image and identify ONLY the most significant fall hazards that are clearly visible.

IMPORTANT RULES:
1. Only report hazards you can CLEARLY SEE in the image
2. Do NOT guess or assume hazards that aren't visible
3. Report a maximum of 6 hazards
4. Each hazard must have confidence >= 0.6

For each hazard, provide:
- category: One of [flooring, lighting, obstacles, furniture, stairs, bathroom, kitchen, bedroom, external, general]
- subcategory: Specific hazard type (e.g., loose_rug, dim_lighting, no_grab_bars)
- severity: One of [low, medium, high, critical]
- confidence: Your confidence level (0.0 to 1.0)
- location: Where in the image (e.g., "center", "left side", "near door")
- description: Brief description of the hazard

Respond in valid JSON format:
{
  "hazards": [
    {
      "category": "category_name",
      "subcategory": "specific_type",
      "severity": "severity_level",
      "confidence": 0.8,
      "location": "location_description",
      "description": "brief description"
    }
  ]
}

If no hazards are visible, respond with: {"hazards": []}
"""


class ModelRunner:
    """Runs inference with different model backends."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def detect_hazards(self, image_path: Path) -> List[Dict]:
        """
        Run hazard detection on an image.

        Args:
            image_path: Path to the image file

        Returns:
            List of detected hazard dicts
        """
        if self.config.model_type == "ollama":
            return self._run_ollama(image_path)
        elif self.config.model_type == "gemini":
            return self._run_gemini(image_path)
        elif self.config.model_type == "gpt4v":
            return self._run_gpt4v(image_path)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def _run_ollama(self, image_path: Path) -> List[Dict]:
        """Run inference with Ollama (local LLaVA)."""
        # Read and encode image
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        try:
            response = requests.post(
                self.config.endpoint,
                json={
                    "model": self.config.model_id,
                    "prompt": DETECTION_PROMPT,
                    "images": [image_b64],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 1500
                    }
                },
                timeout=120
            )
            response.raise_for_status()

            result = response.json()
            text = result.get("response", "")

            return self._parse_response(text)

        except requests.exceptions.RequestException as e:
            print(f"  Error with Ollama: {e}")
            return []

    def _run_gemini(self, image_path: Path) -> List[Dict]:
        """Run inference with Google Gemini."""
        if not self.config.api_key:
            print("  Warning: No GOOGLE_API_KEY set")
            return []

        try:
            import google.generativeai as genai
            from PIL import Image

            genai.configure(api_key=self.config.api_key)
            model = genai.GenerativeModel(self.config.model_id)

            img = Image.open(image_path)
            response = model.generate_content([DETECTION_PROMPT, img])

            return self._parse_response(response.text)

        except ImportError:
            print("  Warning: google-generativeai not installed")
            return []
        except Exception as e:
            print(f"  Error with Gemini: {e}")
            return []

    def _run_gpt4v(self, image_path: Path) -> List[Dict]:
        """Run inference with GPT-4 Vision."""
        if not self.config.api_key:
            print("  Warning: No OPENAI_API_KEY set")
            return []

        try:
            import openai

            with open(image_path, "rb") as f:
                image_bytes = f.read()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            client = openai.OpenAI(api_key=self.config.api_key)

            response = client.chat.completions.create(
                model=self.config.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": DETECTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500
            )

            text = response.choices[0].message.content
            return self._parse_response(text)

        except ImportError:
            print("  Warning: openai not installed")
            return []
        except Exception as e:
            print(f"  Error with GPT-4V: {e}")
            return []

    def _parse_response(self, text: str) -> List[Dict]:
        """Parse model response to extract hazards."""
        hazards = []

        # Try to extract JSON from response
        try:
            # First, try to find a JSON array (some models return array directly)
            text = text.strip()

            # Check if response starts with array
            if text.startswith('['):
                start = text.find('[')
                end = text.rfind(']') + 1
                if end > start:
                    json_str = text[start:end]
                    # Try to fix truncated array
                    if json_str.count('{') > json_str.count('}'):
                        last_complete = json_str.rfind('},')
                        if last_complete > 0:
                            json_str = json_str[:last_complete + 1] + ']'
                        else:
                            # Try to close the last object
                            json_str = json_str.rstrip(',') + '}]'
                    hazards = json.loads(json_str)

            # Check if response is a single object (not wrapped)
            elif text.startswith('{') and '"hazards"' not in text:
                start = text.find('{')
                end = text.rfind('}') + 1
                if end > start:
                    json_str = text[start:end]
                    obj = json.loads(json_str)
                    if 'category' in obj:  # It's a hazard object
                        hazards = [obj]

            # Standard format: {"hazards": [...]}
            else:
                start = text.find('{')
                json_str = text[start:]

                # Fix truncated JSON - count brackets and add missing ones
                open_braces = json_str.count('{') - json_str.count('}')
                open_brackets = json_str.count('[') - json_str.count(']')

                if open_braces > 0 or open_brackets > 0:
                    # Add missing closing brackets/braces
                    json_str = json_str + (']' * open_brackets) + ('}' * open_braces)

                data = json.loads(json_str)
                hazards = data.get('hazards', [])

        except json.JSONDecodeError:
            pass

        # Normalize and filter hazards
        filtered = []
        seen_hazards = set()  # Deduplicate

        for h in hazards:
            # Skip invalid entries
            if not isinstance(h, dict):
                continue

            # Normalize severity (handle numeric values from smaller models)
            severity = h.get('severity', 'medium')
            if isinstance(severity, (int, float)):
                if severity >= 0.75:
                    severity = 'critical'
                elif severity >= 0.5:
                    severity = 'high'
                elif severity >= 0.25:
                    severity = 'medium'
                else:
                    severity = 'low'
            h['severity'] = severity

            # Get confidence
            confidence = h.get('confidence', 0.5)
            if confidence < 0.6:
                continue

            # Deduplicate by category+subcategory
            key = f"{h.get('category', '')}:{h.get('subcategory', '')}"
            if key in seen_hazards:
                continue
            seen_hazards.add(key)

            filtered.append(h)

        # Limit to 6 hazards
        return filtered[:6]


class EvaluationPipeline:
    """Main evaluation pipeline."""

    def __init__(
        self,
        data_dir: str = "data/raw/phele",
        annotation_dir: str = "data/annotations/phele",
        output_dir: str = "results/evaluation"
    ):
        self.data_dir = Path(data_dir)
        self.annotation_dir = Path(annotation_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.evaluator = HazardEvaluator()

    def load_ground_truth(self) -> Dict[str, List[Dict]]:
        """Load ground truth annotations from PHELE dataset."""
        annotations = {}

        # Try combined annotations file
        combined_file = self.annotation_dir / "annotations.json"
        if combined_file.exists():
            with open(combined_file, 'r') as f:
                data = json.load(f)
                if 'annotations' in data:
                    annotations = data['annotations']
                elif isinstance(data, dict) and not any(k in data for k in ['dataset', 'description']):
                    annotations = data

        # Try individual annotation files
        for json_file in self.annotation_dir.glob("*.json"):
            if json_file.name == "annotations.json":
                continue

            with open(json_file, 'r') as f:
                data = json.load(f)

            image_id = json_file.stem
            if image_id not in annotations:
                hazards = data.get('hazards', [])
                annotations[image_id] = hazards

        return annotations

    def get_test_images(self, split: str = "test") -> List[Path]:
        """Get list of test images."""
        images = []

        # Check split directory
        split_dir = self.data_dir / split
        if split_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                images.extend(split_dir.glob(ext))

        # Check root directory if no split found
        if not images:
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                images.extend(self.data_dir.glob(ext))

        return sorted(images)

    def run_evaluation(
        self,
        model_names: List[str],
        split: str = "test",
        max_images: Optional[int] = None
    ) -> Dict[str, EvaluationResult]:
        """
        Run evaluation for specified models.

        Args:
            model_names: List of model names to evaluate
            split: Dataset split to use
            max_images: Maximum number of images to process

        Returns:
            Dict mapping model name to EvaluationResult
        """
        print("=" * 60)
        print("FALL HAZARD DETECTION EVALUATION")
        print("=" * 60)
        print(f"Dataset: PHELE")
        print(f"Split: {split}")
        print(f"Models: {', '.join(model_names)}")
        print()

        # Load ground truth
        print("Loading ground truth annotations...")
        ground_truth = self.load_ground_truth()
        print(f"  Found {len(ground_truth)} annotated images")

        # Get test images
        print("Loading test images...")
        test_images = self.get_test_images(split)
        print(f"  Found {len(test_images)} test images")

        if max_images:
            test_images = test_images[:max_images]
            print(f"  Limited to {max_images} images")

        if not test_images:
            print("\nError: No test images found!")
            print(f"Please add images to: {self.data_dir / split}")
            return {}

        # Run evaluation for each model
        results = {}

        for model_name in model_names:
            if model_name not in MODEL_CONFIGS:
                print(f"\nWarning: Unknown model '{model_name}', skipping")
                continue

            config = MODEL_CONFIGS[model_name]
            print(f"\n{'=' * 60}")
            print(f"Evaluating: {config.name}")
            print(f"{'=' * 60}")

            runner = ModelRunner(config)
            all_predictions = []
            all_ground_truth = []

            for i, image_path in enumerate(test_images):
                image_id = image_path.stem
                print(f"  [{i+1}/{len(test_images)}] {image_id}...", end=" ")

                # Get ground truth
                gt = ground_truth.get(image_id, [])
                all_ground_truth.append(gt)

                # Run inference
                start_time = time.time()
                predictions = runner.detect_hazards(image_path)
                elapsed = time.time() - start_time

                all_predictions.append(predictions)
                print(f"Detected {len(predictions)} hazards (GT: {len(gt)}) [{elapsed:.1f}s]")

            # Calculate metrics
            print("\nCalculating metrics...")
            result = self.evaluator.evaluate_batch(all_predictions, all_ground_truth)
            results[model_name] = result

            # Print summary
            print(self.evaluator.generate_report(result))

        # Save results
        self._save_results(results, model_names, split)

        return results

    def _save_results(
        self,
        results: Dict[str, EvaluationResult],
        model_names: List[str],
        split: str
    ):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results as JSON
        output_file = self.output_dir / f"evaluation_{split}_{timestamp}.json"

        output_data = {
            "timestamp": timestamp,
            "split": split,
            "models": model_names,
            "results": {}
        }

        for model_name, result in results.items():
            output_data["results"][model_name] = {
                "overall_precision": result.overall_precision,
                "overall_recall": result.overall_recall,
                "overall_f1": result.overall_f1,
                "severity_accuracy": result.severity_accuracy,
                "severity_kappa": result.severity_kappa,
                "total_predictions": result.total_predictions,
                "total_ground_truth": result.total_ground_truth,
                "category_metrics": {
                    cat: {
                        "precision": m.precision,
                        "recall": m.recall,
                        "f1_score": m.f1_score,
                        "true_positives": m.true_positives,
                        "false_positives": m.false_positives,
                        "false_negatives": m.false_negatives
                    }
                    for cat, m in result.category_metrics.items()
                }
            }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_file}")

        # Generate comparison table
        if len(results) > 1:
            self._print_comparison_table(results)

    def _print_comparison_table(self, results: Dict[str, EvaluationResult]):
        """Print comparison table for multiple models."""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        header = f"{'Model':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Sev.Acc':<10}"
        print(header)
        print("-" * 60)

        for model_name, result in results.items():
            row = f"{model_name:<15} {result.overall_precision:<12.4f} {result.overall_recall:<12.4f} {result.overall_f1:<12.4f} {result.severity_accuracy:<10.4f}"
            print(row)

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate hazard detection models on PHELE dataset'
    )
    parser.add_argument(
        '--model', '-m',
        nargs='+',
        default=['llava'],
        choices=list(MODEL_CONFIGS.keys()),
        help='Model(s) to evaluate'
    )
    parser.add_argument(
        '--split', '-s',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to use'
    )
    parser.add_argument(
        '--max-images', '-n',
        type=int,
        default=None,
        help='Maximum number of images to evaluate'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw/phele',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--annotation-dir',
        type=str,
        default='data/annotations/phele',
        help='Path to annotations directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluation',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = EvaluationPipeline(
        data_dir=args.data_dir,
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir
    )

    results = pipeline.run_evaluation(
        model_names=args.model,
        split=args.split,
        max_images=args.max_images
    )

    # Exit with appropriate code
    if results:
        print("\nEvaluation completed successfully!")
        sys.exit(0)
    else:
        print("\nEvaluation failed - no results generated")
        sys.exit(1)


if __name__ == '__main__':
    main()
