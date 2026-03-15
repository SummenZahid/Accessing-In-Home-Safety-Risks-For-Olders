"""
Batch Image Classification Script
Classifies PHELE dataset images into low_risk and high_risk folders
using LLaVA via Ollama for fall hazard detection.

Usage:
    python scripts/classify_images.py
    python scripts/classify_images.py --source data/raw/phele/test --max-images 5
    python scripts/classify_images.py --source data/raw/phele/train --model llava:7b
"""

import argparse
import base64
import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import requests


# ─── Risk scoring (copied from app.py to avoid Streamlit dependency) ───

CONFIDENCE_THRESHOLD = 0.6
MAX_HAZARDS = 6


def calculate_risk_score(hazards: list) -> tuple:
    if not hazards:
        return 0, "LOW"
    severity_weights = {"low": 0.25, "medium": 0.50, "high": 0.75, "critical": 1.0}
    category_weights = {
        "stairs": 1.0, "bathroom": 0.95, "flooring": 0.85, "obstacles": 0.85,
        "lighting": 0.80, "furniture": 0.75, "kitchen": 0.75, "bedroom": 0.70,
        "external": 0.80, "general": 0.65
    }
    total_score = 0
    for hazard in hazards:
        severity = hazard.get('severity', 'medium').lower()
        category = hazard.get('category', 'general').lower()
        sev_weight = severity_weights.get(severity, 0.5)
        cat_weight = category_weights.get(category, 0.7)
        total_score += sev_weight * cat_weight * 20
    final_score = min(100, total_score)
    if final_score <= 25:
        level = "LOW"
    elif final_score <= 50:
        level = "MODERATE"
    elif final_score <= 75:
        level = "HIGH"
    else:
        level = "CRITICAL"
    return round(final_score), level


def filter_hazards(hazards: list, confidence_threshold: float, max_hazards: int) -> list:
    filtered = [h for h in hazards if h.get('confidence', 0.5) >= confidence_threshold]
    severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
    filtered.sort(key=lambda h: (
        severity_order.get(h.get('severity', 'medium').lower(), 2),
        -h.get('confidence', 0.5)
    ))
    return filtered[:max_hazards]


def extract_hazards_from_text(text: str) -> list:
    keyword_to_category = {
        'rug': 'flooring', 'mat': 'flooring', 'carpet': 'flooring', 'floor': 'flooring',
        'tile': 'flooring', 'slip': 'flooring', 'wet': 'flooring', 'threshold': 'flooring',
        'cord': 'obstacles', 'cable': 'obstacles', 'wire': 'obstacles', 'clutter': 'obstacles',
        'object': 'obstacles', 'trip': 'obstacles', 'obstruct': 'obstacles',
        'stair': 'stairs', 'step': 'stairs', 'railing': 'stairs', 'handrail': 'stairs',
        'grab bar': 'bathroom', 'shower': 'bathroom', 'tub': 'bathroom', 'bath': 'bathroom',
        'toilet': 'bathroom',
        'light': 'lighting', 'dim': 'lighting', 'dark': 'lighting', 'lamp': 'lighting',
        'chair': 'furniture', 'table': 'furniture', 'couch': 'furniture', 'bed': 'furniture',
        'edge': 'furniture', 'sharp': 'furniture', 'unstable': 'furniture',
    }
    high_severity_keywords = {'stair', 'wet', 'grab bar', 'railing', 'slip', 'tub', 'shower'}
    sentences = re.split(r'[.!?\n]', text)
    found_hazards = []
    seen_keywords = set()
    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        if not sentence_lower:
            continue
        for keyword, category in keyword_to_category.items():
            if keyword in sentence_lower and keyword not in seen_keywords:
                seen_keywords.add(keyword)
                severity = 'high' if keyword in high_severity_keywords else 'medium'
                found_hazards.append({
                    'category': category, 'subcategory': keyword, 'severity': severity,
                    'description': sentence.strip(), 'region': 'center',
                    'confidence': 0.65,
                    'recommendation': f'Address {keyword} hazard to reduce fall risk'
                })
                break
    return found_hazards[:6]


# ─── Ollama inference (adapted from app.py without Streamlit) ───

def check_ollama_status() -> bool:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def analyze_image(image_bytes: bytes, model: str = "llava:7b") -> dict:
    """Analyze image using Ollama. Returns dict with hazards, risk_score, risk_level."""
    try:
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        is_moondream = 'moondream' in model.lower()

        if is_moondream:
            prompt = """Look at this image of a room in a home. List any fall hazards you can see that could cause an elderly person to trip, slip, or fall.

For each hazard, state what it is, how dangerous it is (low, medium, high, or critical), and describe it briefly.

Respond in JSON format:
{"room_type": "room name", "hazards": [{"subcategory": "hazard name", "severity": "low/medium/high/critical", "description": "what you see"}]}

If no hazards are visible, respond: {"room_type": "room name", "hazards": []}"""
            temperature = 0.0
            num_predict = 800
        else:
            prompt = """You are an expert occupational therapist assessing a home for fall hazards.

Analyze this image and identify ONLY the most significant fall hazards that are clearly visible.
Be conservative - only report hazards you can clearly see, not potential or assumed hazards.

For each hazard provide:
1. category: one of (bathroom, stairs, flooring, lighting, obstacles, furniture, kitchen, bedroom, external, general)
2. subcategory: specific hazard name (e.g., "loose rug", "missing grab bar")
3. severity: low, medium, high, or critical
4. description: brief description of the hazard
5. region: where in the image
6. confidence: 0.0 to 1.0
7. recommendation: how to fix it

IMPORTANT RULES:
- Only report 3-6 hazards maximum
- Only report hazards with confidence >= 0.6
- Be specific about what you actually see

Respond ONLY with valid JSON:
{"room_type": "room_type", "hazards": [{"category": "cat", "subcategory": "sub", "severity": "level", "description": "desc", "region": "where", "confidence": 0.8, "recommendation": "fix"}]}

If no clear hazards are visible, return: {"room_type": "room_type", "hazards": []}"""
            temperature = 0.0
            num_predict = 1500

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {"temperature": temperature, "num_predict": num_predict, "seed": 42}
            },
            timeout=120
        )

        if response.status_code != 200:
            return None

        result = response.json()
        response_text = result.get('response', '')
        raw_response_text = response_text

        # Clean response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            for part in response_text.split("```"):
                if "{" in part and "}" in part:
                    response_text = part
                    break

        start_idx = response_text.find("{")
        if start_idx == -1:
            return None

        json_str = response_text[start_idx:]

        try:
            parsed_result = json.loads(json_str)
        except json.JSONDecodeError:
            fixed_json = json_str
            last_valid = None
            for match in re.finditer(r'[}\]"\d]|true|false|null', json_str):
                last_valid = match.end()
            if last_valid:
                fixed_json = json_str[:last_valid]
            trailing = fixed_json[fixed_json.rfind(',') + 1:] if ',' in fixed_json else fixed_json
            if trailing.count('"') % 2 == 1:
                last_comma = fixed_json.rfind(',')
                if last_comma > 0:
                    fixed_json = fixed_json[:last_comma]
            fixed_json = fixed_json.rstrip().rstrip(',')
            open_braces = fixed_json.count('{') - fixed_json.count('}')
            open_brackets = fixed_json.count('[') - fixed_json.count(']')
            fixed_json = fixed_json + (']' * open_brackets) + ('}' * open_braces)
            try:
                parsed_result = json.loads(fixed_json)
            except json.JSONDecodeError:
                fallback_hazards = extract_hazards_from_text(raw_response_text)
                if fallback_hazards:
                    parsed_result = {"hazards": fallback_hazards, "room_type": "unknown"}
                else:
                    parsed_result = {"hazards": [], "room_type": "unknown"}

        hazards = parsed_result.get('hazards', [])
        for hazard in hazards:
            hazard.setdefault('category', 'general')
            hazard.setdefault('region', 'center')
            hazard.setdefault('confidence', 0.7)
            hazard.setdefault('recommendation', 'Address this hazard to reduce fall risk')

        filtered_hazards = filter_hazards(hazards, CONFIDENCE_THRESHOLD, MAX_HAZARDS)
        risk_score, risk_level = calculate_risk_score(filtered_hazards)

        return {
            "hazards": filtered_hazards,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "room_type": parsed_result.get('room_type', 'unknown'),
            "total_hazards": len(filtered_hazards),
            "model_used": model,
        }

    except requests.exceptions.Timeout:
        print("    [TIMEOUT] Analysis timed out")
        return None
    except Exception as e:
        print(f"    [ERROR] {e}")
        return None


# ─── Main classification logic ───

def discover_images(source_dir: Path) -> list:
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    images = sorted([
        f for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ])
    return images


def main():
    parser = argparse.ArgumentParser(description='Classify PHELE images into risk categories')
    parser.add_argument('--source', type=str, default='data/raw/phele/test',
                        help='Source image folder (default: data/raw/phele/test)')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='Output folder (default: data/processed)')
    parser.add_argument('--model', type=str, default='llava:7b',
                        help='Ollama model name (default: llava:7b)')
    parser.add_argument('--threshold', type=int, default=25,
                        help='Risk score cutoff for high risk (default: 25)')
    parser.add_argument('--max-images', type=int, default=0,
                        help='Max images to process (0 = all)')
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    source_dir = project_root / args.source
    output_dir = project_root / args.output
    low_risk_dir = output_dir / 'low_risk'
    high_risk_dir = output_dir / 'high_risk'
    csv_path = output_dir / 'classification_results.csv'

    # Validate source
    if not source_dir.exists():
        print(f"Error: Source folder not found: {source_dir}")
        sys.exit(1)

    # Check Ollama
    print("Checking Ollama status...")
    if not check_ollama_status():
        print("Error: Ollama is not running. Start it with: ollama serve")
        sys.exit(1)
    print(f"Ollama is running. Using model: {args.model}")

    # Discover images
    images = discover_images(source_dir)
    if args.max_images > 0:
        images = images[:args.max_images]
    total = len(images)
    print(f"Found {total} images in {source_dir}")

    if total == 0:
        print("No images found. Exiting.")
        sys.exit(0)

    # Create output folders
    low_risk_dir.mkdir(parents=True, exist_ok=True)
    high_risk_dir.mkdir(parents=True, exist_ok=True)

    # Check for already processed (resume support)
    already_done = set()
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                already_done.add(row['image'])
        print(f"Resuming: {len(already_done)} images already processed")

    # Process images
    results = []
    low_count = 0
    high_count = 0
    fail_count = 0
    times = []

    # Open CSV for appending
    write_header = not csv_path.exists() or len(already_done) == 0
    csv_file = open(csv_path, 'a', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=[
        'image', 'risk_score', 'risk_level', 'total_hazards', 'room_type',
        'hazard_categories', 'time_seconds', 'classification'
    ])
    if write_header:
        writer.writeheader()

    print(f"\nProcessing {total} images (threshold: {args.threshold})...\n")

    for i, image_path in enumerate(images, 1):
        name = image_path.name

        # Skip already processed
        if name in already_done:
            print(f"  [{i}/{total}] {name} — skipped (already processed)")
            continue

        start = time.time()
        print(f"  [{i}/{total}] {name} ... ", end='', flush=True)

        # Read and analyze
        image_bytes = image_path.read_bytes()
        result = analyze_image(image_bytes, model=args.model)
        elapsed = time.time() - start
        times.append(elapsed)

        if result is None:
            print(f"FAILED ({elapsed:.1f}s)")
            fail_count += 1
            writer.writerow({
                'image': name, 'risk_score': '', 'risk_level': 'ERROR',
                'total_hazards': '', 'room_type': '', 'hazard_categories': '',
                'time_seconds': f'{elapsed:.1f}', 'classification': 'error'
            })
            csv_file.flush()
            continue

        score = result['risk_score']
        level = result['risk_level']
        hazards = result['total_hazards']
        room = result['room_type']

        # Classify
        if score <= args.threshold:
            classification = 'low_risk'
            dest = low_risk_dir / name
            low_count += 1
        else:
            classification = 'high_risk'
            dest = high_risk_dir / name
            high_count += 1

        # Copy image
        shutil.copy2(image_path, dest)

        # Get hazard categories for CSV
        categories = ', '.join(set(h.get('category', '') for h in result['hazards']))

        print(f"{level} (score: {score}, hazards: {hazards}) [{elapsed:.1f}s]")

        writer.writerow({
            'image': name, 'risk_score': score, 'risk_level': level,
            'total_hazards': hazards, 'room_type': room,
            'hazard_categories': categories,
            'time_seconds': f'{elapsed:.1f}', 'classification': classification
        })
        csv_file.flush()

    csv_file.close()

    # Summary
    processed = low_count + high_count + fail_count
    avg_time = sum(times) / len(times) if times else 0

    print(f"\n{'='*50}")
    print(f"CLASSIFICATION COMPLETE")
    print(f"{'='*50}")
    print(f"Total processed:  {processed}")
    print(f"  Low risk:       {low_count}")
    print(f"  High risk:      {high_count}")
    print(f"  Failed:         {fail_count}")
    print(f"Avg time/image:   {avg_time:.1f}s")
    print(f"Total time:       {sum(times):.0f}s")
    print(f"\nResults saved to: {csv_path}")
    print(f"Low risk images:  {low_risk_dir}")
    print(f"High risk images: {high_risk_dir}")


if __name__ == '__main__':
    main()
