"""
Fall Hazard Detection Web Application

A Streamlit-based UI for uploading images and detecting
fall hazards for elderly home safety assessment.

Uses Ollama with LLaVA model for FREE, LOCAL inference.
Research project - evaluates pre-trained VLM capabilities.

Run with: streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path
import json
import base64
import re
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configuration
MAX_HAZARDS = 10  # Limit to top 10 hazards
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to show

# Page configuration
st.set_page_config(
    page_title="Fall Hazard Detection System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .risk-moderate {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .risk-critical {
        background-color: #721c24;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stat-card {
        background-color: #e9ecef;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def preprocess_image(image_bytes: bytes, target_brightness: float = 0.5) -> bytes:
    """Preprocess image for consistent analysis."""
    try:
        img = Image.open(BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_array = np.array(img, dtype=np.float32)
        current_brightness = np.mean(img_array) / 255.0

        if current_brightness < 0.3:
            brightness_factor = min(target_brightness / max(current_brightness, 0.1), 2.0)
            img_array = img_array * brightness_factor
        elif current_brightness > 0.7:
            brightness_factor = target_brightness / current_brightness
            img_array = img_array * brightness_factor

        # Contrast enhancement
        for channel in range(3):
            ch = img_array[:, :, channel]
            ch_min, ch_max = np.percentile(ch, [2, 98])
            if ch_max > ch_min:
                ch = (ch - ch_min) / (ch_max - ch_min) * 255
                img_array[:, :, channel] = ch

        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        processed_img = Image.fromarray(img_array)

        # Resize if too large
        max_size = 1024
        if max(processed_img.size) > max_size:
            ratio = max_size / max(processed_img.size)
            new_size = (int(processed_img.width * ratio), int(processed_img.height * ratio))
            processed_img = processed_img.resize(new_size, Image.LANCZOS)

        buffer = BytesIO()
        processed_img.save(buffer, format='JPEG', quality=85)
        return buffer.getvalue()
    except Exception as e:
        return image_bytes


def get_image_stats(image_bytes: bytes) -> dict:
    """Get image statistics."""
    try:
        img = Image.open(BytesIO(image_bytes))
        img_array = np.array(img)
        return {
            "width": img.width,
            "height": img.height,
            "brightness": round(np.mean(img_array) / 255.0, 2),
            "contrast": round(np.std(img_array) / 255.0, 2),
        }
    except Exception:
        return {}


# =============================================================================
# REGION-BASED VISUALIZATION (More reliable than bounding boxes)
# =============================================================================

def draw_region_highlights(image_bytes: bytes, hazards: list) -> bytes:
    """
    Draw region-based highlights instead of exact circles.
    Uses 9-region grid: top-left, top-center, top-right, etc.
    """
    try:
        img = Image.open(BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Create overlay for semi-transparent regions
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        width, height = img.size

        # Define 9 regions
        regions = {
            "top-left": (0, 0, width//3, height//3),
            "top-center": (width//3, 0, 2*width//3, height//3),
            "top-right": (2*width//3, 0, width, height//3),
            "center-left": (0, height//3, width//3, 2*height//3),
            "center": (width//3, height//3, 2*width//3, 2*height//3),
            "center-right": (2*width//3, height//3, width, 2*height//3),
            "bottom-left": (0, 2*height//3, width//3, height),
            "bottom-center": (width//3, 2*height//3, 2*width//3, height),
            "bottom-right": (2*width//3, 2*height//3, width, height),
            # Additional mappings for common descriptions
            "floor": (0, 2*height//3, width, height),
            "ceiling": (0, 0, width, height//4),
            "left": (0, 0, width//3, height),
            "right": (2*width//3, 0, width, height),
            "top": (0, 0, width, height//3),
            "bottom": (0, 2*height//3, width, height),
        }

        # Severity colors with alpha
        severity_colors = {
            "low": (255, 255, 0, 60),       # Yellow
            "medium": (255, 165, 0, 80),     # Orange
            "high": (255, 69, 0, 100),       # Red-Orange
            "critical": (255, 0, 0, 120)     # Red
        }

        # Track which regions have hazards
        region_hazards = {}

        for hazard in hazards:
            region_text = hazard.get('region', 'center').lower()
            severity = hazard.get('severity', 'medium').lower()

            # Find matching region
            matched_region = None
            for region_name in regions:
                if region_name in region_text:
                    matched_region = region_name
                    break

            if not matched_region:
                # Default based on keywords
                if any(w in region_text for w in ['floor', 'ground', 'bottom', 'rug', 'mat']):
                    matched_region = 'bottom'
                elif any(w in region_text for w in ['ceiling', 'light', 'top']):
                    matched_region = 'top'
                elif any(w in region_text for w in ['left', 'side']):
                    matched_region = 'left'
                elif any(w in region_text for w in ['right']):
                    matched_region = 'right'
                else:
                    matched_region = 'center'

            if matched_region not in region_hazards:
                region_hazards[matched_region] = severity
            elif severity_colors.get(severity, (0,0,0,0))[3] > severity_colors.get(region_hazards[matched_region], (0,0,0,0))[3]:
                region_hazards[matched_region] = severity

        # Draw region highlights
        for region_name, severity in region_hazards.items():
            if region_name in regions:
                color = severity_colors.get(severity, (255, 0, 0, 80))
                bbox = regions[region_name]
                draw.rectangle(bbox, fill=color, outline=(255, 0, 0, 200), width=3)

        # Composite overlay onto image
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        img = img.convert('RGB')

        # Add legend
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()

        legend_draw = ImageDraw.Draw(img)
        legend_y = 10
        legend_draw.rectangle([10, legend_y, 200, legend_y + 25], fill=(255, 255, 255, 200))
        legend_draw.text((15, legend_y + 5), f"Hazards detected: {len(hazards)}", fill=(0, 0, 0), font=font)

        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        return buffer.getvalue()

    except Exception as e:
        st.warning(f"Could not draw highlights: {str(e)}")
        return image_bytes


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def get_risk_color(risk_level: str) -> str:
    colors = {
        "low": "risk-low",
        "moderate": "risk-moderate",
        "high": "risk-high",
        "critical": "risk-critical"
    }
    return colors.get(risk_level.lower(), "risk-moderate")


def get_severity_emoji(severity: str) -> str:
    emojis = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
    return emojis.get(severity.lower(), "⚪")


def calculate_risk_score(hazards: list) -> tuple:
    """Calculate risk score from hazards using elderly-vulnerability-aware scoring.

    Scoring considers that an older adult living alone may have multiple
    vulnerabilities (mobility impairment, vision/hearing loss, cognitive decline)
    that compound the danger of each individual hazard. The algorithm applies:
    - Per-hazard weighted scores based on severity and category
    - Cumulative risk multiplier (more hazards = disproportionately more danger)
    - High-severity bonus (critical/high hazards carry extra weight)
    - Minimum floor per hazard to prevent under-scoring
    """
    if not hazards:
        return 0, "LOW"

    severity_weights = {"low": 0.4, "medium": 0.65, "high": 0.85, "critical": 1.0}
    category_weights = {
        "fire": 1.0, "electrical": 1.0, "structural": 0.95,
        "stairs": 1.0, "bathroom": 0.95, "flooring": 0.90, "obstacles": 0.90,
        "lighting": 0.85, "furniture": 0.80, "kitchen": 0.85, "bedroom": 0.75,
        "external": 0.85, "general": 0.70
    }

    # Base multiplier per hazard — higher than before to spread the 0-100 range
    BASE_MULTIPLIER = 30

    total_score = 0
    high_critical_count = 0

    for hazard in hazards:
        severity = hazard.get('severity', 'medium').lower()
        category = hazard.get('category', 'general').lower()
        sev_weight = severity_weights.get(severity, 0.5)
        cat_weight = category_weights.get(category, 0.7)

        # Each hazard contributes a meaningful amount
        hazard_score = sev_weight * cat_weight * BASE_MULTIPLIER
        # Minimum floor: every detected hazard adds at least 5 points
        hazard_score = max(hazard_score, 5.0)
        total_score += hazard_score

        if severity in ('high', 'critical'):
            high_critical_count += 1

    # Cumulative risk multiplier: multiple hazards compound danger
    # 1-2 hazards: no bonus; 3-5: +15%; 6-8: +30%; 9+: +45%
    hazard_count = len(hazards)
    if hazard_count >= 9:
        cumulative_multiplier = 1.45
    elif hazard_count >= 6:
        cumulative_multiplier = 1.30
    elif hazard_count >= 3:
        cumulative_multiplier = 1.15
    else:
        cumulative_multiplier = 1.0

    total_score *= cumulative_multiplier

    # High-severity bonus: extra points when multiple high/critical hazards exist
    if high_critical_count >= 3:
        total_score += 15
    elif high_critical_count >= 2:
        total_score += 10
    elif high_critical_count >= 1:
        total_score += 5

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


def check_ollama_status() -> bool:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except (requests.exceptions.RequestException, OSError):
        return False


def get_available_models() -> list:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return [m['name'] for m in response.json().get('models', [])]
        return []
    except (requests.exceptions.RequestException, OSError):
        return []


def filter_hazards(hazards: list, confidence_threshold: float, max_hazards: int) -> list:
    """Filter hazards by confidence and limit count."""
    # Filter by confidence
    filtered = [h for h in hazards if h.get('confidence', 0.5) >= confidence_threshold]

    # Sort by severity (critical first) then confidence
    severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
    filtered.sort(key=lambda h: (
        severity_order.get(h.get('severity', 'medium').lower(), 2),
        -h.get('confidence', 0.5)
    ))

    # Limit to max
    return filtered[:max_hazards]


def extract_hazards_from_text(text: str) -> list:
    """Fallback: extract hazards from plain text when JSON parsing fails.

    Uses two tiers of keywords:
    - Direct hazard words (fire, clutter, crack) → always treated as hazards
    - Context-dependent words (floor, bath, stair) → only hazards when paired
      with a danger modifier (broken, missing, no, damaged, wet, loose, etc.)
    """
    # Words that are ALWAYS hazards when mentioned
    direct_hazard_keywords = {
        'fire': ('fire', 'critical'), 'flame': ('fire', 'critical'),
        'burn': ('fire', 'critical'), 'smoke': ('fire', 'critical'),
        'blaze': ('fire', 'critical'), 'ignit': ('fire', 'critical'),
        'clutter': ('obstacles', 'critical'), 'cluttered': ('obstacles', 'critical'),
        'debris': ('obstacles', 'high'), 'mess': ('obstacles', 'high'),
        'trip': ('obstacles', 'high'), 'tripping': ('obstacles', 'high'),
        'obstruct': ('obstacles', 'high'), 'blocked': ('obstacles', 'high'),
        'slip': ('flooring', 'critical'), 'slippery': ('flooring', 'critical'),
        'collapse': ('structural', 'critical'), 'collapsed': ('structural', 'critical'),
        'mold': ('structural', 'high'), 'moldy': ('structural', 'high'),
        'overload': ('electrical', 'high'), 'spark': ('electrical', 'critical'),
        'exposed wire': ('electrical', 'critical'),
        'loose rug': ('flooring', 'high'), 'loose carpet': ('flooring', 'high'),
        'no grab bar': ('bathroom', 'critical'), 'no handrail': ('stairs', 'critical'),
        'missing handrail': ('stairs', 'critical'), 'missing grab bar': ('bathroom', 'critical'),
    }

    # Words that are only hazards when near a danger modifier
    context_keywords = {
        'floor': 'flooring', 'tile': 'flooring', 'carpet': 'flooring',
        'rug': 'flooring', 'mat': 'flooring', 'surface': 'flooring',
        'stair': 'stairs', 'step': 'stairs', 'railing': 'stairs', 'handrail': 'stairs',
        'shower': 'bathroom', 'tub': 'bathroom', 'bath': 'bathroom',
        'toilet': 'bathroom', 'grab bar': 'bathroom',
        'light': 'lighting', 'lamp': 'lighting',
        'chair': 'furniture', 'table': 'furniture', 'bed': 'furniture',
        'couch': 'furniture', 'furniture': 'furniture',
        'cord': 'obstacles', 'cable': 'obstacles', 'wire': 'obstacles',
        'outlet': 'electrical', 'wiring': 'electrical', 'electric': 'electrical',
        'wall': 'structural', 'ceiling': 'structural', 'door': 'structural',
    }

    # Modifiers that indicate an actual hazard (not just a room description)
    danger_modifiers = {
        'broken', 'damaged', 'cracked', 'missing', 'no ', 'without',
        'wet', 'slippery', 'loose', 'torn', 'worn', 'unstable', 'wobbly',
        'dark', 'dim', 'poor', 'inadequate', 'blocked', 'cluttered',
        'exposed', 'frayed', 'faulty', 'leaking', 'rusty', 'sharp',
        'hazard', 'danger', 'risk', 'unsafe', 'fall',
    }

    # Negation phrases — if these appear near the keyword, it's NOT a hazard
    negation_patterns = [
        'no ', 'not ', 'no visible', 'none', "don't", "doesn't", "isn't",
        'without any', 'free of', 'absence of', 'lack of', 'well maintained',
        'well-maintained', 'clean', 'tidy', 'organized', 'no sign',
        'not appear', 'not see', 'not detect', 'not find', 'not observe',
    ]

    def has_negation(sentence_lower: str, keyword: str) -> bool:
        """Check if the keyword is negated in the sentence."""
        # Find position of keyword
        idx = sentence_lower.find(keyword)
        # Check the 40 characters before the keyword for negation
        prefix = sentence_lower[max(0, idx - 40):idx]
        return any(neg in prefix for neg in negation_patterns)

    sentences = re.split(r'[.!?\n]', text)
    found_hazards = []
    seen_keywords = set()

    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        if not sentence_lower:
            continue

        # Check direct hazard keywords first
        matched = False
        for keyword, (category, severity) in direct_hazard_keywords.items():
            if keyword in sentence_lower and keyword not in seen_keywords:
                # Skip if keyword is negated
                if has_negation(sentence_lower, keyword):
                    continue
                seen_keywords.add(keyword)
                found_hazards.append({
                    'category': category,
                    'subcategory': keyword,
                    'severity': severity,
                    'description': sentence.strip(),
                    'region': 'center',
                    'confidence': 0.7,
                    'recommendation': f'Address {keyword} hazard to reduce risk for elderly resident'
                })
                matched = True
                break

        if matched:
            continue

        # Check context-dependent keywords — only if danger modifier present
        has_danger = any(mod in sentence_lower for mod in danger_modifiers)
        if not has_danger:
            continue

        for keyword, category in context_keywords.items():
            if keyword in sentence_lower and keyword not in seen_keywords:
                if has_negation(sentence_lower, keyword):
                    continue
                seen_keywords.add(keyword)
                # Determine severity based on modifier
                if any(m in sentence_lower for m in ('broken', 'missing', 'no ', 'exposed', 'blocked')):
                    severity = 'high'
                elif any(m in sentence_lower for m in ('wet', 'slippery', 'dark', 'dim', 'loose')):
                    severity = 'high'
                else:
                    severity = 'medium'
                found_hazards.append({
                    'category': category,
                    'subcategory': f'{sentence_lower.split(keyword)[0].strip().split()[-1] if sentence_lower.split(keyword)[0].strip() else ""} {keyword}'.strip(),
                    'severity': severity,
                    'description': sentence.strip(),
                    'region': 'center',
                    'confidence': 0.65,
                    'recommendation': f'Address {keyword} hazard to reduce risk for elderly resident'
                })
                break

    return found_hazards[:10]


def analyze_with_ollama(image_bytes: bytes, model: str = "llava:7b") -> dict:
    """Analyze image using Ollama with LLaVA model."""
    try:
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        is_moondream = 'moondream' in model.lower()

        if is_moondream:
            # Plain text prompt for Moondream — too small for JSON output
            prompt = """Describe this room. What safety hazards do you see that could hurt an elderly person? List each hazard on a separate line. For each hazard, say how dangerous it is: low, medium, high, or critical.

Look for: fire, smoke, clutter on floor, broken stairs, missing handrails, wet floors, dark areas, exposed wires, broken furniture, sharp objects, blocked doorways, damaged walls or ceiling.

Only describe what you actually see in this image."""
            ollama_temperature = 0.1
            ollama_num_predict = 800
        else:
            # Full prompt for LLaVA and other capable models
            ollama_temperature = 0.0
            ollama_num_predict = 1500
            prompt = """You are an expert home safety assessor conducting a comprehensive safety risk assessment for an elderly person who lives ALONE.

CRITICAL CONTEXT: The person living here may have one or more of these vulnerabilities:
- Limited mobility (uses walker, cane, or wheelchair)
- Poor vision or legally blind
- Hearing impairment or deaf
- Reduced balance and strength
- Cognitive decline or confusion
- Arthritis limiting grip strength
- Slow reaction time in emergencies

Your job is to identify EVERY safety hazard in this image — not just fall risks, but ALL dangers to an elderly person. Err on the side of caution — it is far more dangerous to MISS a hazard than to over-report one.

Examine the ENTIRE image systematically for these hazard types:

FIRE & BURN HAZARDS: open flames, unattended cooking, fire damage, smoke, flammable materials near heat sources, missing smoke detectors, candles, overheated appliances, hot surfaces without guards
FALL HAZARDS: clutter on floor, loose rugs, cords across paths, wet/slippery surfaces, poor lighting, missing grab bars/handrails, uneven flooring, obstacles in walkways
ELECTRICAL HAZARDS: exposed wiring, overloaded outlets, damaged cords, water near electronics
STRUCTURAL HAZARDS: broken furniture, damaged floors/walls, water damage, collapsed shelving, broken glass, exposed nails/screws
CHEMICAL/TOXIC HAZARDS: spills, mold, gas appliances without ventilation, unsanitary conditions
BLOCKED EXITS: furniture or objects blocking doorways, escape routes, or emergency access
SHARP/INJURY HAZARDS: broken glass, sharp edges, protruding objects at head/body height

For each hazard provide:
1. category: one of (bathroom, stairs, flooring, lighting, obstacles, furniture, kitchen, bedroom, external, general, fire, electrical, structural)
2. subcategory: specific hazard name (e.g., "kitchen fire", "exposed wiring", "floor clutter", "broken glass")
3. severity: low, medium, high, or critical
4. description: what you see and WHY it is dangerous for an elderly person
5. region: where in the image (e.g., "floor center", "left wall", "bottom right")
6. confidence: 0.0 to 1.0
7. recommendation: specific actionable fix

SEVERITY GUIDELINES — rate from the perspective of a vulnerable elderly person:
- "critical": immediate life-threatening danger (active fire, gas leak, structural collapse, blocked exit, exposed live wires)
- "high": serious injury risk (burn hazards, major clutter, no grab bars, exposed sharp edges, water near electricity)
- "medium": moderate risk (dim lighting, minor clutter, worn surfaces, cosmetic damage)
- "low": minor concern that still warrants attention

IMPORTANT:
- Report up to 10 hazards — be thorough about what is ACTUALLY VISIBLE
- Fire, smoke, and burn hazards are ALWAYS critical or high severity
- Clutter, mess, and disorganization are SERIOUS hazards for elderly people
- Do NOT hallucinate or invent objects that are not in the image — only describe what you can genuinely see
- Use specific, descriptive subcategory names (e.g., "active stove fire" not just "kitchen")

CRITICAL — AVOID FALSE POSITIVES:
- A shower, bathtub, toilet, sink, bed, table, or chair is NOT a hazard by itself
- Only flag these if they are BROKEN, DAMAGED, or specifically MISSING a safety feature (e.g., "shower without grab bar", "stairs without handrail")
- A clean, well-maintained room with normal furniture should have FEW or ZERO hazards
- If the room looks clean, organized, and well-lit, it is LOW risk — return an empty hazards list or only minor hazards
- Do NOT invent hazards like "clutter" or "tripping hazard" in a tidy, organized room
- Normal furniture arrangement is NOT clutter. Normal room items are NOT obstacles.
- If you cannot clearly see a specific hazard, do NOT report it
- It is BETTER to return zero hazards for a safe room than to fabricate false ones

Identify the room type first.

Respond ONLY with valid JSON (no other text):
{
    "room_type": "bathroom/kitchen/bedroom/living_room/stairs/hallway",
    "hazards": [
        {
            "category": "category",
            "subcategory": "specific hazard",
            "severity": "low/medium/high/critical",
            "description": "what you see and why it's dangerous",
            "region": "where in image",
            "confidence": 0.8,
            "recommendation": "how to fix"
        }
    ]
}"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": ollama_temperature,
                    "num_predict": ollama_num_predict,
                    "seed": 42
                }
            },
            timeout=120
        )

        if response.status_code != 200:
            return None

        result = response.json()
        response_text = result.get('response', '')
        raw_response_text = response_text  # Save before cleaning for debug

        # Clean response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            for part in response_text.split("```"):
                if "{" in part and "}" in part:
                    response_text = part
                    break

        # Extract and fix truncated JSON
        start_idx = response_text.find("{")
        if start_idx == -1:
            # No JSON found — use text fallback (common for Moondream plain text responses)
            fallback_hazards = extract_hazards_from_text(raw_response_text)
            if fallback_hazards:
                parsed_result = {"hazards": fallback_hazards, "room_type": "unknown"}
            else:
                parsed_result = {"hazards": [], "room_type": "unknown"}
            # Skip JSON parsing, go straight to backfill
            hazards = parsed_result.get('hazards', [])
            for hazard in hazards:
                hazard.setdefault('category', 'general')
                hazard.setdefault('region', 'center')
                hazard.setdefault('confidence', 0.7)
                hazard.setdefault('recommendation', 'Address this hazard to reduce fall risk')
            filtered_hazards = filter_hazards(hazards, CONFIDENCE_THRESHOLD, MAX_HAZARDS)
            risk_score, risk_level = calculate_risk_score(filtered_hazards)
            recommendations = []
            for i, hazard in enumerate(filtered_hazards[:5], 1):
                sev = hazard.get('severity', 'medium').upper()
                rec = hazard.get('recommendation', 'Address this hazard')
                recommendations.append(f"Priority {i} ({sev}): {rec}")
            return {
                "hazards": filtered_hazards,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "room_type": parsed_result.get("room_type", "unknown"),
                "total_hazards": len(filtered_hazards),
                "recommendations": recommendations,
                "model_used": model,
                "raw_hazard_count": len(hazards),
                "raw_response": raw_response_text
            }

        json_str = response_text[start_idx:]

        # Try to parse as-is first (works for LLaVA's well-formed JSON)
        try:
            parsed_result = json.loads(json_str)
        except json.JSONDecodeError:
            # Moondream and other models may return truncated JSON
            # Attempt to fix by finding last complete JSON value and closing brackets

            # Find all complete key-value pairs ending with proper JSON values
            # This regex matches complete "key": value patterns
            fixed_json = json_str

            # Remove any incomplete trailing content after last comma
            # Find the last valid JSON segment (ending with }, ], number, string, bool, or null)
            last_valid = None
            for match in re.finditer(r'[}\]"\d]|true|false|null', json_str):
                last_valid = match.end()

            if last_valid:
                fixed_json = json_str[:last_valid]

            # If ends mid-string (odd number of quotes after last complete value), trim
            trailing = fixed_json[fixed_json.rfind(',') + 1:] if ',' in fixed_json else fixed_json
            if trailing.count('"') % 2 == 1:
                # Incomplete string, trim back to last comma
                last_comma = fixed_json.rfind(',')
                if last_comma > 0:
                    fixed_json = fixed_json[:last_comma]

            # Remove trailing comma if present
            fixed_json = fixed_json.rstrip().rstrip(',')

            # Count and add missing brackets
            open_braces = fixed_json.count('{') - fixed_json.count('}')
            open_brackets = fixed_json.count('[') - fixed_json.count(']')

            fixed_json = fixed_json + (']' * open_brackets) + ('}' * open_braces)

            try:
                parsed_result = json.loads(fixed_json)
            except json.JSONDecodeError:
                # Last resort: try extracting hazards from plain text
                fallback_hazards = extract_hazards_from_text(raw_response_text)
                if fallback_hazards:
                    st.warning("Model returned non-JSON response. Extracted hazards from text.")
                    parsed_result = {"hazards": fallback_hazards, "room_type": "unknown"}
                else:
                    st.warning("Model returned incomplete response. Showing partial results.")
                    parsed_result = {"hazards": [], "room_type": "unknown"}

        # Backfill default fields for models with simpler prompts (e.g. Moondream)
        hazards = parsed_result.get('hazards', [])
        for hazard in hazards:
            hazard.setdefault('category', 'general')
            hazard.setdefault('region', 'center')
            hazard.setdefault('confidence', 0.7)
            hazard.setdefault('recommendation', 'Address this hazard to reduce fall risk')

        # Filter hazards
        filtered_hazards = filter_hazards(hazards, CONFIDENCE_THRESHOLD, MAX_HAZARDS)

        risk_score, risk_level = calculate_risk_score(filtered_hazards)

        recommendations = []
        for i, hazard in enumerate(filtered_hazards[:5], 1):
            sev = hazard.get('severity', 'medium').upper()
            rec = hazard.get('recommendation', 'Address this hazard')
            recommendations.append(f"Priority {i} ({sev}): {rec}")

        return {
            "hazards": filtered_hazards,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "room_type": parsed_result.get('room_type', 'unknown'),
            "total_hazards": len(filtered_hazards),
            "recommendations": recommendations,
            "model_used": model,
            "raw_hazard_count": len(hazards),
            "raw_response": raw_response_text
        }

    except json.JSONDecodeError as e:
        st.error(f"Error parsing response: {str(e)}")
        return None
    except requests.exceptions.Timeout:
        st.error("Analysis timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None


def main():
    # Allow sidebar to update global thresholds
    global CONFIDENCE_THRESHOLD, MAX_HAZARDS

    st.markdown('<h1 class="main-header">🏠 Fall Hazard Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Home Safety Assessment for Older Adults</p>', unsafe_allow_html=True)

    ollama_running = check_ollama_status()
    available_models = get_available_models() if ollama_running else []
    vision_models = [m for m in available_models if any(v in m.lower() for v in ['llava', 'bakllava', 'llama3.2-vision', 'moondream'])]

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("Model Status")
        if ollama_running:
            st.success("✅ Ollama running (FREE, LOCAL)")
            if vision_models:
                # Check if both llava and moondream are available for comparison
                has_llava = any('llava' in m.lower() for m in vision_models)
                has_moondream = any('moondream' in m.lower() for m in vision_models)

                if has_llava and has_moondream:
                    comparison_mode = st.checkbox("🔄 Compare Models", value=False,
                                                  help="Run both LLaVA and Moondream side-by-side")
                else:
                    comparison_mode = False

                if not comparison_mode:
                    selected_model = st.selectbox("Vision Model", vision_models, index=0)
                else:
                    selected_model = None  # Will use both models
                    st.info("Will compare: LLaVA vs Moondream")
            else:
                st.warning("No vision models. Run: `ollama pull llava:7b`")
                selected_model = "llava:7b"
                comparison_mode = False
        else:
            st.error("❌ Ollama not running")
            st.code("brew services start ollama")
            selected_model = None
            comparison_mode = False

        st.divider()

        st.subheader("Detection Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.9, CONFIDENCE_THRESHOLD, 0.1,
                                        help="Higher = fewer but more confident detections")
        max_hazards = st.slider("Max Hazards to Show", 3, 10, MAX_HAZARDS,
                               help="Limit number of hazards displayed")

        enable_preprocessing = st.checkbox("Auto-preprocessing", value=True)

        st.divider()

        st.subheader("About")
        st.markdown("""
        **Research Project**: Evaluating VLM capabilities for home hazard detection.

        **Based on**: Westmead Home Safety Assessment

        **Note**: This uses a pre-trained model. Results are approximate and should be validated.
        """)

        st.divider()
        st.subheader("Risk Levels")
        st.markdown("""
        - 🟢 **LOW** (0-25)
        - 🟡 **MODERATE** (26-50)
        - 🟠 **HIGH** (51-75)
        - 🔴 **CRITICAL** (76-100)
        """)

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Image")

        uploaded_file = st.file_uploader(
            "Choose an image of a room",
            type=['jpg', 'jpeg', 'png', 'webp']
        )

        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()

            # Clear old results when a new image is uploaded
            current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get('last_file_id') != current_file_id:
                st.session_state['last_file_id'] = current_file_id
                st.session_state.pop('results', None)
                st.session_state.pop('comparison_results', None)
                st.session_state.pop('comparison_mode', None)
                st.session_state.pop('analyzed', None)
                st.session_state.pop('processed_image', None)
            stats = get_image_stats(image_bytes)

            if enable_preprocessing:
                processed_bytes = preprocess_image(image_bytes)
            else:
                processed_bytes = image_bytes

            # Show image with or without highlights
            if 'results' in st.session_state and st.session_state.get('analyzed') and 'processed_image' in st.session_state:
                hazards = st.session_state['results'].get('hazards', [])
                if hazards:
                    annotated_bytes = draw_region_highlights(
                        st.session_state['processed_image'],
                        hazards
                    )
                    st.image(annotated_bytes, caption="🔴 Hazard Regions Highlighted", use_container_width=True)
                else:
                    st.image(processed_bytes, caption="✅ No Hazards Detected", use_container_width=True)

                with st.expander("📷 Original Image"):
                    st.image(processed_bytes, use_container_width=True)
            else:
                st.image(processed_bytes, caption="Ready for Analysis", use_container_width=True)

            with st.expander("📊 Image Stats"):
                c1, c2 = st.columns(2)
                c1.metric("Brightness", f"{stats.get('brightness', 'N/A')}")
                c2.metric("Contrast", f"{stats.get('contrast', 'N/A')}")

            if not ollama_running:
                st.error("Start Ollama: `brew services start ollama`")
            elif not vision_models:
                st.warning("Pull model: `ollama pull llava:7b`")
            else:
                button_text = "🔍 Compare Models" if comparison_mode else "🔍 Analyze for Hazards"
                if st.button(button_text, type="primary", use_container_width=True):
                    # Update thresholds from sidebar
                    CONFIDENCE_THRESHOLD = confidence_threshold
                    MAX_HAZARDS = max_hazards

                    if comparison_mode:
                        # Run both models for comparison
                        import time

                        # Find the actual model names
                        llava_model = next((m for m in vision_models if 'llava' in m.lower()), 'llava:7b')
                        moondream_model = next((m for m in vision_models if 'moondream' in m.lower()), 'moondream')

                        with st.spinner(f"🤖 Running {llava_model}..."):
                            start_time = time.time()
                            llava_results = analyze_with_ollama(processed_bytes, llava_model)
                            llava_time = time.time() - start_time

                        with st.spinner(f"🤖 Running {moondream_model}..."):
                            start_time = time.time()
                            moondream_results = analyze_with_ollama(processed_bytes, moondream_model)
                            moondream_time = time.time() - start_time

                        if llava_results or moondream_results:
                            st.session_state['comparison_results'] = {
                                'llava': {'results': llava_results, 'time': llava_time, 'model': llava_model},
                                'moondream': {'results': moondream_results, 'time': moondream_time, 'model': moondream_model}
                            }
                            st.session_state['comparison_mode'] = True
                            st.session_state['analyzed'] = True
                            st.session_state['processed_image'] = processed_bytes
                            st.success("✅ Comparison complete!")
                            st.rerun()
                        else:
                            st.error("Both models failed. Try again.")
                    else:
                        # Single model analysis
                        with st.spinner("🤖 Analyzing... (30-60 seconds)"):
                            results = analyze_with_ollama(processed_bytes, selected_model)

                            if results:
                                st.session_state['results'] = results
                                st.session_state['comparison_mode'] = False
                                st.session_state['analyzed'] = True
                                st.session_state['processed_image'] = processed_bytes
                                st.success("✅ Analysis complete!")
                                st.rerun()
                            else:
                                st.error("Analysis failed. Try again.")
        else:
            st.info("👆 Upload an image to begin")

            with st.expander("📸 Tips"):
                st.markdown("""
                - Good lighting
                - Capture full room
                - Include floors and walls
                """)

    with col2:
        st.header("📊 Results")

        # Check if we have comparison results
        if st.session_state.get('comparison_mode') and 'comparison_results' in st.session_state:
            comparison = st.session_state['comparison_results']

            # Comparison summary table
            st.subheader("🔄 Model Comparison")

            llava_data = comparison.get('llava', {})
            moondream_data = comparison.get('moondream', {})
            llava_results = llava_data.get('results') or {}
            moondream_results = moondream_data.get('results') or {}

            # Metrics comparison
            comp_col1, comp_col2 = st.columns(2)

            with comp_col1:
                st.markdown("### LLaVA")
                if llava_results:
                    st.metric("Hazards", llava_results.get('total_hazards', 0))
                    st.metric("Risk Score", f"{llava_results.get('risk_score', 0)}/100")
                    st.caption(f"⏱️ {llava_data.get('time', 0):.1f}s")
                else:
                    st.warning("Analysis failed")

            with comp_col2:
                st.markdown("### Moondream")
                if moondream_results:
                    st.metric("Hazards", moondream_results.get('total_hazards', 0))
                    st.metric("Risk Score", f"{moondream_results.get('risk_score', 0)}/100")
                    st.caption(f"⏱️ {moondream_data.get('time', 0):.1f}s")
                else:
                    st.warning("Analysis failed")

            st.divider()

            # Side-by-side hazard details
            st.subheader("🚨 Detected Hazards")

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.markdown("**LLaVA Detections:**")
                if llava_results and llava_results.get('hazards'):
                    for i, hazard in enumerate(llava_results['hazards'], 1):
                        severity = hazard.get('severity', 'medium').lower()
                        emoji = get_severity_emoji(severity)
                        subcategory = hazard.get('subcategory', 'Unknown')
                        st.markdown(f"{emoji} {subcategory} ({severity})")
                else:
                    st.caption("No hazards detected")

            with detail_col2:
                st.markdown("**Moondream Detections:**")
                if moondream_results and moondream_results.get('hazards'):
                    for i, hazard in enumerate(moondream_results['hazards'], 1):
                        severity = hazard.get('severity', 'medium').lower()
                        emoji = get_severity_emoji(severity)
                        subcategory = hazard.get('subcategory', 'Unknown')
                        st.markdown(f"{emoji} {subcategory} ({severity})")
                else:
                    st.caption("No hazards detected")

            # Debug: show raw model responses
            with st.expander("Debug: Raw Model Responses"):
                debug_col1, debug_col2 = st.columns(2)
                with debug_col1:
                    st.markdown("**LLaVA Raw Response:**")
                    st.code(llava_results.get('raw_response', 'N/A')[:2000] if llava_results else 'N/A', language='json')
                with debug_col2:
                    st.markdown("**Moondream Raw Response:**")
                    st.code(moondream_results.get('raw_response', 'N/A')[:2000] if moondream_results else 'N/A', language='json')

            st.divider()

            # Download combined report
            combined_report = {
                'comparison': True,
                'llava': llava_results,
                'moondream': moondream_results,
                'timing': {
                    'llava_seconds': llava_data.get('time', 0),
                    'moondream_seconds': moondream_data.get('time', 0)
                }
            }
            st.download_button(
                "📥 Download Comparison Report",
                json.dumps(combined_report, indent=2),
                "comparison_report.json",
                "application/json",
                use_container_width=True
            )

        elif 'results' in st.session_state and st.session_state.get('analyzed'):
            results = st.session_state['results']

            risk_class = get_risk_color(results['risk_level'])
            st.markdown(f"""
            <div class="{risk_class}">
                <h2 style="margin:0;">Risk Score: {results['risk_score']}/100</h2>
                <h3 style="margin:0;">{results['risk_level']}</h3>
            </div>
            """, unsafe_allow_html=True)

            st.write("")

            c1, c2, c3 = st.columns(3)
            c1.metric("Hazards", results['total_hazards'])
            c2.metric("Room", results['room_type'].title())
            critical = sum(1 for h in results['hazards'] if h.get('severity', '').lower() == 'critical')
            c3.metric("Critical", critical)

            st.divider()
            st.caption(f"Model: {results.get('model_used', 'llava:7b')} | Filtered from {results.get('raw_hazard_count', '?')} detections")

            st.subheader("🚨 Detected Hazards")

            if results['hazards']:
                for i, hazard in enumerate(results['hazards'], 1):
                    severity = hazard.get('severity', 'medium').lower()
                    emoji = get_severity_emoji(severity)
                    subcategory = hazard.get('subcategory', 'Unknown')
                    conf = hazard.get('confidence', 0.8)

                    with st.expander(f"{emoji} {i}. {subcategory} ({severity.upper()}) - {conf*100:.0f}%",
                                    expanded=(severity == 'critical')):
                        st.markdown(f"**Category:** {hazard.get('category', 'N/A').title()}")
                        st.markdown(f"**Location:** {hazard.get('region', 'N/A')}")
                        st.markdown(f"**Description:** {hazard.get('description', 'N/A')}")
                        st.info(f"💡 {hazard.get('recommendation', 'Address this hazard')}")
            else:
                st.success("✅ No significant hazards detected!")

            st.divider()

            if results.get('recommendations'):
                st.subheader("📋 Action Plan")
                for rec in results['recommendations']:
                    st.markdown(f"• {rec}")

            st.divider()
            st.download_button(
                "📥 Download Report (JSON)",
                json.dumps(results, indent=2),
                "hazard_report.json",
                "application/json",
                use_container_width=True
            )

        else:
            st.markdown("""
            <div class="stat-card">
                <h3>No Analysis Yet</h3>
                <p>Upload an image and click Analyze</p>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Fall Hazard Detection System | MSc Research Project | Ulster University</p>
        <p>⚠️ Research prototype - Results require clinical validation</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
