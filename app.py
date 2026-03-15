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
MAX_HAZARDS = 6  # Limit to top 6 hazards
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to show

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
    except:
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
        except:
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
    """Calculate risk score from hazards."""
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


def check_ollama_status() -> bool:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_available_models() -> list:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return [m['name'] for m in response.json().get('models', [])]
        return []
    except:
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
    """Fallback: extract hazards from plain text when JSON parsing fails."""
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

    text_lower = text.lower()
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
                    'category': category,
                    'subcategory': keyword,
                    'severity': severity,
                    'description': sentence.strip(),
                    'region': 'center',
                    'confidence': 0.65,
                    'recommendation': f'Address {keyword} hazard to reduce fall risk'
                })
                break  # one hazard per sentence

    return found_hazards[:6]


def analyze_with_ollama(image_bytes: bytes, model: str = "llava:7b") -> dict:
    """Analyze image using Ollama with LLaVA model."""
    try:
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        is_moondream = 'moondream' in model.lower()

        if is_moondream:
            # Simplified prompt for Moondream's smaller capacity
            prompt = """Look at this image of a room in a home. List any fall hazards you can see that could cause an elderly person to trip, slip, or fall.

For each hazard, state what it is, how dangerous it is (low, medium, high, or critical), and describe it briefly.

Respond in JSON format:
{"room_type": "room name", "hazards": [{"subcategory": "hazard name", "severity": "low/medium/high/critical", "description": "what you see"}]}

If no hazards are visible, respond: {"room_type": "room name", "hazards": []}"""
            ollama_temperature = 0.0
            ollama_num_predict = 800
        else:
            # Full prompt for LLaVA and other capable models
            ollama_temperature = 0.0
            ollama_num_predict = 1500
            prompt = """You are an expert occupational therapist assessing a home for fall hazards.

Analyze this image and identify ONLY the most significant fall hazards that are clearly visible.
Be conservative - only report hazards you can clearly see, not potential or assumed hazards.

For each hazard provide:
1. category: one of (bathroom, stairs, flooring, lighting, obstacles, furniture, kitchen, bedroom, external, general)
2. subcategory: specific hazard name (e.g., "loose rug", "missing grab bar")
3. severity: low, medium, high, or critical
4. description: brief description of the hazard
5. region: where in the image (e.g., "floor center", "left wall", "top right corner", "bottom of image")
6. confidence: 0.0 to 1.0 (how confident you are this is a real hazard)
7. recommendation: how to fix it

IMPORTANT RULES:
- Only report 3-6 hazards maximum
- Only report hazards with confidence >= 0.6
- Be specific about what you actually see
- Do NOT guess or assume hazards that aren't visible

Identify the room type first.

Respond ONLY with valid JSON (no other text):
{
    "room_type": "bathroom/kitchen/bedroom/living_room/stairs/hallway",
    "hazards": [
        {
            "category": "category",
            "subcategory": "specific hazard",
            "severity": "low/medium/high/critical",
            "description": "what you see",
            "region": "where in image",
            "confidence": 0.8,
            "recommendation": "how to fix"
        }
    ]
}

If no clear hazards are visible, return: {"room_type": "room_type", "hazards": []}"""

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
            return None

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
