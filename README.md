# Fall Risk Detection System

**Assessing In-Home Safety Risks for Older Adults Using Generative Models**

A Generative AI-based system that identifies, quantifies, and explains fall risks in domestic environments for elderly individuals living alone.

## Project Overview

Falls are one of the major contributors to injuries, hospitalizations, and loss of independence for older people living in their own homes. This system uses vision-language models to analyze home environment images and detect potential fall hazards, providing:

- **Hazard Detection**: Identifies environmental hazards based on clinical guidelines
- **Risk Quantification**: Calculates weighted risk scores (0-100 scale)
- **Explainable Outputs**: Provides human-readable explanations and recommendations
- **Clinical Alignment**: Based on the Westmead Home Safety Assessment framework

## Features

- Multi-model support (OpenAI GPT-4 Vision, Google Gemini)
- Multi-pass detection strategy for improved accuracy
- Comprehensive hazard categories aligned with Westmead Assessment
- Clinically-weighted risk scoring algorithm
- Structured JSON output with Pydantic validation
- Interactive Jupyter notebooks for analysis

## Installation

### Prerequisites

- Python 3.10 or higher
- An API key for either:
  - OpenAI (for GPT-4 Vision)
  - Google Cloud (for Gemini)

### Setup

1. Clone the repository:
```bash
cd /path/to/Dissertation/code
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create environment file:
```bash
cp .env.example .env
```

5. Add your API key to `.env`:
```
# For OpenAI
OPENAI_API_KEY=your_openai_key_here

# OR for Google Gemini
GOOGLE_API_KEY=your_google_key_here
```

## Quick Start

### Using Python

```python
from src.models.model_factory import VisionModelFactory
from src.models.base_model import ImageInput
from src.hazard_detection.detector import HazardDetector
from src.risk_scoring.scorer import RiskScorer

# Initialize model (auto-detects based on available API keys)
model = VisionModelFactory.create_auto({})

# Initialize detector and scorer
detector = HazardDetector(model)
scorer = RiskScorer()

# Analyze an image
image = ImageInput(path="path/to/home_image.jpg")
detection_result = detector.detect_hazards(image)
risk_result = scorer.calculate_score(detection_result)

# View results
print(f"Risk Score: {risk_result.total_score}/100")
print(f"Risk Level: {risk_result.risk_level.value}")
print(f"Hazards Found: {len(detection_result.hazards)}")
```

### Using Jupyter Notebooks

Open and run `notebooks/01_getting_started.ipynb` for an interactive walkthrough.

## Project Structure

```
fall-risk-detection/
├── configs/
│   └── prompts/              # Prompt templates (YAML)
├── src/
│   ├── models/               # Vision model integrations
│   │   ├── base_model.py     # Abstract interface & schemas
│   │   ├── gpt4v_model.py    # OpenAI GPT-4 Vision
│   │   ├── gemini_model.py   # Google Gemini
│   │   └── model_factory.py  # Factory pattern
│   ├── hazard_detection/     # Hazard detection module
│   │   ├── categories.py     # Westmead hazard definitions
│   │   └── detector.py       # Detection orchestrator
│   ├── risk_scoring/         # Risk scoring module
│   │   ├── weights.py        # Clinical weights
│   │   └── scorer.py         # Scoring algorithm
│   ├── explainability/       # Explanation generation
│   ├── preprocessing/        # Image preprocessing
│   └── evaluation/           # Evaluation metrics
├── notebooks/                # Jupyter notebooks
├── data/
│   ├── sample/               # Sample test images
│   └── annotations/          # Ground truth (JSON)
├── tests/                    # Unit tests
└── reports/                  # Generated reports
```

## Hazard Categories

Based on the Westmead Home Safety Assessment:

| Category | Examples | Clinical Weight |
|----------|----------|-----------------|
| Stairs | Missing handrails, slippery surfaces | 1.00 (highest) |
| Bathroom | No grab bars, slippery floors | 0.95 |
| Flooring | Loose rugs, uneven surfaces | 0.85 |
| Obstacles | Cords, clutter, blocked paths | 0.85 |
| Lighting | Dim areas, no night lights | 0.75 |
| Furniture | Unstable, wrong height | 0.70 |
| Bedroom | Bed height, path to bathroom | 0.75 |
| Kitchen | High storage, spill areas | 0.70 |

## Risk Scoring

```
Risk Score = Σ (CategoryWeight × SeverityMultiplier × Confidence × HazardWeight)

Severity Multipliers:
- Low: 0.25
- Medium: 0.50
- High: 0.75
- Critical: 1.00

Risk Levels:
- 0-25:  LOW       (green)  - Minor concerns
- 26-50: MODERATE  (yellow) - Address within 2 weeks
- 51-75: HIGH      (orange) - Address within 48 hours
- 76-100: CRITICAL (red)    - Immediate action required
```

## API Reference

### HazardDetector

```python
detector = HazardDetector(vision_model, prompts_dir="configs/prompts")

# Full detection with multi-pass
result = detector.detect_hazards(image, config=DetectionConfig())

# Quick detection (single pass)
result = detector.detect_quick(image, room_type="bathroom")
```

### RiskScorer

```python
scorer = RiskScorer()

# Calculate comprehensive score
risk_result = scorer.calculate_score(detection_result)

# Quick score without breakdown
quick_score = scorer.calculate_quick_score(hazards)
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## License

This project is developed as part of a Masters dissertation at Ulster University.

## References

1. Clemson, L. (1997, 2015). Westmead Home Safety Assessment
2. CDC STEADI Framework for Fall Prevention
3. WHO Global Report on Falls Prevention in Older Age

## Author

**Summen Zahid** (B00996747)
Supervisor: Mark Donnelly
Ulster University - COM748 Masters Research Project
