# Assessing In-Home Safety Risks for Older Adults Using Generative AI

An AI-powered system that detects, quantifies, and explains fall hazards in home environments for elderly individuals using Vision-Language Models (VLMs). Built as part of a Masters dissertation at Ulster University.

## Overview

Falls are the leading cause of injury among older adults — one-third of people aged 65+ fall annually, with 50–60% of falls occurring at home due to environmental hazards. Traditional home safety assessments require trained professionals and take 45–60 minutes per home.

This system automates that process using vision-language models to analyze room photographs and:

- **Detect hazards** across 42 clinically-validated subcategories aligned with the Westmead Home Safety Assessment
- **Score risk** on a 0–100 scale using evidence-based clinical weights
- **Generate recommendations** with prioritized, actionable steps
- **Visualize hazard regions** with color-coded severity overlays

## Features

- **Streamlit web application** for interactive analysis with image upload
- **Multi-model support** — Ollama (LLaVA, Moondream), OpenAI GPT-4 Vision, Google Gemini 2.0 Flash
- **Local-first inference** — runs privately on your machine with free open-source models via Ollama
- **Side-by-side model comparison** (LLaVA vs Moondream)
- **Multi-pass detection** with chain-of-thought prompting and deduplication
- **Image preprocessing** — auto-enhancement (CLAHE, brightness, blur detection)
- **Region-based visualization** — 9-region grid with severity-coded overlays
- **Structured outputs** with Pydantic validation and JSON export
- **Comprehensive evaluation framework** — precision, recall, F1, Cohen's Kappa
- **Batch processing** for dataset-level analysis

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running

### Setup

```bash
# 1. Pull a vision model
ollama pull llava:7b       # ~4GB, recommended
# or
ollama pull moondream      # ~1.6GB, faster but less accurate

# 2. Install dependencies
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the web app
streamlit run app.py
# Opens at http://localhost:8501
```

Upload a room image, click **Analyze for Hazards**, and view the results.

### Optional: Cloud API Models

To use GPT-4 Vision or Gemini, create a `.env` file:

```
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

## Project Structure

```
├── app.py                           # Streamlit web application
├── requirements.txt                 # Python dependencies
├── configs/
│   └── prompts/
│       └── hazard_detection.yaml    # VLM prompt templates
├── src/
│   ├── models/                      # Vision model integrations
│   │   ├── base_model.py            # Abstract interface & Pydantic schemas
│   │   ├── gpt4v_model.py           # OpenAI GPT-4 Vision
│   │   ├── gemini_model.py          # Google Gemini
│   │   └── model_factory.py         # Factory pattern for model selection
│   ├── hazard_detection/            # Core detection logic
│   │   ├── detector.py              # Multi-pass detection orchestrator
│   │   └── categories.py            # Westmead-aligned hazard taxonomy (42 types)
│   ├── risk_scoring/                # Risk quantification
│   │   ├── scorer.py                # Weighted scoring algorithm
│   │   └── weights.py               # Clinical evidence-based weights
│   ├── preprocessing/               # Image quality & enhancement
│   │   ├── image_processor.py       # CLAHE, brightness, blur detection
│   │   └── data_loader.py           # Dataset loading utilities
│   └── evaluation/                  # Evaluation framework
│       ├── run_evaluation.py        # Systematic evaluation pipeline
│       └── metrics.py               # Precision, recall, F1, Cohen's Kappa
├── scripts/                         # Utility scripts
│   ├── classify_images.py           # Batch image classification
│   ├── prepare_training_data.py     # Dataset preparation
│   ├── convert_phele_annotations.py # Annotation format conversion
│   └── download_curated_images.py   # Dataset downloads
├── notebooks/
│   ├── 01_getting_started.ipynb     # Interactive tutorial
│   └── finetune_llava_colab.ipynb   # Fine-tuning guide (Google Colab)
├── data/
│   ├── annotations/                 # Ground truth annotations (JSON, tracked)
│   ├── curated/                     # Curated high/low risk images (not tracked)
│   ├── processed/                   # Pre-processed classified images (not tracked)
│   ├── raw/phele/                   # PHELE dataset, 575 images (not tracked)
│   └── sample/                      # Custom test images
├── tests/                           # pytest test suite
├── docs/                            # Research paper & meeting notes
├── results/evaluation/              # Model evaluation outputs (JSON)
└── reports/                         # Generated reports & figures
```

## Hazard Categories

Based on the Westmead Home Safety Assessment — 42 subcategories across 10 main categories:

| Category | Examples | Clinical Weight |
|----------|----------|-----------------|
| Stairs | Missing handrails, slippery steps, poor lighting | 1.00 |
| Bathroom | No grab bars, slippery surfaces, toilet transfers | 0.95 |
| Flooring | Loose rugs, uneven surfaces, high thresholds | 0.85 |
| Obstacles | Cords, clutter, blocked pathways | 0.85 |
| External | Pathways, ramps, gates | 0.80 |
| Lighting | Dim areas, shadows, no night lights | 0.75 |
| Bedroom | Bed height, path to bathroom | 0.75 |
| Furniture | Unstable items, wrong height, sharp edges | 0.70 |
| Kitchen | High storage, spill hazards | 0.70 |
| General | Other environmental hazards | 0.60 |

## Risk Scoring

```
Risk Score = Σ (CategoryWeight × SeverityMultiplier × Confidence × HazardWeight)

Severity Multipliers:
  Low: 0.25 | Medium: 0.50 | High: 0.75 | Critical: 1.00

Risk Levels (0–100):
  0–25:   LOW       — Minor concerns
  26–50:  MODERATE  — Address within 2 weeks
  51–75:  HIGH      — Address within 48 hours
  76–100: CRITICAL  — Immediate action required
```

## Usage

### Python API

```python
from src.models.model_factory import VisionModelFactory
from src.models.base_model import ImageInput
from src.hazard_detection.detector import HazardDetector
from src.risk_scoring.scorer import RiskScorer

model = VisionModelFactory.create_auto({})
detector = HazardDetector(model)
scorer = RiskScorer()

image = ImageInput(path="path/to/room.jpg")
result = detector.detect_hazards(image)
risk = scorer.calculate_score(result)

print(f"Risk Score: {risk.total_score}/100 ({risk.risk_level.value})")
print(f"Hazards Found: {len(result.hazards)}")
```

### Batch Processing

```bash
python scripts/classify_images.py --source data/raw/phele/test --max-images 100
```

### Evaluation

```bash
python -m src.evaluation.run_evaluation --model llava --split test
```

### Tests

```bash
pytest tests/
pytest tests/ --cov=src --cov-report=html
```

## Models Supported

| Model | Type | Size | Speed | Cost |
|-------|------|------|-------|------|
| LLaVA 7B | Local (Ollama) | ~4 GB | ~30–60s | Free |
| Moondream | Local (Ollama) | ~1.6 GB | ~20s | Free |
| GPT-4 Vision | Cloud (OpenAI) | — | ~5–10s | Paid |
| Gemini 2.0 Flash | Cloud (Google) | — | ~5–10s | Paid |

## Tech Stack

**Core:** Python, Streamlit, Pydantic v2, YAML
**Vision Models:** Ollama, OpenAI API, Google Gemini API
**Image Processing:** OpenCV, Pillow, NumPy
**Evaluation:** scikit-learn, Pandas, Matplotlib, Seaborn
**Testing:** pytest

## References

1. Clemson, L. (1997, 2015). Westmead Home Safety Assessment
2. CDC STEADI Framework for Fall Prevention
3. WHO Global Report on Falls Prevention in Older Age

## Author

**Summen Zahid** (B00996747)
Supervisor: Mark Donnelly
Ulster University — COM748 Masters Research Project

---

*Last updated: March 2026*
