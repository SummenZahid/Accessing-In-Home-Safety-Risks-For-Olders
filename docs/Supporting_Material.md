# Supporting Digital Material

## Assessing In-Home Safety Risks for Older Adults Using Generative Vision-Language Models

**Student:** Summen Zahid (B00996747)
**Supervisor:** Mark Donnelly
**Module:** COM742 MSc Research Project
**Institution:** Ulster University
**Date:** 2025/26

---

## Table of Contents

1. [Extended Literature Review](#1-extended-literature-review)
2. [Development Life Cycle and Tools](#2-development-life-cycle-and-tools)
3. [Professional, Ethical, Social and Sustainability Issues](#3-professional-ethical-social-and-sustainability-issues)
4. [Critical Appraisal and Lessons Learnt](#4-critical-appraisal-and-lessons-learnt)

---

## 1. Extended Literature Review

### 1.1 The Global Burden of Falls

Falls represent one of the most significant public health challenges facing ageing populations. The WHO estimates 684,000 fatal falls annually, making falls the second leading cause of unintentional injury death [WHO, 2007]. Approximately 95% of hip fractures result from falls with only 25% making full recovery [Stevens & Rudd, 2013]. Fear of falling leads to activity restriction, social isolation, and accelerated functional decline [Scheffer et al., 2008].

### 1.2 Environmental Risk Factors

Rubenstein (2006) categorised fall risk into intrinsic (person-related) and extrinsic (environment-related) factors. Environmental hazards are often more readily modifiable than intrinsic factors. Key hazards with reported odds ratios include: loose rugs and mats (OR 2.1-3.4) [Speechley & Tinetti, 1991], absence of grab bars (OR 2.8-4.2) [Sattin et al., 1998], missing handrails (OR 3.1-4.8) [Nevitt et al., 1989], and inadequate lighting (OR 1.5-2.0) [Gill et al., 1999]. These evidence-based risk ratios directly informed the clinical weighting scheme implemented in the system.

### 1.3 Assessment Instruments and Their Limitations

The Westmead Home Safety Assessment (WeHSA) by Clemson (1997) is the gold standard, with 72 items across 11 categories, inter-rater reliability ICC = 0.83-0.96, and demonstrated predictive validity for fall outcomes. HOME FAST provides 25-item rapid screening in 15-20 minutes but sacrifices detail. SAFER-HOME integrates functional and environmental assessment but requires even more time. All require professional administration, creating a fundamental scalability barrier.

### 1.4 Vision-Language Models in Healthcare

VLMs represent a paradigm shift from task-specific models to general-purpose visual reasoning. GPT-4V demonstrates human-level performance on visual question-answering benchmarks [OpenAI, 2023]. Gemini 2.0 Flash offers strong multimodal understanding with efficient inference. Open-source alternatives like LLaVA (7B parameters) [Liu et al., 2023] enable local deployment, while Moondream (~1.6B) targets edge computing scenarios. Healthcare applications have shown promise: Med-PaLM 2 achieved 86.5% on medical questions [Singhal et al., 2023], and GPT-4V demonstrated diagnostic capabilities comparable to dermatology specialists [Zakka et al., 2024]. However, environmental safety assessment using VLMs remains unexplored, representing the research gap this project addresses.

---

## 2. Development Life Cycle and Tools

### 2.1 Research Methodology

This project follows the Design Science Research Methodology (DSRM) [Peffers et al., 2007], appropriate for creating and evaluating IT artefacts:

**Phase 1 - Problem Identification**: Literature review on fall risk assessment instruments; analysis of scalability limitations in existing clinical tools; identification of the VLM opportunity.

**Phase 2 - Objectives Definition**: Automated hazard detection from photographs; clinical alignment with Westmead Assessment taxonomy; explainable, quantified risk outputs on a 0-100 scale.

**Phase 3 - Design and Development**: Modular pipeline architecture with five components (preprocessing, model interface, detection engine, risk scoring, reporting); factory pattern enabling multi-model support; YAML-based prompt configuration for iterative refinement.

**Phase 4 - Demonstration**: Streamlit web application for interactive assessment; model comparison interface (LLaVA vs Moondream side-by-side); batch evaluation pipeline.

**Phase 5 - Evaluation**: Unit testing with 70+ tests across all modules; functional evaluation using the PHELE dataset (575 images); multi-model comparison across four VLM backends.

**Phase 6 - Communication**: Research paper, supporting documentation, and oral presentation.

### 2.2 Development Tools

| Category | Tool | Purpose |
|----------|------|---------|
| Language | Python 3.12 | Primary development language |
| IDE | Visual Studio Code | Code editing and debugging |
| Version Control | Git | Source code management |
| VLM APIs | OpenAI API, Google GenAI | Cloud vision model integration |
| Local Models | Ollama | LLaVA and Moondream inference |
| Data Validation | Pydantic v2 | Schema validation for VLM outputs |
| Image Processing | OpenCV, Pillow | Quality assessment and preprocessing |
| Web UI | Streamlit | Interactive application |
| Testing | Pytest | Unit and integration testing |
| Configuration | PyYAML | Prompt template management |
| Metrics | Scikit-learn | Evaluation metrics computation |

### 2.3 Sprint-Based Development

**Sprint 1 (Weeks 1-2)**: Project setup, literature review, base architecture design with abstract model interface.

**Sprint 2 (Weeks 3-4)**: Vision model abstraction layer (BaseVisionModel, factory pattern), hazard category definitions (42 subcategories with clinical metadata).

**Sprint 3 (Weeks 5-6)**: Clinical weight implementation from Westmead evidence, risk scoring algorithm with diminishing returns, Pydantic result schemas.

**Sprint 4 (Weeks 7-8)**: Multi-pass detection strategy, chain-of-thought prompt engineering, YAML prompt configuration, Streamlit web application.

**Sprint 5 (Weeks 9-10)**: Evaluation pipeline, unit tests (70+ passing), model comparison framework, documentation and paper writing.

### 2.4 Key Design Decisions

**Factory Pattern for Model Abstraction**: Enables seamless switching between GPT-4V, Gemini, LLaVA, and Moondream without modifying detection or scoring logic. New models require only implementing the BaseVisionModel interface.

**Pydantic Schemas for Output Validation**: VLM outputs are inherently variable. Pydantic enforces structured, type-validated outputs (DetectedHazard, HazardDetectionResult) ensuring downstream components receive consistent data regardless of model backend.

**YAML-based Prompt Configuration**: Separating prompts from code enables rapid A/B testing and iterative refinement. The 341-line prompt configuration includes role specification, room-specific prompts, few-shot examples, and confidence calibration guidelines.

**Multi-Pass Detection**: Single-pass analysis misses subtle hazards, particularly in complex scenes. The three-pass approach (global scan, category-specific focus, chain-of-thought reasoning) achieves approximately 2x improvement in hazard count and category coverage.

---

## 3. Professional, Ethical, Social and Sustainability Issues

### 3.1 Professional Issues

**Code Quality**: The codebase adheres to PEP 8 style guidelines with comprehensive docstrings, type hints throughout, and 70+ unit tests achieving >80% code coverage. A modular architecture with clear separation of concerns supports maintainability.

**Data Protection**: The system is designed with GDPR principles. No persistent storage of user images occurs on servers. API calls use encrypted HTTPS connections. The Ollama-based local model option enables fully offline, privacy-preserving deployment. Clear data handling documentation accompanies the system.

**Professional Responsibility**: The system includes explicit disclaimers positioning it as a screening tool that augments but does not replace professional assessment. Recommendations are framed as suggestions, not medical or clinical advice. System limitations are documented prominently.

### 3.2 Ethical Issues

**Privacy**: Home photographs contain sensitive information. Mitigations include local processing options (Ollama/LLaVA), minimal data retention policies, and user consent workflow recommendations.

**Algorithmic Bias**: VLMs are trained predominantly on Western home environments, potentially underperforming in diverse cultural contexts. Mitigation involves diverse test image collection and cultural adaptation guidelines for deployment.

**Informed Consent and Liability**: The system provides screening, not diagnosis; professional follow-up is recommended for high-risk scores.

### 3.3 Social Issues

**Digital Divide**: Older adults face technology barriers. The system accommodates this through family/caregiver involvement workflows, simple Streamlit interface design, and potential deployment through healthcare providers.

**Healthcare Equity**: The system could extend assessment access to rural and underserved areas, but technology access disparities could exacerbate health inequalities. Positioning as a complement to existing care pathways is essential.

**Impact on Healthcare Workers**: The system augments occupational therapists, enabling focus on complex cases while automated screening handles initial assessment.

### 3.4 Sustainability Issues

**Environmental**: Mitigations include efficient prompt design minimising API calls and local model alternatives (LLaVA via Ollama) that reduce network energy costs.

**Economic**: API costs (~$0.01-0.05 per image) are sustainable for screening programmes. Open-source models provide long-term cost sustainability without API dependencies.

**Social**: The open-source codebase and documentation support knowledge transfer and local capacity building.

---

## 4. Critical Appraisal and Lessons Learnt

### 4.1 Project Achievements

The project successfully implemented: (1) a comprehensive hazard taxonomy of 42 clinically-validated types aligned with the Westmead Assessment, (2) working VLM integration with four backends (GPT-4V, Gemini, LLaVA, Moondream), (3) a clinically-weighted risk scoring algorithm with diminishing returns, (4) a multi-pass detection strategy improving coverage by approximately 2x, (5) a robust codebase with 70+ passing unit tests, and (6) a functional Streamlit web application with model comparison capabilities.

Technical innovations include the factory pattern enabling model interchangeability, structured output validation via Pydantic schemas, a region-based visualisation system (9-grid) replacing unreliable VLM-generated bounding boxes, and a text-based fallback hazard extraction mechanism for models that cannot produce valid JSON.

### 4.2 Challenges and Resolutions

**Challenge 1 — Risk Scores Stuck at Moderate**: Initial testing revealed that regardless of image severity, risk scores remained in the 25-50 (MODERATE) range. A severely cluttered, damaged room scored only 29/100 on LLaVA and 6/100 on Moondream (Figure 1). Root cause analysis identified three compounding issues: (a) the prompt instructed the model to "Be conservative" and report "ONLY 3-6 hazards maximum" with confidence >= 0.6, (b) the scoring formula used a per-hazard multiplier of only 20, meaning 3 medium hazards scored just 25.5/100, and (c) Moondream returned placeholder text instead of analysing the image.

*Resolution*: The scoring algorithm was redesigned with higher severity weights (medium: 0.50→0.65, high: 0.75→0.85), an increased base multiplier (20→30), a cumulative risk multiplier for multiple hazards (3+ hazards: +15%, 6+: +30%, 9+: +45%), and high-severity bonuses. With this formula, the same 3 medium hazards now score 60.5/100 (HIGH), accurately reflecting the danger. The prompt was rewritten to encourage thorough detection of up to 10 hazards.

![Figure 1: Before fix — severely cluttered room scored only 29/100 (LLaVA) and 6/100 (Moondream)](screenshots/before_fix_low_scores.png)

**Challenge 2 — Scope Limited to Falls Only**: Testing with a kitchen fire image revealed that neither model detected the fire — LLaVA returned 0 hazards, Moondream returned 2 hallucinated hazards unrelated to fire. The system was designed exclusively for fall hazards, missing critical dangers like fire, electrical, and structural hazards that pose equal or greater risk to elderly residents living alone.

*Resolution*: The hazard taxonomy was expanded from fall-specific to comprehensive home safety. New categories added: fire/burn hazards, electrical hazards, and structural hazards. Prompts were rewritten to cover fire, smoke, exposed wiring, structural damage, blocked exits, and chemical/toxic hazards. Category weights were updated (fire: 1.0, electrical: 1.0, structural: 0.95). The text fallback extractor was extended with fire and structural keywords.

**Challenge 3 — Model Hallucination on Clean Rooms**: A clean, modern bathroom with a glass shower scored 100/100 on both models. LLaVA reported "stair (critical)", "tub (critical)", "shower (critical)" — treating normal fixtures as hazards. Moondream copied the JSON template placeholder text verbatim instead of analysing the image (Figure 2).

*Resolution*: Three targeted fixes were applied. First, the LLaVA prompt was augmented with explicit anti-hallucination instructions: "A shower, bathtub, toilet, sink is NOT a hazard by itself. Only flag if BROKEN, DAMAGED, or MISSING a safety feature." Second, Moondream's prompt was simplified to plain text (removing all JSON templates that it was copying), relying on a keyword-based text fallback extractor. Third, the text fallback was redesigned with a two-tier keyword system: direct hazard words (fire, clutter, slip) always trigger, while context-dependent words (floor, bath, stair) only trigger when paired with danger modifiers (broken, missing, wet, loose). A negation detector was added to prevent sentences like "no visible clutter" from triggering false positives.

![Figure 2: After fix — clean modern living room correctly scores 0/100 with no hazards detected](screenshots/after_fix_clean_room.png)

**Challenge 4 — Moondream Analysis Failures**: Moondream (~1.6B parameters) could not produce valid JSON, causing "Analysis failed" errors. When given a JSON template, it copied the template verbatim. When given a plain-text prompt, the response had no JSON markers, causing the parser to return None.

*Resolution*: A dedicated plain-text processing path was implemented for Moondream. When no JSON is found in the response (no `{` character), the system falls through to the text fallback extractor instead of returning None. This extractor scans for hazard-related keywords with context awareness, assigns appropriate severity levels, and produces structured hazard data from natural language descriptions.

**Ground Truth Scarcity**: No public dataset of home images with labelled fall hazards exists. The PHELE dataset provides real domestic images but lacks comprehensive hazard annotations. A curated test collection of 56 images (31 high-risk, 25 low-risk) was assembled from royalty-free sources to enable systematic evaluation across risk levels.

**VLM Output Variability**: Same image with identical prompt produced varying detection counts across runs. Resolution included temperature set to 0.0 (LLaVA) and 0.1 (Moondream), fixed seed (42), confidence thresholding, and deduplication. Lesson: VLM stochasticity requires architectural design accommodations.

### 4.3 Limitations Acknowledged

The primary limitations are: (1) absence of large-scale clinical validation against expert assessments, (2) reliance on single-image analysis rather than comprehensive multi-room assessment, (3) commercial API cost and availability dependencies for highest-quality detection, (4) Western-centric hazard definitions requiring cultural adaptation, and (5) image quality sensitivity that preprocessing can only partially mitigate.

### 4.4 Lessons Learnt

**Technical**: Start with simple prompts and add complexity incrementally. Design for VLM output variability from the beginning. Modular architecture enables rapid iteration across model backends.

**Project Management**: Scope management is critical for time-limited projects. Early supervisor engagement prevents rework. Buffer time for unexpected challenges (e.g., model output debugging) is essential.

**Research**: Ethical considerations are integral to healthcare AI development. Evaluation methodology requires early planning, particularly regarding ground truth data availability.

### 4.5 Personal Reflection

This project provided valuable experience in applying cutting-edge AI technology to a real-world healthcare problem, navigating ethical complexities of AI in vulnerable populations, balancing technical ambition with practical constraints, and communicating research across disciplinary boundaries. The work has reinforced the importance of clinical grounding in healthcare AI and the potential for VLMs to democratise access to safety assessment services.

---

## Appendices

### Appendix A: Hazard Category Definitions

Complete definitions of all 42 hazard subcategories with clinical weights, Westmead references, examples, and detection keywords are implemented in `src/hazard_detection/categories.py` (894 lines).

### Appendix B: Prompt Templates

Full prompt configurations including system prompts, room-specific analysis prompts, chain-of-thought instructions, and output format specifications are in `configs/prompts/hazard_detection.yaml` (341 lines).

### Appendix C: Code Repository

The complete source code includes: implementation source (`src/`), unit tests (`tests/`), evaluation pipeline, Streamlit web application (`app.py`), Jupyter notebook demonstrations (`notebooks/`), and configuration files (`configs/`).

### Appendix D: Sample Detection Output

```json
{
  "hazards": [
    {
      "category": "bathroom",
      "subcategory": "bath_no_grab_bars",
      "severity": "critical",
      "confidence": 0.92,
      "description": "No grab bars visible near bathtub area",
      "location": "Bathtub wall",
      "recommendation": "Install grab bars on wall adjacent to bathtub"
    },
    {
      "category": "flooring",
      "subcategory": "loose_rug",
      "severity": "high",
      "confidence": 0.85,
      "description": "Small rug without non-slip backing at bathroom entrance",
      "location": "Bathroom doorway",
      "recommendation": "Remove rug or secure with non-slip backing"
    }
  ],
  "risk_score": 68,
  "risk_level": "HIGH",
  "room_type": "bathroom"
}
```

### Appendix E: Test Results

```
======================== test session starts ========================
platform darwin -- Python 3.12.8, pytest-8.0.0
collected 100 items

tests/test_detection.py ........................ [23%]
tests/test_metrics.py .................. [41%]
tests/test_models.py .................... [61%]
tests/test_preprocessing.py ............ [73%]
tests/test_scoring.py .................. [100%]

==================== 70 passed, 30 skipped ======================
```

---

### Appendix F: Demo Video

A demonstration video showing the complete system workflow is available at:

[Demo Video Link — to be added]

The video covers: uploading room images, running LLaVA and Moondream analysis, interpreting risk scores and hazard detections, model comparison mode, and testing across high-risk and low-risk environments.

---

*Supporting material submitted in partial fulfilment of MSc Research Project (COM742), Ulster University, 2025/26.*

**Word Count:** ~2,400 words (excluding appendices)
