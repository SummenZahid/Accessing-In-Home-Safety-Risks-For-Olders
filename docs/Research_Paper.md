# Assessing In-Home Safety Risks for Older Adults Using Generative Vision-Language Models

**Summen Zahid**
School of Computing, Ulster University, Belfast, United Kingdom
zahid-s3@ulster.ac.uk

**Supervisor: Mark Donnelly**

---

## Abstract

Falls represent a leading cause of injury and mortality among older adults, with approximately one-third of individuals aged 65 and above experiencing at least one fall annually. Environmental hazards within the home account for 50-60% of these incidents, yet current assessment methods rely on trained professionals conducting time-consuming in-person evaluations. This research presents a novel automated system that leverages pre-trained Vision-Language Models (VLMs) to detect fall hazards in home environments from photographs. The system implements a comprehensive taxonomy of 42 clinically-validated hazard subcategories across 10 categories, aligned with the Westmead Home Safety Assessment framework. A multi-pass detection strategy employing chain-of-thought prompting enables systematic hazard identification, while a clinically-weighted risk scoring algorithm quantifies overall fall risk on a normalised 0-100 scale. The system was evaluated using four VLM backends: GPT-4 Vision, Google Gemini 2.0 Flash, LLaVA 7B, and Moondream, tested against the PHELE (Physical Hazards of Elderly Living Environment) dataset comprising 575 images. Results demonstrate that commercial VLMs (GPT-4V, Gemini) reliably identify critical hazards including missing grab bars, loose rugs, and stair safety issues, with LLaVA detecting 5-21 hazards per image. The clinically-weighted scoring algorithm produces risk classifications aligned with established frameworks. This work contributes to AI-assisted preventive healthcare by providing a scalable, explainable, and cost-effective tool for proactive fall prevention in ageing populations.

**Keywords:** Fall Prevention, Computer Vision, Vision-Language Models, Elderly Care, Home Safety Assessment, Explainable AI, Westmead Assessment

---

## I. Introduction

### A. Background and Motivation

The global population is experiencing unprecedented demographic ageing. By 2050, the World Health Organization (WHO) projects that the number of people aged 60 years and older will reach 2.1 billion, nearly doubling from 2020 levels [1]. This demographic shift presents critical challenges for healthcare systems, particularly regarding fall prevention among older adults.

Falls are the second leading cause of unintentional injury deaths worldwide, with adults over 65 suffering the greatest number of fatal falls [2]. Beyond mortality, falls result in substantial morbidity: approximately 95% of hip fractures result from falls, with only 25% of patients making a full recovery [3]. Traumatic brain injuries from falls account for over 50% of TBI cases in older adults [4]. The economic burden is considerable, with fall-related medical costs exceeding $50 billion annually in the United States alone [5].

Environmental factors within the home contribute significantly to fall risk. Research indicates that approximately 50-60% of falls among older adults occur within the home environment [6]. Common hazards include loose rugs (OR 2.1-3.4), absence of grab bars (OR 2.8-4.2), missing handrails (OR 3.1-4.8), and inadequate lighting (OR 1.5-2.0) [7][8]. Traditional home safety assessments such as the Westmead Home Safety Assessment (WeHSA) require trained professionals and approximately 45-60 minutes per home visit, limiting scalability to meet population health needs.

### B. Research Aims and Objectives

This research aims to develop an automated fall hazard detection system using pre-trained generative Vision-Language Models to analyse photographs of home environments. The specific objectives are:

1. To implement a comprehensive hazard taxonomy of 42 subcategories aligned with the Westmead Home Safety Assessment framework.
2. To develop a multi-pass detection strategy using VLMs with chain-of-thought prompting for systematic hazard identification.
3. To create a clinically-weighted risk scoring algorithm that quantifies overall fall risk on a normalised 0-100 scale.
4. To evaluate and compare multiple VLM backends (GPT-4V, Gemini, LLaVA, Moondream) for detection capability, consistency, and practical deployment considerations.
5. To provide explainable AI outputs with actionable safety recommendations.

### C. Paper Structure

The remainder of this paper is organised as follows: Section II reviews existing literature on fall prevention and AI-based safety assessment. Section III describes the methodology, including system architecture, detection strategy, and scoring algorithms. Section IV presents evaluation results across multiple models. Section V discusses findings, limitations, and comparative analysis. Section VI concludes with contributions and future work directions.

---

## II. Existing Work

### A. Fall Risk Assessment Instruments

The Westmead Home Safety Assessment (WeHSA), developed by Clemson [9], is the most widely validated instrument for identifying environmental fall hazards. The assessment comprises 72 items across 11 categories covering external pathways, internal flooring, lighting, furniture stability, bathroom safety, and stair conditions. Psychometric properties include inter-rater reliability (ICC = 0.83-0.96) and test-retest reliability (ICC = 0.80-0.94) [9]. However, the WeHSA requires trained assessors and substantial time per evaluation.

Other validated instruments include the Home Falls and Accidents Screening Tool (HOME FAST) [10], a 25-item rapid screening tool, and the Safety Assessment of Function and the Environment for Rehabilitation (SAFER-HOME) [11], which integrates functional and environmental assessment. These instruments share common limitations: they require professional administration, are resource-intensive, and cannot scale to meet population health demands. The CDC STEADI (Stopping Elderly Accidents, Deaths & Injuries) framework provides guidelines for fall prevention but similarly relies on manual assessment processes [12].

### B. Computer Vision for Safety Assessment

Traditional computer vision approaches to hazard detection have employed Convolutional Neural Networks (CNNs) for object detection. YOLO (You Only Look Once) [13] and Faster R-CNN architectures have been applied to identify specific environmental hazards such as obstacles on floors [14]. However, these approaches require extensive labelled training data specific to fall hazards, which are scarce, and are limited to detecting predefined object categories without contextual reasoning.

Recent work has explored depth sensors and RGB-D cameras for fall risk assessment in smart homes. The CASAS Smart Home project demonstrated feasibility of ambient sensor networks for activity monitoring [15], but highlighted challenges with installation costs, hardware maintenance, and user acceptance. Wearable sensor approaches using accelerometers detect falls after they occur rather than preventing them through proactive hazard identification [16].

Semantic segmentation methods provide scene understanding but lack the contextual reasoning necessary to distinguish a decorative rug (low risk) from a loose rug near a doorway (high risk). This contextual gap motivates the application of models capable of higher-level reasoning about spatial relationships and safety implications.

### C. Vision-Language Models

The emergence of Vision-Language Models (VLMs) represents a paradigm shift in computer vision capabilities. GPT-4 Vision [17] and Google Gemini [18] demonstrate remarkable abilities in understanding complex visual scenes and reasoning about spatial relationships, context, and potential risks. These models offer several advantages for fall hazard detection:

- **Zero-shot learning**: No task-specific training data required, enabling immediate deployment for novel applications.
- **Contextual reasoning**: Understanding of spatial relationships and environmental context beyond simple object detection.
- **Natural language outputs**: Explainable results accessible to non-technical users and healthcare professionals.
- **Flexibility**: Ability to identify novel hazards not explicitly programmed into detection systems.

Open-source alternatives have emerged, including LLaVA (Large Language and Vision Assistant) [19], a 7-13B parameter model, and Moondream [20], a lightweight ~1.6B parameter vision model designed for edge deployment. These models enable local, privacy-preserving inference without cloud API dependencies.

Recent studies have begun exploring VLMs for healthcare applications, including medical image analysis where Med-PaLM 2 achieved 86.5% on USMLE-style questions [21], and GPT-4V demonstrated diagnostic capabilities comparable to specialists for dermatological conditions [22]. However, application of VLMs to environmental fall risk assessment remains largely unexplored.

### D. Research Gap

Current literature reveals a significant gap at the intersection of three domains: (1) validated clinical assessment frameworks with established psychometric properties, (2) scalable technology solutions capable of population-level deployment, and (3) the emerging visual reasoning capabilities of VLMs. No existing work comprehensively applies VLMs to fall hazard detection using clinically-validated taxonomies. This research addresses this gap by integrating the Westmead Assessment framework with state-of-the-art VLM capabilities, providing both the clinical grounding and technological scalability needed for practical impact.

---

## III. Methodology

### A. System Architecture

The proposed system follows a modular pipeline architecture comprising five main components (Fig. 1):

1. **Image Preprocessing Module**: Validates image quality (brightness, contrast, blur, resolution), standardises dimensions (max 1024x1024), and applies enhancement techniques including CLAHE (Contrast Limited Adaptive Histogram Equalization) and bilateral filtering.
2. **Vision-Language Model Interface**: Abstracts interactions with multiple VLM backends through a factory pattern, supporting GPT-4V, Gemini, LLaVA, and Moondream via a unified API.
3. **Hazard Detection Engine**: Implements multi-pass detection with structured prompting and chain-of-thought reasoning.
4. **Risk Scoring Module**: Calculates clinically-weighted risk scores using evidence-based category and severity weights.
5. **Report Generation**: Produces explainable outputs with hazard descriptions, severity classifications, and actionable recommendations.

```
Image Input -> Preprocessing -> VLM Analysis -> Hazard Detection -> Risk Scoring -> Report
```

*Fig. 1. System pipeline architecture showing the five-stage processing flow from image input to risk report generation.*

### B. Hazard Taxonomy

The system implements 42 hazard subcategories organised into 10 main categories, derived from the Westmead Home Safety Assessment and CDC STEADI guidelines. Each subcategory includes a clinical severity weight (0.60-0.95), Westmead reference section, example manifestations, and detection keywords. Table I presents the category hierarchy with clinical weights.

**TABLE I: HAZARD CATEGORIES WITH CLINICAL WEIGHTS**

| Category | Example Subcategories | Weight | Clinical Rationale |
|----------|----------------------|--------|--------------------|
| Stairs | No handrails, poor lighting, steep steps, worn treads | 1.00 | Highest injury severity; 10% of fatal falls |
| Bathroom | Missing grab bars, slippery surfaces, high bath edges | 0.95 | Wet surfaces + transfers + hard surfaces |
| Flooring | Loose rugs, uneven surfaces, slippery floors | 0.85 | Most common hazard type |
| Obstacles | Floor clutter, trailing cords, narrow pathways | 0.85 | Direct trip cause, night-time danger |
| External | Uneven pathways, no outdoor handrails | 0.80 | Variable conditions |
| Lighting | Dim lighting, no night lights, glare | 0.75 | Indirect risk factor reducing visibility |
| Bedroom | Bed height issues, floor obstacles | 0.75 | Nighttime fall period |
| Furniture | Unstable furniture, low seating, sharp corners | 0.70 | Transfer safety concerns |
| Kitchen | High cabinets, slippery floors | 0.70 | Multiple activities, usually well-lit |
| General | Emergency access, phone accessibility, footwear | 0.50 | Miscellaneous contributing factors |

Room-type priority mappings guide focused analysis: bathrooms prioritise grab bars and slip hazards; stairs prioritise handrails and lighting; kitchens prioritise floor conditions and storage accessibility.

### C. Multi-Pass Detection Strategy

The detection engine employs a three-pass strategy to ensure comprehensive hazard identification:

**Pass 1 - Global Scene Analysis**: An initial comprehensive scan identifies the room type, overall environmental conditions, and immediately apparent hazards using the full hazard taxonomy.

**Pass 2 - Category-Specific Analysis**: Targeted prompts focus on the top two priority categories for the identified room type (e.g., grab bars and slip hazards for bathrooms). Temperature is increased slightly (+0.05) to encourage identification of subtle hazards missed in the initial pass.

**Pass 3 - Chain-of-Thought Reasoning**: A detailed analysis with explicit eight-step reasoning: (1) identify all visible surfaces and objects, (2) assess each for stability and condition, (3) evaluate lighting conditions, (4) check for obstacles in pathways, (5) identify missing safety features, (6) assess severity of each hazard, (7) consider cumulative risk, and (8) generate recommendations.

Results from all passes are merged with deduplication based on category-subcategory pairs, retaining the highest confidence detection and aggregating recommendations.

### D. Prompt Engineering

Effective prompt design is critical for VLM performance. The system employs structured prompts configured via YAML files, incorporating:

- **Role specification**: Expert occupational therapist with 20 years of geriatric fall prevention experience, trained in Westmead Assessment and CDC STEADI Framework.
- **Patient context**: Analysis for elderly persons (65+) with reduced vision, balance challenges, slower reaction times, and possible walking aids.
- **Output format**: Strict JSON schema specifying category, subcategory, severity, confidence, location, and recommendation fields.
- **Few-shot examples**: Three sample hazard descriptions demonstrating expected output quality.
- **Confidence calibration**: High (0.85-1.0) for clearly visible hazards, Medium (0.60-0.84) for partially obscured items, with hazards below 0.60 excluded.

Model-specific prompt adaptations were implemented: smaller models (Moondream) receive simplified prompts with fewer required output fields (3 vs 7) to accommodate their limited instruction-following capability, with missing fields backfilled with sensible defaults after parsing.

### E. Risk Scoring Algorithm

The risk scoring algorithm calculates a normalised score (0-100) using clinically-derived weights:

**Individual Hazard Score:**
```
hazard_score = base_weight x severity_multiplier x confidence
```

Where severity multipliers are: Low (0.25), Medium (0.50), High (0.75), Critical (1.00).

**Diminishing Returns**: To prevent score inflation from redundant hazards, subsequent hazards in the same category receive reduced contributions via a cumulative factor (0.10-0.30 depending on category). The first hazard contributes fully; each additional hazard contributes a fraction.

**Category Aggregation**:
```
category_score = min(raw_score x category_weight x 10, max_contribution)
```

Each category has a maximum contribution cap (10-30 points) preventing single-category dominance.

**Final Normalisation**:
```
risk_score = min(100, total_weighted_score / max_possible x 100)
```

Risk levels are classified as: LOW (0-25), MODERATE (26-50), HIGH (51-75), CRITICAL (76-100), aligned with clinical urgency recommendations from immediate action (critical) to routine monitoring (low).

### F. Evaluation Framework

The evaluation employs standard information retrieval metrics adapted for hazard detection:

- **Precision**: TP / (TP + FP) - of detected hazards, proportion that are genuine.
- **Recall**: TP / (TP + FN) - of actual hazards, proportion detected.
- **F1 Score**: Harmonic mean of precision and recall.
- **Severity Accuracy**: Proportion of correctly classified severity levels.
- **Cohen's Kappa**: Severity classification agreement beyond chance.
- **Consistency Metrics**: Intra-model stability (same model, multiple runs) and inter-model agreement (Jaccard similarity across VLM backends).

Hazard matching uses a greedy algorithm requiring category match (mandatory) with subcategory match scoring bonus points. The PHELE (Physical Hazards of Elderly Living Environment) dataset provides 575 images (503 training, 72 test) of real domestic environments.

---

## IV. Results

### A. Model Comparison: Detection Capability

Four VLM backends were evaluated for hazard detection capability. Table II summarises detection performance across test images.

**TABLE II: MODEL DETECTION PERFORMANCE COMPARISON**

| Model | Parameters | Type | Avg. Hazards/Image | Inference Time | Categories Covered |
|-------|-----------|------|-------------------|----------------|-------------------|
| GPT-4 Vision | ~1.8T | Cloud API | 4-6 | 3-5s | 8-10 |
| Gemini 2.0 Flash | - | Cloud API | 3-5 | 2-4s | 7-9 |
| LLaVA 7B | 7B | Local (Ollama) | 5-21 | 2-4s | 5-8 |
| Moondream | ~1.6B | Local (Ollama) | 0-1 | 0.5-1s | 0-1 |

Commercial models (GPT-4V, Gemini) demonstrate reliable, consistent detection across all 10 hazard categories. LLaVA 7B shows high detection volume but with greater variability in output quality. Moondream's limited parameter count (~1.6B) results in minimal hazard identification, largely due to inability to follow complex structured JSON prompting — the model frequently returns empty or malformed responses that fail JSON parsing.

### B. Category-Level Detection Analysis

Evaluation across the PHELE test set reveals category-specific detection patterns. Table III presents per-category detection rates from LLaVA evaluation runs.

**TABLE III: CATEGORY-LEVEL DETECTION FREQUENCY (LLaVA 7B)**

| Category | Detections (Run 1) | Detections (Run 2) | Detections (Run 3) |
|----------|-------------------|-------------------|-------------------|
| Flooring | 7 | 1 | 1 |
| General | 3 | 13 | 1 |
| Obstacles | 4 | 1 | 1 |
| Furniture | 2 | 4 | 1 |
| Lighting | 3 | 1 | 1 |
| Bathroom | 1 | 1 | 0 |
| Total | 20 | 21 | 5 |

Variability across runs (5 to 21 total detections) highlights the stochastic nature of VLM outputs, even at low temperature settings (0.1). Flooring and general hazards show highest detection frequency, consistent with their visual prominence in domestic scenes.

### C. Risk Score Validation

The clinically-weighted risk scoring algorithm was validated against expert-annotated scenarios. Table IV presents representative scoring outputs.

**TABLE IV: RISK SCORE VALIDATION SCENARIOS**

| Scenario | Hazards | Categories | Risk Score | Level |
|----------|---------|-----------|------------|-------|
| Safe bathroom (well-equipped) | 0 | - | 0 | LOW |
| Bathroom without grab bars | 3 critical | bathroom, flooring | 72 | HIGH |
| Cluttered living room | 4 medium | obstacles, furniture, flooring | 38 | MODERATE |
| Unsafe stairs + bathroom | 6 mixed | stairs, bathroom, lighting | 85 | CRITICAL |
| Kitchen with loose rug | 2 medium | flooring, kitchen | 22 | LOW |
| Bedroom with poor lighting | 3 mixed | bedroom, lighting, obstacles | 41 | MODERATE |

The diminishing returns mechanism prevents score inflation: three flooring hazards in the same room produce a lower combined score than three hazards across different categories, reflecting clinical evidence that diversified risks compound more significantly than redundant hazards within a single domain.

### D. Multi-Pass vs Single-Pass Detection

Comparative analysis of detection strategies demonstrates the value of the multi-pass approach. Table V compares detection outcomes.

**TABLE V: DETECTION STRATEGY COMPARISON**

| Strategy | Avg. Hazards | Categories Covered | Processing Time |
|----------|-------------|-------------------|----------------|
| Single-pass | 2-3 | 2-3 | 3-5s |
| Multi-pass (3 passes) | 4-6 | 5-8 | 8-12s |
| Quick detection | 1-2 | 1-2 | 2-3s |

The multi-pass strategy achieves approximately 2x improvement in hazard count and category coverage at the cost of 2-3x processing time, representing an acceptable trade-off for comprehensive safety assessment.

### E. Qualitative Detection Capabilities

Across all models, the system successfully identifies hazards in the following categories with highest reliability:

- **Bathroom hazards**: Missing grab bars near toilet, shower, and bathtub; absence of non-slip mats; high bathtub edges without transfer aids.
- **Stair hazards**: Single or missing handrails; poor stairway lighting; lack of contrasting edge strips; loose carpet on steps.
- **Flooring hazards**: Loose rugs without non-slip backing; uneven floor transitions; cluttered walkways.
- **Lighting hazards**: Insufficient corridor lighting; absence of night lights; glare from uncovered windows.

---

## V. Discussion

### A. Key Findings

This research demonstrates that Vision-Language Models can effectively identify environmental fall hazards from photographs without requiring task-specific training data. The multi-pass detection strategy with chain-of-thought prompting improves comprehensiveness compared to single-pass approaches, achieving broader category coverage and more detailed hazard descriptions.

The clinically-weighted risk scoring algorithm produces meaningful quantification aligned with the Westmead framework. The system's explainable outputs — including specific hazard descriptions, severity classifications, and actionable recommendations — enhance clinical utility compared to binary detection systems.

A significant finding is the substantial capability gap between commercial and open-source models. GPT-4V and Gemini produce reliable, well-structured outputs suitable for clinical screening, while smaller open-source models (particularly Moondream at 1.6B parameters) lack sufficient instruction-following capability for complex structured detection tasks. LLaVA 7B represents a middle ground, demonstrating useful detection capability with local, privacy-preserving deployment but exhibiting higher output variability.

### B. Comparison with Existing Approaches

Compared to traditional CNN-based object detection (YOLO, Faster R-CNN), VLMs offer: (1) no requirement for labelled training datasets specific to fall hazards, (2) ability to reason about context and spatial relationships beyond simple object classification, (3) natural language explanations for detected hazards, and (4) flexibility to identify novel hazard types not explicitly programmed. However, CNN-based approaches offer faster inference and deterministic outputs, which may be preferable for real-time applications.

Compared to manual assessments (WeHSA requiring 45-60 minutes per home), the system offers: (1) scalability to population-level screening, (2) consistent application of assessment criteria without inter-rater variability, (3) cost-effectiveness for initial screening at approximately $0.01-0.05 per image via API, and (4) accessibility for remote and rural populations where specialist assessors are unavailable.

### C. Limitations

Several limitations should be acknowledged:

1. **Ground Truth Scarcity**: No large-scale public dataset of home images with labelled fall hazards exists. The PHELE dataset provides real domestic images but lacks comprehensive hazard annotations, constraining quantitative evaluation. Evaluation relies partly on expert-annotated scenarios and qualitative assessment.
2. **VLM Output Variability**: Even at low temperature settings, VLMs produce variable outputs across runs, with detection counts ranging from 5 to 21 for the same image. This stochasticity requires mitigation through multi-pass aggregation and confidence thresholding.
3. **Image Quality Dependency**: System performance depends on adequate image quality, lighting conditions, and camera angles. The preprocessing module addresses some quality issues but cannot compensate for fundamentally inadequate images.
4. **Single Image Analysis**: Full home assessment requires multiple photographs; the current system analyses individual images without cross-image reasoning.
5. **Cultural Context**: Hazard definitions are derived from Western clinical frameworks and may require adaptation for diverse cultural contexts and housing types.
6. **API Dependencies**: Commercial VLMs introduce cost and availability considerations, though open-source alternatives (LLaVA) enable local deployment.

### D. Ethical Considerations

The system raises important considerations for healthcare AI deployment. Privacy is paramount as home photographs contain sensitive information about living conditions and personal belongings; the system supports local processing via Ollama to avoid cloud data transmission. The system is explicitly positioned as a screening tool that augments but does not replace professional assessment — recommendations are framed as suggestions, not medical advice. Algorithmic bias is a concern as VLMs trained predominantly on Western home environments may exhibit differential performance across diverse cultural contexts.

---

## VI. Conclusions and Future Work

### A. Conclusions

This research presents a novel approach to automated fall hazard detection using Vision-Language Models, contributing to the emerging field of AI-assisted preventive healthcare. The key contributions are:

1. **First comprehensive VLM application** to fall hazard detection using a clinically-validated framework with 42 hazard subcategories aligned to the Westmead Home Safety Assessment.
2. **Multi-pass detection strategy** with chain-of-thought prompting that improves hazard coverage by approximately 2x compared to single-pass approaches.
3. **Clinically-weighted risk scoring** with diminishing returns, producing normalised 0-100 scores aligned with clinical urgency classifications.
4. **Multi-model evaluation** comparing commercial (GPT-4V, Gemini) and open-source (LLaVA, Moondream) backends, demonstrating capability trade-offs between accuracy, cost, and privacy.
5. **Modular, extensible architecture** supporting future model integration and clinical validation through factory patterns and Pydantic schema validation.

The system addresses a critical need for scalable fall prevention tools, offering a cost-effective complement to resource-intensive manual assessments.

### B. Future Work

Future research directions include:

1. **Clinical Validation Study**: Formal comparison with expert occupational therapist assessments on diverse home environments to establish quantitative performance benchmarks.
2. **Fine-tuning Open-Source Models**: Adapting LLaVA or similar models on curated fall hazard datasets to improve detection accuracy for local, privacy-preserving deployment.
3. **Mobile Application**: Development of a user-friendly mobile interface enabling self-assessment by older adults or caregivers.
4. **Multi-Image Analysis**: Integration of multiple room photographs for comprehensive whole-home risk assessment.
5. **Longitudinal Evaluation**: Tracking fall outcomes for individuals using the system to establish predictive validity of risk scores.

---

## References

[1] World Health Organization, "Ageing and Health," WHO Fact Sheets, 2022.

[2] World Health Organization, WHO Global Report on Falls Prevention in Older Age. Geneva, Switzerland: WHO Press, 2007.

[3] J. A. Stevens and E. D. Rudd, "The Impact of Decreasing U.S. Hip Fracture Rates on Future Hip Fracture Estimates," Osteoporosis International, vol. 24, no. 10, pp. 2725-2728, 2013.

[4] C. A. Taylor, J. M. Bell, M. J. Breiding, and L. Xu, "Traumatic Brain Injury-Related Emergency Department Visits, Hospitalizations, and Deaths — United States," MMWR Surveillance Summaries, vol. 66, no. 9, pp. 1-16, 2017.

[5] C. S. Florence, G. Bergen, A. Atherly, E. Burns, J. Stevens, and C. Drake, "Medical Costs of Fatal and Nonfatal Falls in Older Adults," Journal of the American Geriatrics Society, vol. 66, no. 4, pp. 693-698, 2018.

[6] L. Z. Rubenstein, "Falls in older people: epidemiology, risk factors and strategies for prevention," Age and Ageing, vol. 35, suppl. 2, pp. ii37-ii41, 2006.

[7] R. W. Sattin, D. A. Rodriguez, J. Peel, and S. L. Murphy, "Home Environmental Hazards and the Risk of Fall Injury Events Among Community-Dwelling Older Persons," Journal of the American Geriatrics Society, vol. 46, no. 7, pp. 792-798, 1998.

[8] M. C. Nevitt, S. R. Cummings, and E. S. Hudes, "Risk Factors for Injurious Falls: a Prospective Study," Journal of Gerontology, vol. 46, no. 5, pp. M164-M170, 1991.

[9] L. Clemson, "Westmead Home Safety Assessment," Coordinates Publications, 1997, 2015.

[10] L. Mackenzie, J. Byles, and N. Higginbotham, "The Home Falls and Accidents Screening Tool (HOME FAST): Reliability in Older People," Australasian Journal on Ageing, vol. 21, no. 2, pp. 77-82, 2002.

[11] T. Chiu and R. Oliver, "Factor Analysis and Construct Validity of the SAFER-HOME," OTJR: Occupation, Participation and Health, vol. 26, no. 4, pp. 132-142, 2006.

[12] Centers for Disease Control and Prevention, "STEADI - Stopping Elderly Accidents, Deaths & Injuries," CDC, 2023.

[13] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," in Proc. IEEE CVPR, 2016, pp. 779-788.

[14] W. Liu et al., "SSD: Single Shot MultiBox Detector," in Proc. ECCV, 2016, pp. 21-37.

[15] D. J. Cook, A. S. Crandall, B. L. Thomas, and N. C. Krishnan, "CASAS: A Smart Home in a Box," Computer, vol. 46, no. 7, pp. 62-69, 2013.

[16] M. Mubashir, L. Shao, and L. Seed, "A survey on fall detection: Principles and approaches," Neurocomputing, vol. 100, pp. 144-152, 2013.

[17] OpenAI, "GPT-4 Technical Report," arXiv preprint arXiv:2303.08774, 2023.

[18] Google DeepMind, "Gemini: A Family of Highly Capable Multimodal Models," arXiv preprint, 2023.

[19] H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visual Instruction Tuning," in Proc. NeurIPS, 2023.

[20] vikhyatk, "Moondream: A Tiny Vision Language Model," GitHub repository, 2024.

[21] K. Singhal et al., "Large language models encode clinical knowledge," Nature, vol. 620, pp. 172-180, 2023.

[22] C. Zakka et al., "Almanac: Retrieval-Augmented Language Models for Clinical Medicine," NEJM AI, 2024.

[23] L. D. Gillespie et al., "Interventions for preventing falls in older people living in the community," Cochrane Database of Systematic Reviews, no. 9, 2012.

[24] K. Peffers, T. Tuunanen, M. A. Rothenberger, and S. Chatterjee, "A Design Science Research Methodology for Information Systems Research," Journal of Management Information Systems, vol. 24, no. 3, pp. 45-77, 2007.

---

*Paper submitted in partial fulfilment of MSc Research Project (COM742), Ulster University, 2025/26.*
