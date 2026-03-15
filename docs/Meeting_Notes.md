# Research Project: Fall Hazard Detection Using Vision-Language Models

**Student:** Summen Zahid (B00996747) | **Supervisor:** Mark Donnelly

---

## Research Question

**Can Vision-Language Models (VLMs) reliably detect environmental fall hazards in home settings with accuracy comparable to clinical assessments?**

---

## Evaluation Metrics

### Detection Metrics
| Metric | Formula | Purpose |
|--------|---------|---------|
| **Precision** | TP / (TP + FP) | Of detected hazards, how many are real? |
| **Recall** | TP / (TP + FN) | Of real hazards, how many were found? |
| **F1 Score** | 2 × (P × R) / (P + R) | Balance of precision and recall |

### Severity Classification
| Metric | Purpose |
|--------|---------|
| **Accuracy** | % correct severity levels (low/medium/high/critical) |
| **Cohen's Kappa** | Agreement beyond chance (>0.6 = good, >0.8 = excellent) |

### Consistency
| Metric | Purpose |
|--------|---------|
| **Intra-model** | Same model, multiple runs - stability |
| **Inter-model** | Gemini vs GPT-4V - agreement |
| **Jaccard Similarity** | Category overlap: \|A ∩ B\| / \|A ∪ B\| |

---

## Clinical Framework (Westmead Home Safety Assessment)

| Category | Weight | Rationale |
|----------|--------|-----------|
| Stairs | 1.00 | Highest injury severity |
| Bathroom | 0.95 | Wet surfaces + transfers |
| Flooring | 0.85 | Most common hazard |
| Obstacles | 0.85 | Direct trip cause |
| External | 0.80 | Variable conditions |
| Bedroom | 0.75 | Nighttime falls |
| Lighting | 0.75 | Reduces visibility |
| Furniture | 0.70 | Unstable support |
| Kitchen | 0.70 | Spills, reaching |
| General | 0.50 | Footwear, pets |

**Severity Multipliers:** Low (0.25) → Medium (0.50) → High (0.75) → Critical (1.00)

**Risk Score:** 0-100 scale → Low (0-25), Moderate (26-50), High (51-75), Critical (76-100)

---

## Research Methodology

1. **Literature Review** → Clinical fall risk factors, existing tools
2. **Framework Design** → Map Westmead to VLM prompts
3. **Data Collection** → PHELE dataset + staged images
4. **Model Evaluation** → Gemini/GPT-4V vs ground truth
5. **Consistency Testing** → Multiple runs, varied conditions
6. **Clinical Validation** → Compare to Westmead checklist
7. **Limitation Analysis** → Edge cases (clutter, poor lighting)

---

## Models Under Evaluation

- **Google Gemini Vision** (Primary)
- **GPT-4 Vision** (Comparison)

**Why VLMs?** Provide explainable outputs - not just "hazard detected" but "loose rug near doorway poses trip risk"

---

## Known Limitations

| Limitation | Mitigation |
|------------|------------|
| Cluttered scenes | Multi-angle images |
| Low-quality images | Preprocessing |
| Lighting variation | Brightness normalization |
| Model hallucinations | Confidence thresholds |
| No depth perception | Acknowledge in limitations |

---

## Research Contribution

Empirical evaluation of VLM reliability for proactive home hazard assessment - a novel application in healthcare-adjacent AI with clinical validation against established frameworks.

**Hypothesis:** VLMs can detect home hazards with ≥70% F1 score and provide clinically useful explanations.
