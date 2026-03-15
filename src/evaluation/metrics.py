"""
Evaluation Metrics for Fall Hazard Detection

Provides comprehensive metrics for evaluating hazard detection
accuracy, severity classification, and model consistency.
"""

from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from enum import Enum

# Try to import sklearn metrics, provide fallbacks if not available
try:
    from sklearn.metrics import (
        precision_recall_fscore_support,
        confusion_matrix,
        cohen_kappa_score,
        accuracy_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class DetectionMatch:
    """
    Represents a match between predicted and ground truth hazard.

    Attributes:
        predicted: Predicted hazard dict (or None for false negative)
        ground_truth: Ground truth hazard dict (or None for false positive)
        category_match: Whether categories match
        subcategory_match: Whether subcategories match
        severity_match: Whether severity levels match
    """
    predicted: Optional[Dict]
    ground_truth: Optional[Dict]
    category_match: bool = False
    subcategory_match: bool = False
    severity_match: bool = False


@dataclass
class CategoryMetrics:
    """
    Metrics for a specific hazard category.

    Attributes:
        category: Category name
        true_positives: Count of correct detections
        false_positives: Count of incorrect detections
        false_negatives: Count of missed hazards
        precision: TP / (TP + FP)
        recall: TP / (TP + FN)
        f1_score: Harmonic mean of precision and recall
    """
    category: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0


@dataclass
class EvaluationResult:
    """
    Complete evaluation results.

    Attributes:
        overall_precision: Weighted average precision
        overall_recall: Weighted average recall
        overall_f1: Weighted average F1 score
        category_metrics: Metrics per category
        severity_accuracy: Accuracy of severity classification
        severity_kappa: Cohen's Kappa for severity agreement
        confusion_matrix: Category confusion matrix
        total_predictions: Total number of predictions
        total_ground_truth: Total ground truth hazards
        matches: List of prediction-ground truth matches
    """
    overall_precision: float
    overall_recall: float
    overall_f1: float
    category_metrics: Dict[str, CategoryMetrics]
    severity_accuracy: float
    severity_kappa: float
    confusion_matrix: Optional[np.ndarray]
    total_predictions: int
    total_ground_truth: int
    matches: List[DetectionMatch] = field(default_factory=list)


class HazardEvaluator:
    """
    Evaluate hazard detection performance against ground truth.

    Provides:
    - Per-category precision, recall, F1
    - Severity classification accuracy and Cohen's Kappa
    - Confusion matrices
    - Match analysis between predictions and ground truth

    Usage:
        evaluator = HazardEvaluator()

        # Single image evaluation
        result = evaluator.evaluate(predictions, ground_truth)

        # Batch evaluation
        result = evaluator.evaluate_batch(all_predictions, all_ground_truths)
    """

    # Valid categories for evaluation
    VALID_CATEGORIES = {
        "flooring", "lighting", "obstacles", "furniture",
        "stairs", "bathroom", "kitchen", "bedroom",
        "external", "general"
    }

    # Severity levels in order
    SEVERITY_LEVELS = ["low", "medium", "high", "critical"]

    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize the evaluator.

        Args:
            iou_threshold: IoU threshold for bounding box matching
                          (only used if bounding boxes are provided)
        """
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> EvaluationResult:
        """
        Evaluate predictions against ground truth for a single image.

        Args:
            predictions: List of predicted hazard dicts with keys:
                - category: str
                - subcategory: str
                - severity: str
                - confidence: float (optional)
                - bounding_box: dict (optional)
            ground_truth: List of ground truth hazard dicts with same keys

        Returns:
            EvaluationResult with all metrics
        """
        # Match predictions to ground truth
        matches = self._match_hazards(predictions, ground_truth)

        # Calculate category metrics
        category_metrics = self._calculate_category_metrics(matches)

        # Calculate overall metrics
        overall = self._calculate_overall_metrics(category_metrics)

        # Calculate severity metrics
        severity_acc, severity_kappa = self._calculate_severity_metrics(matches)

        # Build confusion matrix
        conf_matrix = self._build_confusion_matrix(matches)

        return EvaluationResult(
            overall_precision=overall['precision'],
            overall_recall=overall['recall'],
            overall_f1=overall['f1'],
            category_metrics=category_metrics,
            severity_accuracy=severity_acc,
            severity_kappa=severity_kappa,
            confusion_matrix=conf_matrix,
            total_predictions=len(predictions),
            total_ground_truth=len(ground_truth),
            matches=matches
        )

    def evaluate_batch(
        self,
        all_predictions: List[List[Dict]],
        all_ground_truths: List[List[Dict]]
    ) -> EvaluationResult:
        """
        Evaluate predictions across multiple images.

        Args:
            all_predictions: List of prediction lists (one per image)
            all_ground_truths: List of ground truth lists (one per image)

        Returns:
            Aggregated EvaluationResult
        """
        # Flatten all predictions and ground truths
        flat_predictions = []
        flat_ground_truth = []

        for preds, gts in zip(all_predictions, all_ground_truths):
            flat_predictions.extend(preds)
            flat_ground_truth.extend(gts)

        return self.evaluate(flat_predictions, flat_ground_truth)

    def _match_hazards(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> List[DetectionMatch]:
        """
        Match predictions to ground truth hazards.

        Uses greedy matching based on category and subcategory.
        """
        matches = []
        used_gt_indices: Set[int] = set()

        # Sort predictions by confidence (if available) for better matching
        sorted_preds = sorted(
            enumerate(predictions),
            key=lambda x: x[1].get('confidence', 0.5),
            reverse=True
        )

        # Match each prediction
        for pred_idx, pred in sorted_preds:
            best_match_idx = None
            best_match_score = 0

            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in used_gt_indices:
                    continue

                score = self._match_score(pred, gt)
                if score > best_match_score:
                    best_match_score = score
                    best_match_idx = gt_idx

            if best_match_idx is not None and best_match_score > 0:
                # Found a match
                gt = ground_truth[best_match_idx]
                used_gt_indices.add(best_match_idx)

                matches.append(DetectionMatch(
                    predicted=pred,
                    ground_truth=gt,
                    category_match=pred.get('category', '').lower() == gt.get('category', '').lower(),
                    subcategory_match=pred.get('subcategory', '').lower() == gt.get('subcategory', '').lower(),
                    severity_match=pred.get('severity', '').lower() == gt.get('severity', '').lower()
                ))
            else:
                # False positive
                matches.append(DetectionMatch(
                    predicted=pred,
                    ground_truth=None,
                    category_match=False,
                    subcategory_match=False,
                    severity_match=False
                ))

        # Add false negatives (unmatched ground truth)
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx not in used_gt_indices:
                matches.append(DetectionMatch(
                    predicted=None,
                    ground_truth=gt,
                    category_match=False,
                    subcategory_match=False,
                    severity_match=False
                ))

        return matches

    def _match_score(self, pred: Dict, gt: Dict) -> float:
        """
        Calculate matching score between prediction and ground truth.

        Higher score = better match.
        """
        score = 0.0

        # Category match (required for any score)
        pred_cat = pred.get('category', '').lower()
        gt_cat = gt.get('category', '').lower()

        if pred_cat != gt_cat:
            return 0.0

        score += 1.0

        # Subcategory match (bonus)
        pred_sub = pred.get('subcategory', '').lower()
        gt_sub = gt.get('subcategory', '').lower()

        if pred_sub == gt_sub:
            score += 1.0
        elif pred_sub in gt_sub or gt_sub in pred_sub:
            score += 0.5

        # Bounding box IoU (if available)
        if 'bounding_box' in pred and 'bounding_box' in gt:
            iou = self._calculate_iou(pred['bounding_box'], gt['bounding_box'])
            if iou >= self.iou_threshold:
                score += iou

        return score

    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """Calculate Intersection over Union for two bounding boxes."""
        x1 = max(box1.get('x', 0), box2.get('x', 0))
        y1 = max(box1.get('y', 0), box2.get('y', 0))
        x2 = min(
            box1.get('x', 0) + box1.get('width', 0),
            box2.get('x', 0) + box2.get('width', 0)
        )
        y2 = min(
            box1.get('y', 0) + box1.get('height', 0),
            box2.get('y', 0) + box2.get('height', 0)
        )

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1.get('width', 0) * box1.get('height', 0)
        area2 = box2.get('width', 0) * box2.get('height', 0)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _calculate_category_metrics(
        self,
        matches: List[DetectionMatch]
    ) -> Dict[str, CategoryMetrics]:
        """Calculate precision, recall, F1 per category."""
        category_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'tp': 0, 'fp': 0, 'fn': 0}
        )

        for match in matches:
            if match.predicted is not None and match.ground_truth is not None:
                # True positive
                category = match.ground_truth.get('category', 'unknown').lower()
                category_counts[category]['tp'] += 1
            elif match.predicted is not None:
                # False positive
                category = match.predicted.get('category', 'unknown').lower()
                category_counts[category]['fp'] += 1
            elif match.ground_truth is not None:
                # False negative
                category = match.ground_truth.get('category', 'unknown').lower()
                category_counts[category]['fn'] += 1

        metrics = {}
        for category, counts in category_counts.items():
            tp, fp, fn = counts['tp'], counts['fp'], counts['fn']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[category] = CategoryMetrics(
                category=category,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1_score=round(f1, 4)
            )

        return metrics

    def _calculate_overall_metrics(
        self,
        category_metrics: Dict[str, CategoryMetrics]
    ) -> Dict[str, float]:
        """Calculate weighted average metrics across categories."""
        if not category_metrics:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        total_tp = sum(m.true_positives for m in category_metrics.values())
        total_fp = sum(m.false_positives for m in category_metrics.values())
        total_fn = sum(m.false_negatives for m in category_metrics.values())

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4)
        }

    def _calculate_severity_metrics(
        self,
        matches: List[DetectionMatch]
    ) -> Tuple[float, float]:
        """Calculate severity classification accuracy and Kappa."""
        y_true = []
        y_pred = []

        for match in matches:
            if match.predicted is not None and match.ground_truth is not None:
                gt_sev = match.ground_truth.get('severity', 'medium').lower()
                pred_sev = match.predicted.get('severity', 'medium').lower()

                if gt_sev in self.SEVERITY_LEVELS and pred_sev in self.SEVERITY_LEVELS:
                    y_true.append(self.SEVERITY_LEVELS.index(gt_sev))
                    y_pred.append(self.SEVERITY_LEVELS.index(pred_sev))

        if not y_true:
            return 0.0, 0.0

        # Accuracy
        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

        # Cohen's Kappa (weighted)
        if SKLEARN_AVAILABLE:
            kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        else:
            # Simple Kappa calculation fallback
            kappa = self._simple_kappa(y_true, y_pred)

        return round(accuracy, 4), round(kappa, 4)

    def _simple_kappa(self, y_true: List[int], y_pred: List[int]) -> float:
        """Simple Cohen's Kappa calculation without sklearn."""
        n = len(y_true)
        if n == 0:
            return 0.0

        # Observed agreement
        po = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n

        # Expected agreement
        classes = set(y_true) | set(y_pred)
        pe = 0.0
        for c in classes:
            true_c = sum(1 for t in y_true if t == c) / n
            pred_c = sum(1 for p in y_pred if p == c) / n
            pe += true_c * pred_c

        # Kappa
        if pe == 1:
            return 1.0
        return (po - pe) / (1 - pe)

    def _build_confusion_matrix(
        self,
        matches: List[DetectionMatch]
    ) -> Optional[np.ndarray]:
        """Build confusion matrix for category predictions."""
        categories = sorted(self.VALID_CATEGORIES)
        cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

        y_true = []
        y_pred = []

        for match in matches:
            if match.predicted is not None and match.ground_truth is not None:
                gt_cat = match.ground_truth.get('category', '').lower()
                pred_cat = match.predicted.get('category', '').lower()

                if gt_cat in cat_to_idx and pred_cat in cat_to_idx:
                    y_true.append(cat_to_idx[gt_cat])
                    y_pred.append(cat_to_idx[pred_cat])

        if not y_true:
            return None

        if SKLEARN_AVAILABLE:
            return confusion_matrix(y_true, y_pred, labels=range(len(categories)))
        else:
            # Simple confusion matrix
            n_classes = len(categories)
            matrix = np.zeros((n_classes, n_classes), dtype=int)
            for t, p in zip(y_true, y_pred):
                matrix[t][p] += 1
            return matrix

    def generate_report(self, result: EvaluationResult) -> str:
        """
        Generate a text report from evaluation results.

        Args:
            result: EvaluationResult object

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "HAZARD DETECTION EVALUATION REPORT",
            "=" * 60,
            "",
            "OVERALL METRICS",
            "-" * 40,
            f"  Precision: {result.overall_precision:.4f}",
            f"  Recall:    {result.overall_recall:.4f}",
            f"  F1 Score:  {result.overall_f1:.4f}",
            "",
            f"  Total Predictions:  {result.total_predictions}",
            f"  Total Ground Truth: {result.total_ground_truth}",
            "",
            "SEVERITY CLASSIFICATION",
            "-" * 40,
            f"  Accuracy:      {result.severity_accuracy:.4f}",
            f"  Cohen's Kappa: {result.severity_kappa:.4f}",
            "",
            "PER-CATEGORY METRICS",
            "-" * 40,
        ]

        # Sort categories by F1 score
        sorted_cats = sorted(
            result.category_metrics.items(),
            key=lambda x: x[1].f1_score,
            reverse=True
        )

        for cat, metrics in sorted_cats:
            lines.append(f"\n  {cat.upper()}")
            lines.append(f"    TP: {metrics.true_positives}, "
                        f"FP: {metrics.false_positives}, "
                        f"FN: {metrics.false_negatives}")
            lines.append(f"    Precision: {metrics.precision:.4f}, "
                        f"Recall: {metrics.recall:.4f}, "
                        f"F1: {metrics.f1_score:.4f}")

        lines.extend(["", "=" * 60])

        return "\n".join(lines)


class ConsistencyEvaluator:
    """
    Evaluate model consistency across multiple runs.

    Tests:
    - Intra-model: Same model, same image, multiple runs
    - Inter-model: Different models, same image
    """

    def __init__(self):
        self.severity_order = ["low", "medium", "high", "critical"]

    def evaluate_intra_consistency(
        self,
        results: List[List[Dict]]
    ) -> Dict[str, float]:
        """
        Evaluate consistency across multiple runs of the same model.

        Args:
            results: List of detection results (each is list of hazards)

        Returns:
            Consistency metrics
        """
        if len(results) < 2:
            return {'hazard_count_std': 0.0, 'category_agreement': 1.0}

        # Hazard count consistency
        counts = [len(r) for r in results]
        count_std = np.std(counts)

        # Category agreement (Jaccard similarity)
        category_sets = [
            set(h.get('category', '').lower() for h in r)
            for r in results
        ]

        similarities = []
        for i in range(len(category_sets)):
            for j in range(i + 1, len(category_sets)):
                set_i, set_j = category_sets[i], category_sets[j]
                if not set_i and not set_j:
                    similarities.append(1.0)
                elif not set_i or not set_j:
                    similarities.append(0.0)
                else:
                    jaccard = len(set_i & set_j) / len(set_i | set_j)
                    similarities.append(jaccard)

        avg_similarity = np.mean(similarities) if similarities else 1.0

        return {
            'hazard_count_std': round(count_std, 2),
            'hazard_count_mean': round(np.mean(counts), 2),
            'category_agreement': round(avg_similarity, 4)
        }

    def evaluate_inter_consistency(
        self,
        model_results: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """
        Evaluate agreement between different models.

        Args:
            model_results: Dict mapping model name to detection results

        Returns:
            Agreement metrics between model pairs
        """
        model_names = list(model_results.keys())
        pairwise = {}

        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                result1 = model_results[name1]
                result2 = model_results[name2]

                cats1 = set(h.get('category', '').lower() for h in result1)
                cats2 = set(h.get('category', '').lower() for h in result2)

                if not cats1 and not cats2:
                    jaccard = 1.0
                elif not cats1 or not cats2:
                    jaccard = 0.0
                else:
                    jaccard = len(cats1 & cats2) / len(cats1 | cats2)

                pair_key = f"{name1}_vs_{name2}"
                pairwise[pair_key] = {
                    'category_jaccard': round(jaccard, 4),
                    'count_diff': abs(len(result1) - len(result2)),
                    'common_categories': list(cats1 & cats2)
                }

        return {
            'pairwise_agreement': pairwise,
            'mean_jaccard': round(
                np.mean([p['category_jaccard'] for p in pairwise.values()]),
                4
            ) if pairwise else 0.0
        }
