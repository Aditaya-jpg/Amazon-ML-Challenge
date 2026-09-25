"""
LightGBM entity matching model and macro F_0.5 threshold optimizer.
"""

import collections
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import lightgbm as lgb
from src.features import FEATURE_NAMES


def compute_entity_f05(
    y_true_set: Set[str],
    y_pred_set: Set[str],
) -> float:
    """
    Compute F_0.5 score for a single Source 1 entity:
    - Singletons (y_true empty): 1.0 if y_pred empty, else 0.0
    - Non-singletons: standard F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    if len(y_true_set) == 0:
        return 1.0 if len(y_pred_set) == 0 else 0.0
    
    if len(y_pred_set) == 0:
        return 0.0

    tp = len(y_true_set.intersection(y_pred_set))
    precision = tp / len(y_pred_set)
    recall = tp / len(y_true_set)

    if precision == 0.0 or recall == 0.0:
        return 0.0

    denominator = 0.25 * precision + recall
    if denominator == 0.0:
        return 0.0

    return (1.25 * precision * recall) / denominator


def evaluate_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
) -> float:
    """
    Calculate the macro-average F_0.5 score over all S1 entities in ground_truth.
    """
    total_f05 = 0.0
    total_entities = len(ground_truth)
    if total_entities == 0:
        return 0.0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        total_f05 += compute_entity_f05(true_set, pred_set)

    return total_f05 / total_entities


class EntityMatcherModel:
    """
    LightGBM classifier for scoring candidate pairs and tuning decision thresholds.
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self.clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.08,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            importance_type="gain",
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit LightGBM model on candidate pairs."""
        print(f"Training LightGBM on {len(X):,} pairs (Positive rate: {np.mean(y)*100:.2f}%)...")
        self.clf.fit(X, y)
        print("Feature Importances:")
        for name, imp in zip(FEATURE_NAMES, self.clf.feature_importances_):
            print(f"  {name:25s}: {imp:8.2f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of match."""
        if len(X) == 0:
            return np.array([])
        return self.clf.predict_proba(X)[:, 1]

    def optimize_threshold(
        self,
        X_val: np.ndarray,
        pair_ids_val: List[Tuple[str, str]],
        val_ground_truth: Dict[str, Set[str]],
        threshold_candidates: Optional[List[float]] = None,
    ) -> float:
        """
        Grid search for decision threshold T* that maximizes macro F_0.5 on validation split.
        """
        if threshold_candidates is None:
            threshold_candidates = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

        print("Scoring validation candidate pairs...")
        probs = self.predict_proba(X_val)

        # Pre-group probabilities by s1_id
        s1_to_candidates = collections.defaultdict(list)
        for prob, (s1_id, cand_id) in zip(probs, pair_ids_val):
            s1_to_candidates[s1_id].append((cand_id, prob))

        best_score = -1.0
        best_thresh = self.threshold

        print("\nThreshold Tuning on Validation Holdout (Macro F_0.5):")
        for t in threshold_candidates:
            preds = {}
            for s1_id in val_ground_truth.keys():
                cand_list = s1_to_candidates.get(s1_id, [])
                matched = {cid for cid, p in cand_list if p >= t}
                preds[s1_id] = matched

            score = evaluate_macro_f05(val_ground_truth, preds)
            print(f"  Threshold {t:4.2f} -> Macro F_0.5 = {score:.4f}")
            if score > best_score:
                best_score = score
                best_thresh = t

        print(f"\n=> Optimal Threshold Selected: {best_thresh:.2f} (Macro F_0.5 = {best_score:.4f})")
        self.threshold = best_thresh
        return best_thresh
