"""Comprehensive metric calculation and reporting for QML and Classical models (Review 2).

Calculates standard binary classification metrics, ranking metrics (AUROC, AUPRC),
confusion matrix statistics, per-attack-type recall, and runtime profiling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    attack_types: np.ndarray | list[str] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive performance metrics for binary anomaly classification."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Core rates
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # Continuous ranking metrics
    auroc = None
    auprc = None
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        # AUROC requires both classes to be present in y_true
        if len(np.unique(y_true)) > 1:
            try:
                auroc = float(roc_auc_score(y_true, y_prob))
                auprc = float(average_precision_score(y_true, y_prob))
            except Exception:
                auroc = None
                auprc = None

    metrics: dict[str, Any] = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "specificity": spec,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "auroc": auroc,
        "auprc": auprc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "sample_count": len(y_true),
        "positive_count": int(np.sum(y_true)),
        "negative_count": int(len(y_true) - np.sum(y_true)),
    }

    # Per-attack breakdown if metadata is available
    if attack_types is not None:
        attack_types = np.asarray(attack_types)
        unique_attacks = [a for a in np.unique(attack_types) if a not in ("none", "normal")]
        per_attack: dict[str, dict[str, Any]] = {}
        for atk in unique_attacks:
            mask = (attack_types == atk)
            if np.sum(mask) > 0:
                atk_true = y_true[mask]
                atk_pred = y_pred[mask]
                atk_rec = float(recall_score(atk_true, atk_pred, zero_division=0))
                per_attack[atk] = {
                    "count": int(np.sum(mask)),
                    "detected": int(np.sum((atk_true == 1) & (atk_pred == 1))),
                    "recall": atk_rec,
                }
        metrics["per_attack_recall"] = per_attack

    return metrics
