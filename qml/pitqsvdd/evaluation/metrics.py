"""Evaluation metrics computation for PI-TQSVDD."""

from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_point_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    attack_types: list[str] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute comprehensive point-level anomaly detection metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    auroc = None
    auprc = None
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        if len(np.unique(y_true)) > 1:
            try:
                auroc = float(roc_auc_score(y_true, y_score))
                auprc = float(average_precision_score(y_true, y_score))
            except Exception:
                auroc = None
                auprc = None

    # Per-attack-type breakdown
    per_attack: dict[str, Any] = {}
    if attack_types is not None:
        atk_arr = np.asarray(attack_types)
        unique_atks = [a for a in np.unique(atk_arr) if a not in ("none", "normal", None, "")]
        for atk in unique_atks:
            mask = atk_arr == atk
            atk_true = y_true[mask]
            atk_pred = y_pred[mask]
            atk_rec = float(recall_score(atk_true, atk_pred, zero_division=0)) if len(atk_true) > 0 else 0.0
            per_attack[str(atk)] = {
                "count": int(mask.sum()),
                "recall": atk_rec,
            }

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": spec,
        "fpr": fpr,
        "fnr": fnr,
        "auroc": auroc,
        "auprc": auprc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "per_attack_recall": per_attack,
    }
