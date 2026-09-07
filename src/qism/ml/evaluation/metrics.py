"""Window-level classification metrics and continuous ROC/PR curves (Requirements 65–73, 177, 178)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Calculate window-level confusion matrix and derived rates.

    Formulas:
        Precision   = TP / (TP + FP)
        Recall      = TP / (TP + FN)
        F1          = 2 * (Precision * Recall) / (Precision + Recall)
        FPR         = FP / (FP + TN)
        FNR         = FN / (FN + TP)
        Specificity = TN / (TN + FP)
    """
    yt = np.asarray(y_true, dtype=int).ravel()
    yp = np.asarray(y_pred, dtype=int).ravel()

    # Calculate confusion matrix components
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tp = int(np.sum((yt == 1) & (yp == 1)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
    accuracy = float((tp + tn) / len(yt)) if len(yt) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "specificity": specificity,
        "accuracy": accuracy,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "n_samples": len(yt),
        "n_positives": int(np.sum(yt == 1)),
        "n_negatives": int(np.sum(yt == 0)),
    }


def calculate_continuous_curve_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    """Calculate AUROC, AUPRC, and curve coordinates from continuous anomaly scores."""
    yt = np.asarray(y_true, dtype=int).ravel()
    sc = np.asarray(scores, dtype=np.float64).ravel()

    # Filter out NaNs
    valid = ~np.isnan(sc) & ~np.isnan(yt)
    yt = yt[valid]
    sc = sc[valid]

    n_pos = int(np.sum(yt == 1))
    n_neg = int(np.sum(yt == 0))

    if n_pos == 0 or n_neg == 0:
        # Cannot calculate ROC/PR if only one class exists
        return {
            "auroc": 0.5,
            "auprc": float(n_pos / max(1, len(yt))),
            "roc_curve": {"fpr": [], "tpr": [], "thresholds": []},
            "pr_curve": {"precision": [], "recall": [], "thresholds": []},
        }

    try:
        auroc = float(roc_auc_score(yt, sc))
        fpr_arr, tpr_arr, roc_th = roc_curve(yt, sc)
    except Exception:
        auroc = 0.5
        fpr_arr, tpr_arr, roc_th = np.array([]), np.array([]), np.array([])

    try:
        auprc = float(average_precision_score(yt, sc))
        pr_p, pr_r, pr_th = precision_recall_curve(yt, sc)
    except Exception:
        auprc = float(n_pos / len(yt))
        pr_p, pr_r, pr_th = np.array([]), np.array([]), np.array([])

    return {
        "auroc": auroc,
        "auprc": auprc,
        "roc_curve": {
            "fpr": fpr_arr.tolist() if fpr_arr.size <= 200 else fpr_arr[::len(fpr_arr)//100].tolist(),
            "tpr": tpr_arr.tolist() if tpr_arr.size <= 200 else tpr_arr[::len(tpr_arr)//100].tolist(),
            "thresholds": roc_th.tolist() if roc_th.size <= 200 else roc_th[::len(roc_th)//100].tolist(),
        },
        "pr_curve": {
            "precision": pr_p.tolist() if pr_p.size <= 200 else pr_p[::len(pr_p)//100].tolist(),
            "recall": pr_r.tolist() if pr_r.size <= 200 else pr_r[::len(pr_r)//100].tolist(),
            "thresholds": pr_th.tolist() if pr_th.size <= 200 else pr_th[::len(pr_th)//100].tolist(),
        },
    }
