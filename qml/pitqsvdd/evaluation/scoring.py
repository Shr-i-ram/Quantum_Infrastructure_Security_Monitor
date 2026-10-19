"""Scoring and evaluation pipeline for PI-TQSVDD."""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torch.nn as nn

from qml.pitqsvdd.data import PITQSVDDDatasetManager
from qml.pitqsvdd.evaluation.metrics import compute_point_metrics
from qml.pitqsvdd.evaluation.thresholding import ThresholdCalibrator


def evaluate_model(
    model: nn.Module,
    dm: PITQSVDDDatasetManager,
    calibrator: ThresholdCalibrator,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> dict[str, Any]:
    """Execute evaluation across validation, test, and generalization partitions.

    1. Calibrate threshold strictly on normal validation partition.
    2. Score test and generalization sets.
    3. Compute point-level metrics, per-attack-family metrics, and score distributions.
    """
    model.eval()

    # 1. Validation scoring and threshold calibration
    val_loader = dm.get_dataloader("val", batch_size=8, shuffle=False)
    val_combined_list = []
    val_labels_list = []
    val_meta = dm.get_raw_metadata("val")

    with torch.no_grad():
        for batch in val_loader:
            windows, _, _, labels = batch
            if hasattr(model, "score") and "alpha" in model.score.__code__.co_varnames:
                s_comb, s_norm, s_temp = model.score(windows, alpha=alpha, beta=beta)
            elif hasattr(model, "score"):
                s_comb = model.score(windows)
            else:
                s_comb = torch.zeros(len(windows))
            val_combined_list.append(s_comb.cpu().numpy())
            val_labels_list.append(labels.cpu().numpy())

    val_combined = np.concatenate(val_combined_list, axis=0) if val_combined_list else np.array([])
    val_labels = np.concatenate(val_labels_list, axis=0) if val_labels_list else np.array([])

    # Isolate normal validation scores for calibration
    val_normal_scores = val_combined[val_labels == 0]
    if len(val_normal_scores) == 0:
        val_normal_scores = val_combined  # Fallback if no 0 labels

    threshold = calibrator.fit(val_normal_scores)
    val_pred = calibrator.apply(val_combined)
    val_atks = [m.get("attack_type", "normal") for m in val_meta]
    val_metrics = compute_point_metrics(val_labels, val_pred, val_combined, val_atks)

    # 2. Test scoring
    def _score_partition(split_name: str) -> dict[str, Any]:
        loader = dm.get_dataloader(split_name, batch_size=8, shuffle=False)
        meta = dm.get_raw_metadata(split_name)
        s_comb_list, s_norm_list, s_temp_list, y_list = [], [], [], []

        with torch.no_grad():
            for batch in loader:
                windows, _, _, labels = batch
                if hasattr(model, "score") and "alpha" in model.score.__code__.co_varnames:
                    s_comb, s_norm, s_temp = model.score(windows, alpha=alpha, beta=beta)
                elif hasattr(model, "score"):
                    s_comb = model.score(windows)
                    s_norm = s_comb
                    s_temp = torch.zeros_like(s_comb)
                else:
                    s_comb = s_norm = s_temp = torch.zeros(len(windows))

                s_comb_list.append(s_comb.cpu().numpy())
                s_norm_list.append(s_norm.cpu().numpy())
                s_temp_list.append(s_temp.cpu().numpy())
                y_list.append(labels.cpu().numpy())

        s_c = np.concatenate(s_comb_list, axis=0) if s_comb_list else np.array([])
        s_n = np.concatenate(s_norm_list, axis=0) if s_norm_list else np.array([])
        s_t = np.concatenate(s_temp_list, axis=0) if s_temp_list else np.array([])
        y = np.concatenate(y_list, axis=0) if y_list else np.array([])

        preds = calibrator.apply(s_c, threshold=threshold)
        atks = [m.get("attack_type", "normal") for m in meta]
        metrics = compute_point_metrics(y, preds, s_c, atks)

        return {
            "metrics": metrics,
            "scores_combined": s_c.tolist(),
            "scores_normal": s_n.tolist(),
            "scores_temporal": s_t.tolist(),
            "predictions": preds.tolist(),
            "labels": y.tolist(),
            "metadata": meta,
        }

    test_res = _score_partition("test")
    gen_res = _score_partition("gen")

    return {
        "calibrated_threshold": threshold,
        "val": {
            "metrics": val_metrics,
            "scores_combined": val_combined.tolist(),
            "predictions": val_pred.tolist(),
            "labels": val_labels.tolist(),
        },
        "test": test_res,
        "generalization": gen_res,
    }
