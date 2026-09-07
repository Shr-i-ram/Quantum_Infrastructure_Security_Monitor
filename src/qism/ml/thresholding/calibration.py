"""Threshold calibration strictly derived from normal validation scores (Requirements 48–52, 128–130)."""

from __future__ import annotations

from typing import Any

import numpy as np


class ThresholdCalibrator:
    """Calibrates anomaly thresholds on normal validation data without observing test attacks."""

    def __init__(self, target_fpr: float = 0.01):
        self.target_fpr = target_fpr

    def calibrate(
        self,
        normal_val_scores: np.ndarray,
        target_fpr: float | None = None,
    ) -> dict[str, Any]:
        """Calibrate decision threshold for a requested nominal false positive rate.

        Formula:
            tau = Quantile_{1 - target_fpr}(S_normal_val)
        """
        scores = np.asarray(normal_val_scores, dtype=np.float64)
        scores = scores[~np.isnan(scores)]
        if scores.size == 0:
            raise ValueError("Cannot calibrate threshold on empty score array.")

        fpr = target_fpr if target_fpr is not None else self.target_fpr
        quantile = float(np.clip(1.0 - fpr, 0.0, 1.0))
        threshold = float(np.percentile(scores, quantile * 100.0))

        # Calculate empirical validation FPR
        empirical_fpr = float(np.mean(scores > threshold))

        return {
            "threshold": threshold,
            "nominal_fpr": fpr,
            "empirical_val_fpr": empirical_fpr,
            "val_score_min": float(np.min(scores)),
            "val_score_max": float(np.max(scores)),
            "val_score_median": float(np.median(scores)),
            "val_score_std": float(np.std(scores)),
            "n_val_samples": int(scores.size),
        }

    def calibrate_multi_operating_points(
        self,
        normal_val_scores: np.ndarray,
        target_fprs: list[float] | tuple[float, ...] = (0.05, 0.01, 0.005, 0.001),
    ) -> dict[str, dict[str, Any]]:
        """Calibrate multiple thresholds across standard operational false-alarm rates."""
        results = {}
        for fpr in target_fprs:
            results[f"fpr_{fpr}"] = self.calibrate(normal_val_scores, target_fpr=fpr)
        return results
