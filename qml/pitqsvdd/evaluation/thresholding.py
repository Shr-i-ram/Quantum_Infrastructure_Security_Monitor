"""Normal-only threshold calibration for PI-TQSVDD."""

from __future__ import annotations

import numpy as np


class ThresholdCalibrator:
    """Calibrates continuous anomaly score thresholds strictly using normal validation data."""

    def __init__(self, percentile: float = 95.0):
        if not (0.0 < percentile < 100.0):
            raise ValueError(f"percentile must be between 0 and 100, got {percentile}")
        self.percentile = percentile
        self.calibrated_threshold: float | None = None

    def fit(self, val_normal_scores: np.ndarray) -> float:
        """Calibrate decision threshold on normal validation scores."""
        scores = np.asarray(val_normal_scores).flatten()
        if len(scores) == 0:
            raise ValueError("Cannot calibrate threshold on empty normal validation scores.")

        self.calibrated_threshold = float(np.percentile(scores, self.percentile))
        return self.calibrated_threshold

    def apply(self, scores: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Apply calibrated or explicit threshold to classify samples as anomaly (1) or normal (0)."""
        th = threshold if threshold is not None else self.calibrated_threshold
        if th is None:
            raise RuntimeError("Threshold has not been calibrated. Call fit() first.")
        return (np.asarray(scores) >= th).astype(int)
