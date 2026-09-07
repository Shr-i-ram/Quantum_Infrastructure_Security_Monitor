"""Base anomaly detector abstraction (Requirements 12–14, 46, 239)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class AnomalyDetector(ABC):
    """Abstract base class for all QISM anomaly detection models."""

    def __init__(self, **kwargs: Any):
        self.hyperparameters: dict[str, Any] = kwargs
        self.threshold: float | None = None
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> AnomalyDetector:
        """Fit detector on normal training observations."""
        pass

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute continuous anomaly scores.

        Contract: HIGHER score must strictly mean MORE ANOMALOUS.
        """
        pass

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Binary anomaly classification: 1 if score > threshold else 0."""
        scores = self.score(X)
        th = threshold if threshold is not None else self.threshold
        if th is None:
            raise ValueError("No threshold provided and model has no calibrated threshold.")
        return (scores > th).astype(int)

    def set_threshold(self, threshold: float) -> None:
        """Assign calibrated decision threshold."""
        self.threshold = float(threshold)

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist model artifacts to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> AnomalyDetector:
        """Load model artifacts from disk."""
        pass
