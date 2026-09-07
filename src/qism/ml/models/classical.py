"""Classical machine learning baseline anomaly detectors (Requirements 8, 14, 46, 129, 130)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from qism.ml.models.base import AnomalyDetector


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest anomaly detector (sklearn.ensemble.IsolationForest).

    Score orientation: S(x) = -decision_function(x).
    Higher score indicates higher anomaly confidence.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        random_state: int = 42,
        **kwargs: Any,
    ):
        super().__init__(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            **kwargs,
        )
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

        valid_keys = {"max_samples", "max_features", "bootstrap", "verbose", "warm_start"}
        sk_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=1,
            **sk_kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> IsolationForestDetector:
        X_arr = np.asarray(X, dtype=np.float64)
        self.model.fit(X_arr)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before score.")
        X_arr = np.asarray(X, dtype=np.float64)
        # decision_function returns negative values for anomalies, so invert
        return -self.model.decision_function(X_arr)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hyperparameters": self.hyperparameters,
            "threshold": self.threshold,
            "model": self.model,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> IsolationForestDetector:
        path = Path(path)
        payload = joblib.load(path)
        det = cls(**payload["hyperparameters"])
        det.threshold = payload["threshold"]
        det.model = payload["model"]
        det.is_fitted = True
        return det


class OneClassSVMDetector(AnomalyDetector):
    """One-Class Support Vector Machine (sklearn.svm.OneClassSVM).

    Score orientation: S(x) = -decision_function(x).
    Higher score indicates higher anomaly confidence.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        nu: float = 0.05,
        **kwargs: Any,
    ):
        super().__init__(kernel=kernel, gamma=gamma, nu=nu, **kwargs)
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        valid_keys = {"tol", "shrinking", "cache_size", "verbose", "max_iter"}
        sk_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        self.model = OneClassSVM(
            kernel=self.kernel,
            gamma=self.gamma,
            nu=self.nu,
            **sk_kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> OneClassSVMDetector:
        X_arr = np.asarray(X, dtype=np.float64)
        self.model.fit(X_arr)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before score.")
        X_arr = np.asarray(X, dtype=np.float64)
        return -self.model.decision_function(X_arr)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hyperparameters": self.hyperparameters,
            "threshold": self.threshold,
            "model": self.model,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> OneClassSVMDetector:
        path = Path(path)
        payload = joblib.load(path)
        det = cls(**payload["hyperparameters"])
        det.threshold = payload["threshold"]
        det.model = payload["model"]
        det.is_fitted = True
        return det


class LOFDetector(AnomalyDetector):
    """Local Outlier Factor configured for novelty detection (sklearn.neighbors.LocalOutlierFactor).

    Score orientation: S(x) = -decision_function(x).
    Higher score indicates higher anomaly confidence.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float | str = "auto",
        **kwargs: Any,
    ):
        super().__init__(n_neighbors=n_neighbors, contamination=contamination, **kwargs)
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        valid_keys = {"algorithm", "leaf_size", "metric", "p", "metric_params"}
        sk_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
            **sk_kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> LOFDetector:
        X_arr = np.asarray(X, dtype=np.float64)
        # Ensure n_neighbors <= n_samples
        n_samples = len(X_arr)
        effective_k = min(self.n_neighbors, max(1, n_samples - 1))
        if effective_k != self.model.n_neighbors:
            self.model.n_neighbors = effective_k
        self.model.fit(X_arr)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before score.")
        X_arr = np.asarray(X, dtype=np.float64)
        return -self.model.decision_function(X_arr)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hyperparameters": self.hyperparameters,
            "threshold": self.threshold,
            "model": self.model,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> LOFDetector:
        path = Path(path)
        payload = joblib.load(path)
        det = cls(**payload["hyperparameters"])
        det.threshold = payload["threshold"]
        det.model = payload["model"]
        det.is_fitted = True
        return det
