"""Statistical and distance-based baseline anomaly detectors (Requirements 8, 131, 132, 264)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance, MinCovDet

from qism.ml.models.base import AnomalyDetector


class MahalanobisDetector(AnomalyDetector):
    """Mahalanobis distance anomaly detector with ridge regularization.

    Distance score:
        D_M(x) = sqrt( (x - mu)^T * (Sigma + lambda*I)^(-1) * (x - mu) )
    """

    def __init__(self, regularization: float = 1e-4, **kwargs: Any):
        super().__init__(regularization=regularization, **kwargs)
        self.regularization = regularization
        self.mean_: np.ndarray | None = None
        self.inv_cov_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> MahalanobisDetector:
        X_arr = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X_arr, axis=0)
        cov = np.cov(X_arr, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        elif cov.ndim == 1:
            cov = np.diag(cov)

        # Ridge regularization to guarantee invertibility
        reg_cov = cov + self.regularization * np.eye(cov.shape[0])
        self.inv_cov_ = np.linalg.pinv(reg_cov)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.mean_ is None or self.inv_cov_ is None:
            raise RuntimeError("Model must be fit before score.")
        X_arr = np.asarray(X, dtype=np.float64)
        diff = X_arr - self.mean_
        # Vectorized Mahalanobis distance: sqrt(sum(diff * (diff @ inv_cov), axis=1))
        scores = np.sqrt(np.maximum(0.0, np.sum(diff * (diff @ self.inv_cov_), axis=1)))
        return scores

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "regularization": self.regularization,
            "threshold": self.threshold,
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "inv_cov": self.inv_cov_.tolist() if self.inv_cov_ is not None else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> MahalanobisDetector:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        detector = cls(regularization=data["regularization"])
        detector.threshold = data["threshold"]
        detector.mean_ = np.array(data["mean"], dtype=np.float64) if data["mean"] is not None else None
        detector.inv_cov_ = np.array(data["inv_cov"], dtype=np.float64) if data["inv_cov"] is not None else None
        detector.is_fitted = True
        return detector


class RobustCovarianceDetector(AnomalyDetector):
    """Robust covariance / Minimum Covariance Determinant (MCD) anomaly detector."""

    def __init__(self, support_fraction: float | None = None, **kwargs: Any):
        super().__init__(support_fraction=support_fraction, **kwargs)
        self.support_fraction = support_fraction
        self.model_: MinCovDet | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> RobustCovarianceDetector:
        X_arr = np.asarray(X, dtype=np.float64)
        try:
            mcd = MinCovDet(support_fraction=self.support_fraction, random_state=42)
            mcd.fit(X_arr)
            self.model_ = mcd
        except Exception:
            # Fallback to empirical covariance if sample count is low
            emp = EmpiricalCovariance()
            emp.fit(X_arr)
            self.model_ = emp
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model_ is None:
            raise RuntimeError("Model must be fit before score.")
        X_arr = np.asarray(X, dtype=np.float64)
        # mahalanobis distance returned by scikit-learn
        d2 = self.model_.mahalanobis(X_arr)
        return np.sqrt(np.maximum(0.0, d2))

    def save(self, path: str | Path) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "support_fraction": self.support_fraction,
            "threshold": self.threshold,
            "model": self.model_,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> RobustCovarianceDetector:
        import joblib
        path = Path(path)
        payload = joblib.load(path)
        det = cls(support_fraction=payload["support_fraction"])
        det.threshold = payload["threshold"]
        det.model_ = payload["model"]
        det.is_fitted = True
        return det
