"""Model registry and factory for anomaly detector baselines (Requirement 13)."""

from __future__ import annotations

from typing import Any

from qism.ml.models.base import AnomalyDetector
from qism.ml.models.classical import (
    IsolationForestDetector,
    LOFDetector,
    OneClassSVMDetector,
)
from qism.ml.models.neural import AutoencoderDetector
from qism.ml.models.statistical import MahalanobisDetector, RobustCovarianceDetector
from qism.ml.models.temporal import TemporalAutoencoderDetector

MODEL_REGISTRY: dict[str, type[AnomalyDetector]] = {
    "mahalanobis": MahalanobisDetector,
    "statistical": MahalanobisDetector,
    "robust_covariance": RobustCovarianceDetector,
    "isolation_forest": IsolationForestDetector,
    "one_class_svm": OneClassSVMDetector,
    "lof": LOFDetector,
    "autoencoder": AutoencoderDetector,
    "temporal_autoencoder": TemporalAutoencoderDetector,
}


def build_model(model_type: str, **kwargs: Any) -> AnomalyDetector:
    """Instantiate an anomaly detector from registry by name."""
    norm_type = model_type.lower().strip()
    if norm_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. Available: {list(MODEL_REGISTRY.keys())}")
    cls = MODEL_REGISTRY[norm_type]
    return cls(**kwargs)
