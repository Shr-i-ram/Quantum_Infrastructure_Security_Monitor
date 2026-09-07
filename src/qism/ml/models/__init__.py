"""Model abstractions and registry for QISM anomaly detectors."""

from __future__ import annotations

from qism.ml.models.base import AnomalyDetector
from qism.ml.models.classical import (
    IsolationForestDetector,
    LOFDetector,
    OneClassSVMDetector,
)
from qism.ml.models.neural import AutoencoderDetector
from qism.ml.models.registry import MODEL_REGISTRY, build_model
from qism.ml.models.statistical import MahalanobisDetector, RobustCovarianceDetector
from qism.ml.models.temporal import TemporalAutoencoderDetector

__all__ = [
    "AnomalyDetector",
    "MahalanobisDetector",
    "RobustCovarianceDetector",
    "IsolationForestDetector",
    "OneClassSVMDetector",
    "LOFDetector",
    "AutoencoderDetector",
    "TemporalAutoencoderDetector",
    "MODEL_REGISTRY",
    "build_model",
]
