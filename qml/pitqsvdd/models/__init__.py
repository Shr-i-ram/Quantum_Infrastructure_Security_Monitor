"""Models package for PI-TQSVDD."""

from qml.pitqsvdd.models.context_aware_pitqsvdd import ContextAwarePITQSVDD
from qml.pitqsvdd.models.static_qsvdd import StaticQSVDD
from qml.pitqsvdd.models.temporal_pitqsvdd import TemporalPITQSVDD
from qml.pitqsvdd.models.temporal_predictor import LatentTemporalPredictor

MODEL_REGISTRY = {
    "static_qsvdd": StaticQSVDD,
    "temporal_pitqsvdd": TemporalPITQSVDD,
    "context_aware_pitqsvdd": ContextAwarePITQSVDD,
}


def build_model(model_type: str, circuit, config):
    """Instantiate a model given its type string, circuit, and model configuration."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {list(MODEL_REGISTRY.keys())}")
    cls = MODEL_REGISTRY[model_type]
    return cls(circuit=circuit, config=config)


__all__ = [
    "StaticQSVDD",
    "TemporalPITQSVDD",
    "ContextAwarePITQSVDD",
    "LatentTemporalPredictor",
    "MODEL_REGISTRY",
    "build_model",
]
