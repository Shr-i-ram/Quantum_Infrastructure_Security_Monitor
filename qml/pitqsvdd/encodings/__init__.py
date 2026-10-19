"""Encoding methods module for PI-TQSVDD."""

from qml.pitqsvdd.encodings.amplitude import AmplitudeEncoding
from qml.pitqsvdd.encodings.angle import AngleEncoding
from qml.pitqsvdd.encodings.base import BaseEncoding
from qml.pitqsvdd.encodings.data_reuploading import DataReuploadingEncoding
from qml.pitqsvdd.encodings.feature_group import FeatureGroupEncoding
from qml.pitqsvdd.encodings.iqp_zz import IQPZZEncoding
from qml.pitqsvdd.encodings.learned_projection import LearnedProjectionEncoding, LearnedProjectionModule

ENCODING_REGISTRY = {
    "angle": AngleEncoding,
    "data_reuploading": DataReuploadingEncoding,
    "amplitude": AmplitudeEncoding,
    "iqp_zz": IQPZZEncoding,
    "feature_group": FeatureGroupEncoding,
    "learned_projection": LearnedProjectionEncoding,
}


import inspect

def get_encoding(name: str, n_qubits: int, **kwargs) -> BaseEncoding:
    """Factory function to instantiate an encoding by name."""
    if name not in ENCODING_REGISTRY:
        raise ValueError(f"Unknown encoding '{name}'. Available: {list(ENCODING_REGISTRY.keys())}")
    cls = ENCODING_REGISTRY[name]
    sig = inspect.signature(cls.__init__)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(n_qubits=n_qubits, **valid_kwargs)


__all__ = [
    "BaseEncoding",
    "AngleEncoding",
    "DataReuploadingEncoding",
    "AmplitudeEncoding",
    "IQPZZEncoding",
    "FeatureGroupEncoding",
    "LearnedProjectionEncoding",
    "LearnedProjectionModule",
    "ENCODING_REGISTRY",
    "get_encoding",
]
