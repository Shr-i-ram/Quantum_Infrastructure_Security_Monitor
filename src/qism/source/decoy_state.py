"""Decoy-state intensity class source.

Three intensity classes: signal, decoy, vacuum, drawn per pulse with
probabilities (p_signal, p_decoy, p_vacuum) that must sum to 1. Each class
carries a mean photon number (mu_signal, mu_decoy, mu_vacuum). Photon numbers
are sampled per pulse from Poisson(mu_i) — never artificially forced to
single-photon behaviour.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from qism.config import SourceConfig
from qism.types import IntensityClass


class DecoyStateSource:
    """Samples the intensity class (and therefore mu) for each pulse."""

    CLASSES: tuple = (IntensityClass.SIGNAL, IntensityClass.DECOY, IntensityClass.VACUUM)

    def __init__(self, config: SourceConfig):
        self.config = config
        self.probabilities = np.array(
            [config.signal_probability, config.decoy_probability, config.vacuum_probability],
            dtype=np.float64,
        )
        self.mus = np.array([config.mu_signal, config.mu_decoy, config.mu_vacuum], dtype=np.float64)

    def sample_intensity_indices(self, rng: Generator, n: int) -> np.ndarray:
        """Sample n intensity class indices: 0=signal, 1=decoy, 2=vacuum."""
        return rng.choice(3, size=n, p=self.probabilities).astype(np.int8)

    def mu_values(self, intensity_indices: np.ndarray) -> np.ndarray:
        """Mean photon number mu for the given class indices."""
        return self.mus[intensity_indices]

    def class_name(self, intensity_indices: np.ndarray) -> np.ndarray:
        """Human-readable class names ('signal'/'decoy'/'vacuum')."""
        names = np.array([cls.value for cls in self.CLASSES], dtype=object)
        return names[intensity_indices]

    @property
    def probabilities_dict(self) -> dict:
        return {cls.value: float(self.probabilities[i]) for i, cls in enumerate(self.CLASSES)}
