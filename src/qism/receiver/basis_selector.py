"""Bob's random basis selection: P(Z)=0.5, P(X)=0.5 by default."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from qism.config import ReceiverConfig


class BasisSelector:
    """Samples Bob's measurement bases."""

    def __init__(self, config: ReceiverConfig):
        self.config = config

    def sample(self, rng: Generator, n: int) -> np.ndarray:
        """Sample n Bob bases (0=Z, 1=X)."""
        p_z = self.config.basis_z_probability
        return (rng.random(n) < p_z).astype(np.int8)
