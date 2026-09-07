"""Alice's source model: random bits and bases.

Alice prepares a weak coherent pulse in polarization state:

    Z0 = H, Z1 = V, X0 = D, X1 = A

with P(A=0)=P(A=1)=0.5 and P(Z)=P(X)=0.5 by default. No full quantum
state-vector simulation is performed in V0.1 — the logical (basis, bit)
description is sufficient for the decoy-state BB84 measurement model.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from qism.config import SourceConfig

# Polarization mapping: (basis, bit) -> polarization label
POLARIZATION_MAP: dict[tuple[int, int], str] = {
    (0, 0): "H",  # Z, 0 -> horizontal
    (0, 1): "V",  # Z, 1 -> vertical
    (1, 0): "D",  # X, 0 -> diagonal
    (1, 1): "A",  # X, 1 -> anti-diagonal
}


class Alice:
    """Generates Alice's random bits and bases."""

    def __init__(self, config: SourceConfig):
        self.config = config

    def sample_bits(self, rng: Generator, n: int) -> np.ndarray:
        """Sample n Alice bits as int8 0/1 with P(0)=bit_probability_zero."""
        p_zero = self.config.bit_probability_zero
        return (rng.random(n) < p_zero).astype(np.int8)

    def sample_bases(self, rng: Generator, n: int) -> np.ndarray:
        """Sample n Alice bases (0=Z, 1=X) with P(Z)=basis_z_probability."""
        p_z = self.config.basis_z_probability
        return (rng.random(n) < p_z).astype(np.int8)

    @staticmethod
    def polarization_label(basis: int, bit: int) -> str:
        """Polarization state prepared by Alice: Z0=H, Z1=V, X0=D, X1=A."""
        return POLARIZATION_MAP[(int(basis), int(bit))]
