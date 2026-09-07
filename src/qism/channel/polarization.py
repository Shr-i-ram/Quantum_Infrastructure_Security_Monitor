"""Polarization misalignment model.

For the simplified V0.1 model:

    theta   (degrees, converted to radians)
    p_mis   = sin^2(theta)

For equal bases, P(correct) = 1 - p_mis and P(error) = p_mis. For different
bases the ideal BB84 measurement outcome is uniformly random (before detector
imperfections). This module is the dedicated measurement abstraction that a
more sophisticated polarization model can replace in later versions.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from qism.config import ChannelConfig


class PolarizationModel:
    """Polarization misalignment probability and detector routing."""

    def __init__(self, config: ChannelConfig):
        self.config = config
        self.theta_rad = float(np.deg2rad(config.misalignment_deg))
        self.p_mis = float(np.sin(self.theta_rad) ** 2)

    def misalignment_probability(self) -> float:
        """p_mis = sin^2(theta)."""
        return self.p_mis

    @staticmethod
    def misalignment_probability_formula(theta_deg: float) -> float:
        """Exact formula: sin^2(deg2rad(theta))."""
        return float(np.sin(np.deg2rad(theta_deg)) ** 2)

    def p_mis_at_angle(self, theta_deg: float) -> float:
        """p_mis = sin^2(deg2rad(theta_deg))."""
        return float(np.sin(np.deg2rad(float(theta_deg))) ** 2)

    def intended_detector(
        self,
        rng: Generator,
        alice_bases: np.ndarray,
        bob_bases: np.ndarray,
        alice_bits: np.ndarray,
        p_mis: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """Ideal BB84 measurement outcome + misalignment: which detector the
        signal photons reach (0 or 1).

        * Same bases: detector = Alice's bit, except with probability p_mis
          the photon lands on the opposite detector.
        * Different bases: the measurement result is random (P(0)=P(1)=0.5).
        """
        alice_bases = np.asarray(alice_bases)
        bob_bases = np.asarray(bob_bases)
        alice_bits = np.asarray(alice_bits)
        n = len(alice_bases)
        same = alice_bases == bob_bases
        prob = self.p_mis if p_mis is None else p_mis
        flip = rng.random(n) < prob
        intended_same = np.where(flip, 1 - alice_bits, alice_bits)
        intended_diff = rng.integers(0, 2, n)
        return np.where(same, intended_same, intended_diff).astype(np.int8)
