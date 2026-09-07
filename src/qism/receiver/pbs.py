"""Polarizing beam splitter (PBS) routing.

The PBS routes incoming photons to one of Bob's two detectors according to the
ideal BB84 measurement outcome, perturbed by polarization misalignment:

    same basis, correct  -> detector = Alice's bit          (prob 1 - p_mis)
    same basis, error    -> opposite detector               (prob p_mis)
    different bases      -> uniformly random detector

This is a dedicated measurement abstraction: a full polarization-state model
can replace it in later versions without touching the rest of Bob.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator


class PolarizingBeamSplitter:
    """Determines which detector receives the signal photons of each pulse."""

    def __init__(self, misalignment_probability: float):
        self.p_mis = float(misalignment_probability)

    def intended_detector(
        self, rng: Generator, alice_bases: np.ndarray, bob_bases: np.ndarray, alice_bits: np.ndarray
    ) -> np.ndarray:
        """Ideal measurement outcome + misalignment routing (detector 0 or 1)."""
        alice_bases = np.asarray(alice_bases)
        bob_bases = np.asarray(bob_bases)
        alice_bits = np.asarray(alice_bits)
        n = len(alice_bases)
        same = alice_bases == bob_bases
        flip = rng.random(n) < self.p_mis
        intended_same = np.where(flip, 1 - alice_bits, alice_bits)
        intended_diff = rng.integers(0, 2, n)
        return np.where(same, intended_same, intended_diff).astype(np.int8)
