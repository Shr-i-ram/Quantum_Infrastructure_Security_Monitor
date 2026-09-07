"""Detector timing: jitter, gate acceptance, dark-count arrival times.

Model (V0.1):

    * Legitimate photon detection:  t_arrival = t_expected + epsilon,
      epsilon ~ Normal(0, sigma_t^2).
    * The gate is centred on the expected arrival time with width
      gate_width; an event is accepted iff |t_arrival - t_expected| <=
      gate_width / 2.
    * Dark clicks arrive uniformly inside the gate (dark counts are uniform
      in time; the detector only integrates during the gate).

This logic is isolated in its own module so future time-shift and after-gate
attacks can modify timing without rewriting detector code.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator


def sample_signal_arrival_offset(rng: Generator, n: int, sigma_t: float) -> np.ndarray:
    """epsilon ~ Normal(0, sigma_t^2), the timing jitter for signal photons."""
    return rng.normal(0.0, float(sigma_t), n)


def gate_acceptance(offsets: np.ndarray, gate_width_s: float) -> np.ndarray:
    """|t_arrival - t_gate| <= gate_width / 2 (gate centred on expected arrival)."""
    half = float(gate_width_s) / 2.0
    return np.abs(offsets) <= half


def sample_dark_arrival_offset(rng: Generator, n: int, gate_width_s: float) -> np.ndarray:
    """Dark-click arrival offset, uniform within the gate: U(-W/2, W/2)."""
    half = float(gate_width_s) / 2.0
    return rng.uniform(-half, half, n)
