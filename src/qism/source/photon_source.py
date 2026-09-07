"""Weak coherent pulse (WCP) photon-number statistics.

Photon number follows  N_gamma ~ Poisson(mu):

    P(N=n) = exp(-mu) * mu^n / n!

The simulator explicitly samples the photon number for every pulse and never
forces single-photon behaviour. This distribution is the physical basis for
future PNS-attack modelling and decoy-state analysis.

Numerically stable forms:
    P0    = exp(-mu)
    P1    = mu * exp(-mu)
    P>=2  = 1 - exp(-mu) * (1 + mu)      (computed as -expm1(-mu) - P1)
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator


class PhotonSource:
    """Samples Poisson photon numbers for a WCP source."""

    @staticmethod
    def sample_photon_numbers(rng: Generator, mu: np.ndarray) -> np.ndarray:
        """Sample photon numbers: n_i ~ Poisson(mu_i)."""
        return rng.poisson(mu).astype(np.int16)

    @staticmethod
    def apply_intensity_jitter(rng: Generator, mu: np.ndarray, jitter_std: float) -> np.ndarray:
        """Apply pulse-to-pulse Gaussian intensity jitter: mu_i ~ N(mu_0, sigma_mu^2), clamped >= 0."""
        if jitter_std <= 0.0:
            return mu
        noise = rng.normal(0.0, float(jitter_std), size=len(mu))
        return np.maximum(0.0, mu + noise)

    @staticmethod
    def poisson_probabilities(mu: float) -> tuple[float, float, float, float]:
        """Return (P0, P1, P2, P>=2) for a coherent state with mean mu.

        P2 is included for convenience (yield reporting); P>=2 = P2 + P3 + ...
        """
        mu = float(mu)
        p0 = np.exp(-mu)
        p1 = mu * p0
        p2 = 0.5 * mu * mu * p0
        p_ge2 = -np.expm1(-mu) - p1  # stable 1 - exp(-mu)(1+mu)
        return p0, p1, p2, p_ge2
