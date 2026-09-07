"""Sampling algorithms for QISM population generation.

Supports:
- Standard Monte Carlo pseudo-random sampling
- Latin Hypercube Sampling (LHS) for space-filling multi-dimensional coverage
"""

from __future__ import annotations

from typing import Any
import numpy as np

from qism.population.distributions import CategoricalDistribution, FixedDistribution, SampledParameter


def latin_hypercube_sample(
    parameters: dict[str, SampledParameter],
    n_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Generate n_samples using Latin Hypercube Sampling for continuous parameters.

    For each continuous dimension d, divides [0, 1] into n_samples intervals,
    samples one point uniformly within each interval, and applies a random permutation.
    Points are transformed into physical parameter values using the inverse CDF (ppf).
    Fixed and categorical parameters are evaluated directly.
    """
    if n_samples <= 0:
        return []

    # Separate continuous parameters from discrete/fixed
    continuous_keys: list[str] = []
    discrete_keys: list[str] = []

    for k, p in parameters.items():
        if isinstance(p, (FixedDistribution, CategoricalDistribution)):
            discrete_keys.append(k)
        else:
            continuous_keys.append(k)

    d = len(continuous_keys)
    lhs_matrix = np.zeros((n_samples, d), dtype=np.float64)

    for j in range(d):
        # Sample uniformly within each stratum [i/N, (i+1)/N]
        strata = (np.arange(n_samples) + rng.uniform(0.0, 1.0, size=n_samples)) / n_samples
        rng.shuffle(strata)
        lhs_matrix[:, j] = strata

    samples: list[dict[str, Any]] = []
    for i in range(n_samples):
        row: dict[str, Any] = {}
        for j, k in enumerate(continuous_keys):
            u = lhs_matrix[i, j]
            row[k] = parameters[k].ppf(u)
        for k in discrete_keys:
            row[k] = parameters[k].sample(rng)
        samples.append(row)

    return samples


def monte_carlo_sample(
    parameters: dict[str, SampledParameter],
    n_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Generate n_samples using independent Monte Carlo draws."""
    samples: list[dict[str, Any]] = []
    for _ in range(n_samples):
        row: dict[str, Any] = {}
        for k, p in parameters.items():
            row[k] = p.sample(rng)
        samples.append(row)
    return samples


def sample_parameter_population(
    parameters: dict[str, SampledParameter],
    n_samples: int,
    rng: np.random.Generator,
    strategy: str = "latin_hypercube",
) -> list[dict[str, Any]]:
    """Sample parameter combinations according to the configured sampling strategy."""
    strat = str(strategy).lower()
    if strat in ("latin_hypercube", "lhs"):
        return latin_hypercube_sample(parameters, n_samples, rng)
    return monte_carlo_sample(parameters, n_samples, rng)
