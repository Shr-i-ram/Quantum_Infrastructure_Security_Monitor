"""Tests for QISM V0.5 parameter distributions and sampling (Requirements 12–16, 67, 68, 187)."""

import numpy as np
import pytest
from scipy import stats

from qism.config import ParameterDistributionConfig
from qism.population.distributions import (
    BetaDistribution,
    CategoricalDistribution,
    FixedDistribution,
    LogNormalDistribution,
    NormalDistribution,
    SampledParameter,
    TruncatedNormalDistribution,
    UniformDistribution,
    build_distribution,
)
from qism.population.sampling import latin_hypercube_sample, monte_carlo_sample


def test_uniform_distribution():
    dist = UniformDistribution("p_unif", a=10.0, b=30.0)
    rng = np.random.default_rng(42)
    samples = np.array([dist.sample(rng) for _ in range(5000)])

    assert np.all(samples >= 10.0)
    assert np.all(samples <= 30.0)
    assert np.isclose(np.mean(samples), 20.0, atol=0.5)

    # KS test against uniform(10, 20)
    res = stats.ks_1samp(samples, stats.uniform(10.0, 20.0).cdf)
    assert res.pvalue > 0.01


def test_normal_distribution():
    dist = NormalDistribution("p_norm", mean=20.0, std=2.0)
    rng = np.random.default_rng(42)
    samples = np.array([dist.sample(rng) for _ in range(5000)])

    assert np.isclose(np.mean(samples), 20.0, atol=0.2)
    assert np.isclose(np.std(samples), 2.0, atol=0.2)


def test_truncated_normal_distribution():
    dist = TruncatedNormalDistribution("p_trunc", mean=0.20, std=0.03, min_val=0.10, max_val=0.30)
    rng = np.random.default_rng(42)
    samples = np.array([dist.sample(rng) for _ in range(5000)])

    assert np.all(samples >= 0.10)
    assert np.all(samples <= 0.30)
    assert np.isclose(np.mean(samples), 0.20, atol=0.01)


def test_log_normal_distribution():
    dist = LogNormalDistribution("p_lognorm", mean=6.9, std=0.4, min_val=100.0, max_val=5000.0)
    rng = np.random.default_rng(42)
    samples = np.array([dist.sample(rng) for _ in range(5000)])

    assert np.all(samples >= 100.0)
    assert np.all(samples <= 5000.0)
    # Median should be close to exp(6.9) ≈ 992
    assert np.isclose(np.median(samples), np.exp(6.9), rtol=0.15)


def test_beta_distribution():
    dist = BetaDistribution("p_beta", alpha=2.0, beta=5.0, min_val=0.0, max_val=1.0)
    rng = np.random.default_rng(42)
    samples = np.array([dist.sample(rng) for _ in range(5000)])

    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)
    expected_mean = 2.0 / (2.0 + 5.0)  # ~0.2857
    assert np.isclose(np.mean(samples), expected_mean, atol=0.02)


def test_fixed_and_categorical_distributions():
    fixed = FixedDistribution("p_fixed", value=42.5)
    rng = np.random.default_rng(42)
    assert fixed.sample(rng) == 42.5
    assert fixed.ppf(0.5) == 42.5

    cat = CategoricalDistribution("p_cat", choices=["a", "b", "c"], weights=[0.1, 0.7, 0.2])
    cat_samples = [cat.sample(rng) for _ in range(2000)]
    assert set(cat_samples).issubset({"a", "b", "c"})
    b_freq = cat_samples.count("b") / 2000.0
    assert np.isclose(b_freq, 0.7, atol=0.05)


def test_build_distribution_factory():
    cfg = ParameterDistributionConfig(
        distribution="truncated_normal",
        mean=0.15,
        std=0.02,
        min_val=0.05,
        max_val=0.25,
        unit="dimensionless",
        description="Detector efficiency",
    )
    sampled_param = build_distribution("detectors.efficiency", cfg)
    assert sampled_param.name == "detectors.efficiency"
    assert sampled_param.unit == "dimensionless"
    assert isinstance(sampled_param, TruncatedNormalDistribution)


def test_sampling_strategies():
    params = {
        "x": UniformDistribution("x", a=0.0, b=10.0),
        "y": NormalDistribution("y", mean=50.0, std=5.0),
    }
    rng = np.random.default_rng(123)

    # 1. Monte Carlo
    mc_samples = monte_carlo_sample(params, 50, rng)
    assert len(mc_samples) == 50
    for s in mc_samples:
        assert 0.0 <= s["x"] <= 10.0

    # 2. Latin Hypercube
    lhs_samples = latin_hypercube_sample(params, 50, rng)
    assert len(lhs_samples) == 50
    x_vals = [s["x"] for s in lhs_samples]
    # LHS guarantees stratified coverage across the domain
    x_strata = [int(v / 2.0) for v in x_vals]  # 5 bins of width 2
    # Expect each of 5 bins to have at least some samples
    assert len(set(x_strata)) >= 4
