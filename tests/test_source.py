"""TEST 1–3 — Poisson photon statistics, basis and bit probabilities,
decoy-state intensity fractions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chisquare

from helpers import make_config
from qism.random import make_rng
from qism.source.alice import POLARIZATION_MAP, Alice
from qism.source.decoy_state import DecoyStateSource
from qism.source.photon_source import PhotonSource

N = 200_000


def test_poisson_photon_statistics() -> None:
    """TEST 1: empirical photon-number frequencies match Poisson(0.5)."""
    mu = 0.5
    rng = make_rng(7)
    samples = PhotonSource.sample_photon_numbers(rng, np.full(N, mu))
    max_bin = 5  # P(6|0.5)*N ~ 2.6 < 5, merge the tail into one bin
    observed, _ = np.histogram(samples, bins=np.arange(-0.5, max_bin + 1.5, 1.0))
    tail = int(np.sum(samples >= max_bin + 1))
    observed = list(observed) + [tail]
    expected = [float(_pmf(k, mu) * N) for k in range(max_bin + 1)] + [
        float((1 - sum(_pmf(k, mu) for k in range(max_bin + 1))) * N)
    ]
    stat, p = chisquare(observed, expected)
    assert p > 0.001, f"chi-square p={p} — photon statistics deviate from Poisson({mu})"


def test_poisson_probability_formulas() -> None:
    """P0/P1/P2/P>=2 formulas match the definitions exactly."""
    mu = 0.5
    p0, p1, p2, p_ge2 = PhotonSource.poisson_probabilities(mu)
    assert p0 == pytest.approx(np.exp(-mu), rel=1e-12)
    assert p1 == pytest.approx(mu * np.exp(-mu), rel=1e-12)
    assert p2 == pytest.approx(0.5 * mu**2 * np.exp(-mu), rel=1e-12)
    assert p_ge2 == pytest.approx(1 - np.exp(-mu) * (1 + mu), rel=1e-12)
    assert p0 + p1 + p_ge2 == pytest.approx(1.0, abs=1e-12)


def test_multiphoton_fraction_present() -> None:
    """The simulator does not artificially force single-photon behaviour."""
    rng = make_rng(11)
    samples = PhotonSource.sample_photon_numbers(rng, np.full(100_000, 0.5))
    frac_multi = np.mean(samples >= 2)
    assert frac_multi > 0.05  # P>=2 for mu=0.5 is ~0.09


def test_basis_probabilities() -> None:
    """TEST 2: P(Z) ~= P(X) ~= 0.5."""
    cfg = make_config()
    alice = Alice(cfg.source)
    rng = make_rng(3)
    bases = alice.sample_bases(rng, N)
    assert np.mean(bases == 0) == pytest.approx(0.5, abs=0.01)


def test_bit_probabilities() -> None:
    """TEST 3: P(0) ~= P(1) ~= 0.5."""
    cfg = make_config()
    alice = Alice(cfg.source)
    rng = make_rng(5)
    bits = alice.sample_bits(rng, N)
    assert np.mean(bits == 0) == pytest.approx(0.5, abs=0.01)


def test_polarization_mapping() -> None:
    assert POLARIZATION_MAP == {(0, 0): "H", (0, 1): "V", (1, 0): "D", (1, 1): "A"}
    assert Alice.polarization_label(0, 0) == "H"
    assert Alice.polarization_label(1, 1) == "A"


def test_intensity_class_fractions() -> None:
    """Signal/decoy/vacuum pulse fractions match the configured probabilities."""
    cfg = make_config()
    decoy = DecoyStateSource(cfg.source)
    rng = make_rng(9)
    idx = decoy.sample_intensity_indices(rng, N)
    frac = np.bincount(idx, minlength=3) / N
    assert frac[0] == pytest.approx(0.80, abs=0.01)
    assert frac[1] == pytest.approx(0.15, abs=0.01)
    assert frac[2] == pytest.approx(0.05, abs=0.01)


def test_mu_values_per_class() -> None:
    cfg = make_config()
    decoy = DecoyStateSource(cfg.source)
    assert decoy.mus.tolist() == [0.5, 0.1, 0.0]
    names = decoy.class_name(np.array([0, 1, 2]))
    assert list(names) == ["signal", "decoy", "vacuum"]


def _pmf(k: int, mu: float) -> float:
    import math

    return math.exp(-mu) * mu**k / math.factorial(k)
