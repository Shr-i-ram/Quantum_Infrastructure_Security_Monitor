"""TEST 4–5 — channel attenuation and polarization misalignment, plus
propagation delay."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import make_config
from qism.channel.loss import ChannelLoss
from qism.channel.polarization import PolarizationModel
from qism.channel.timing import ChannelTiming
from qism.units import SPEED_OF_LIGHT_M_PER_S


def test_channel_transmittance_exact() -> None:
    """TEST 4: eta_ch = 10^(-(alpha*L + L_sys)/10) exactly."""
    cfg = make_config()
    loss = ChannelLoss(cfg.channel)
    expected = 10.0 ** (-(0.20 * 25.0 + 1.0) / 10.0)
    assert loss.total_loss_db == pytest.approx(6.0)
    assert loss.transmittance() == pytest.approx(expected, rel=1e-12)
    assert loss.transmittance() == pytest.approx(10.0**-0.6, rel=1e-12)


def test_transmittance_formula_matches() -> None:
    cfg = make_config()
    loss = ChannelLoss(cfg.channel)
    for d in (0.0, 5.0, 25.0, 50.0):
        assert loss.transmittance_at(d) == pytest.approx(
            ChannelLoss.transmittance_formula(0.20, d, 1.0), rel=1e-12
        )


def test_transmittance_decreases_with_distance() -> None:
    cfg = make_config()
    loss = ChannelLoss(cfg.channel)
    values = [loss.transmittance_at(d) for d in (0.0, 10.0, 25.0, 50.0, 100.0)]
    assert all(values[i] > values[i + 1] for i in range(len(values) - 1))


def test_misalignment_exact() -> None:
    """TEST 5: p_mis = sin^2(theta)."""
    cfg = make_config()  # theta = 1.0 deg
    pol = PolarizationModel(cfg.channel)
    expected = np.sin(np.deg2rad(1.0)) ** 2
    assert pol.p_mis == pytest.approx(expected, rel=1e-12)
    assert pol.misalignment_probability() == pytest.approx(expected, rel=1e-12)
    assert PolarizationModel.misalignment_probability_formula(1.0) == pytest.approx(
        expected, rel=1e-12
    )


def test_zero_misalignment() -> None:
    cfg = make_config(**{"channel.misalignment_deg": 0.0})
    assert PolarizationModel(cfg.channel).p_mis == pytest.approx(0.0)


def test_channel_delay() -> None:
    """t_channel = n_eff * L / c."""
    cfg = make_config()
    timing = ChannelTiming(cfg.channel)
    expected = 1.468 * 25000.0 / SPEED_OF_LIGHT_M_PER_S
    assert timing.channel_delay_s == pytest.approx(expected, rel=1e-12)
    assert ChannelTiming.channel_delay_formula(
        25000.0, 1.468, SPEED_OF_LIGHT_M_PER_S
    ) == pytest.approx(expected)


def test_expected_arrival() -> None:
    cfg = make_config()
    timing = ChannelTiming(cfg.channel)
    ts = np.array([0.0, 1e-6, 2e-6])
    arr = timing.expected_arrival(ts)
    np.testing.assert_allclose(arr, ts + timing.channel_delay_s)


def test_gate_and_jitter_units() -> None:
    cfg = make_config()
    assert cfg.receiver.gate_width_s == pytest.approx(1e-9)
    assert cfg.detectors.detector_0.timing_jitter_s == pytest.approx(50e-12)
