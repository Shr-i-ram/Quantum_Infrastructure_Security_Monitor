"""TEST 11 — configuration validation: invalid parameters must produce useful
validation errors; valid configurations load cleanly."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from helpers import make_config
from qism.config import apply_overrides, default_config, load_config


def test_default_config_loads() -> None:
    cfg = default_config()
    assert cfg.protocol.n_pulses == 100_000
    assert cfg.protocol.repetition_period_s == pytest.approx(1e-6)


def test_probability_above_one_rejected() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        make_config(**{"source.signal_probability": 1.5})


def test_probability_below_zero_rejected() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        make_config(**{"receiver.basis_z_probability": -0.1})


def test_negative_detector_efficiency_rejected() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        make_config(**{"detectors.detector_0.efficiency": -0.05})


def test_negative_distance_rejected() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        make_config(**{"channel.distance_km": -1.0})


def test_intensity_probabilities_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="sum to 1"):
        make_config(**{"source.vacuum_probability": 0.10})  # 0.80+0.15+0.10=1.05


def test_intensity_probabilities_sum_okay() -> None:
    cfg = make_config(**{"source.vacuum_probability": 0.05})
    assert cfg.source.vacuum_probability == pytest.approx(0.05)


def test_negative_jitter_rejected() -> None:
    with pytest.raises(ValidationError):
        make_config(**{"detectors.detector_0.timing_jitter_s": -1e-12})


def test_negative_pulse_count_rejected() -> None:
    with pytest.raises(ValidationError):
        make_config(**{"protocol.n_pulses": 0})


def test_unknown_config_path_rejected() -> None:
    with pytest.raises(KeyError):
        apply_overrides(default_config(), {"channel.not_a_real_param": 1.0})


def test_config_hash_deterministic() -> None:
    a = make_config()
    b = make_config()
    assert a.config_hash() == b.config_hash()
    c = make_config(**{"channel.distance_km": 30.0})
    assert a.config_hash() != c.config_hash()


def test_default_yaml_round_trip(tmp_path) -> None:

    cfg = make_config()
    from qism.config import save_config

    p = tmp_path / "cfg.yaml"
    save_config(cfg, p)
    loaded = load_config(p)
    assert loaded.config_hash() == cfg.config_hash()


def test_vacuum_mu_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        make_config(**{"source.mu_vacuum": -0.1})


def test_future_sections_exist_disabled() -> None:
    """Attack/environment sections exist with disabled defaults (V0.4/V0.3
    ready); V0.2 detector dynamics are enabled by default with the
    rate-dependent dead-time model."""
    cfg = default_config()
    assert cfg.attack.enabled is False
    assert cfg.attack.type == "none"
    assert cfg.environment.type == "stable"
    assert cfg.environment.enabled is False
    # V0.2 defaults: dead time + afterpulsing ON, rate-dependent model
    assert cfg.detectors.dead_time.enabled is True
    assert cfg.detectors.dead_time.model == "rate_dependent"
    assert cfg.detectors.dead_time.tau_dead_s == pytest.approx(50e-9)
    assert cfg.detectors.dead_time.recovery_model == "hard_dead_time"
    assert cfg.detectors.afterpulse.enabled is True
    assert cfg.detectors.afterpulse.probability == pytest.approx(0.01)
    assert cfg.detectors.afterpulse.decay_model == "exponential"
    assert cfg.detectors.state.initial_state == "ready"
    assert cfg.detectors.dead_time_enabled is True  # backwards-compat property
    assert cfg.detectors.afterpulse_enabled is True


def test_legacy_flat_detector_keys_migrate() -> None:
    """V0.1 configs with the flat ``detectors.dead_time_enabled`` etc. keys
    keep loading unchanged (migrated onto the nested V0.2 structure)."""
    from qism.config import QKDConfig

    data = {
        "detectors": {
            "detector_0": {
                "label": "D0",
                "efficiency": 0.15,
                "dark_count_rate_hz": 1000.0,
                "timing_jitter_s": 50e-12,
            },
            "detector_1": {
                "label": "D1",
                "efficiency": 0.15,
                "dark_count_rate_hz": 1000.0,
                "timing_jitter_s": 50e-12,
            },
            "dead_time_enabled": False,
            "dead_time_s": 100.0e-9,
            "afterpulse_enabled": False,
            "p_afterpulse": 0.0,
            "tau_afterpulse_s": 100.0e-9,
        }
    }
    cfg = QKDConfig.model_validate(data)
    assert cfg.detectors.dead_time.enabled is False
    assert cfg.detectors.dead_time.tau_dead_s == pytest.approx(100e-9)
    assert cfg.detectors.dead_time.model == "fixed"  # legacy number keeps fixed semantics
    assert cfg.detectors.afterpulse.enabled is False
    assert cfg.detectors.dynamics_enabled is False  # V0.1 vectorized path


def test_legacy_flat_keys_enable_dynamics() -> None:
    """A V0.1 config with dead time enabled migrates with dynamics active."""
    from qism.config import QKDConfig

    data = {
        "detectors": {
            "detector_0": {
                "label": "D0",
                "efficiency": 0.15,
                "dark_count_rate_hz": 1000.0,
                "timing_jitter_s": 50e-12,
            },
            "detector_1": {
                "label": "D1",
                "efficiency": 0.15,
                "dark_count_rate_hz": 1000.0,
                "timing_jitter_s": 50e-12,
            },
            "dead_time_enabled": True,
            "dead_time_s": 120.0e-9,
            "afterpulse_enabled": True,
            "p_afterpulse": 0.05,
            "tau_afterpulse_s": 90.0e-9,
        }
    }
    cfg = QKDConfig.model_validate(data)
    assert cfg.detectors.dynamics_enabled is True
    assert cfg.detectors.dead_time.tau_dead_s == pytest.approx(120e-9)
    assert cfg.detectors.afterpulse.probability == pytest.approx(0.05)
    assert cfg.detectors.afterpulse.tau_afterpulse_s == pytest.approx(90e-9)
    assert cfg.detectors.dynamics_enabled


def test_detector_dynamics_overrides() -> None:
    cfg = apply_overrides(
        default_config(),
        {
            "detectors.dead_time.enabled": False,
            "detectors.afterpulse.enabled": False,
            "detectors.dead_time.model": "fixed",
            "detectors.dead_time.tau_dead_s": 7e-9,
        },
    )
    assert cfg.detectors.dead_time.enabled is False
    assert cfg.detectors.afterpulse.enabled is False
    assert cfg.detectors.dynamics_enabled is False
    assert cfg.detectors.dead_time.model == "fixed"
    assert cfg.detectors.dead_time.tau_dead_s == pytest.approx(7e-9)
