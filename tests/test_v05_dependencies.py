"""Tests for QISM V0.5 parameter dependencies and physical validation (Requirements 10, 18, 19, 20, 60–62)."""

import numpy as np
import pytest

from qism.config import QKDConfig, default_config
from qism.population.dependencies import resolve_dependencies
from qism.population.validator import validate_scenario_config


def test_channel_dependencies_and_clamping():
    rng = np.random.default_rng(100)
    raw = {
        "channel.distance_km": -5.0,
        "channel.attenuation_db_per_km": 0.01,
        "channel.insertion_loss_db": -0.5,
    }
    resolved, regime = resolve_dependencies(raw, rng)
    assert resolved["channel.distance_km"] >= 1.0
    assert resolved["channel.attenuation_db_per_km"] >= 0.05
    assert resolved["channel.insertion_loss_db"] >= 0.0
    assert regime == "nominal"


def test_detector_pair_correlation():
    rng = np.random.default_rng(101)
    raw = {
        "detectors.efficiency": 0.20,
        "detectors.dark_count_rate_hz": 1000.0,
    }
    resolved, _ = resolve_dependencies(raw, rng)
    assert "detectors.efficiency" not in resolved  # expanded to pair
    eff0 = resolved["detectors.detector_0.efficiency"]
    eff1 = resolved["detectors.detector_1.efficiency"]
    assert 0.15 <= eff0 <= 0.25
    assert 0.15 <= eff1 <= 0.25
    # D0 and D1 should have small fabrication mismatch, not identical or unrelated
    assert abs(eff0 - eff1) < 0.02

    dark0 = resolved["detectors.detector_0.dark_count_rate_hz"]
    dark1 = resolved["detectors.detector_1.dark_count_rate_hz"]
    assert abs(dark0 - dark1) < 350.0


def test_decoy_hierarchy_enforcement():
    rng = np.random.default_rng(102)
    raw = {
        "source.mu_signal": 0.40,
        "source.mu_decoy": 0.50,  # Inversion violation!
    }
    resolved, _ = resolve_dependencies(raw, rng)
    # Must enforce mu_signal > mu_decoy > mu_vacuum == 0
    assert resolved["source.mu_signal"] > resolved["source.mu_decoy"]
    assert resolved["source.mu_vacuum"] == 0.0


def test_edge_case_regimes():
    rng = np.random.default_rng(103)
    raw = {"channel.distance_km": 25.0, "source.mu_signal": 0.50}

    # Low signal regime
    resolved_ls, reg_ls = resolve_dependencies(raw, rng, is_edge_case=True, edge_case_type="low_signal")
    assert reg_ls == "low_signal"
    assert resolved_ls["source.mu_signal"] < 0.30

    # High loss regime
    resolved_hl, reg_hl = resolve_dependencies(raw, rng, is_edge_case=True, edge_case_type="high_loss")
    assert reg_hl == "high_loss"
    assert resolved_hl["channel.distance_km"] >= 45.0

    # High noise regime
    resolved_hn, reg_hn = resolve_dependencies(raw, rng, is_edge_case=True, edge_case_type="high_noise")
    assert reg_hn == "high_noise"
    assert resolved_hn["background.rate_hz"] >= 5000.0


def test_physical_validator_accepts_valid():
    cfg = default_config()
    errors = validate_scenario_config(cfg)
    assert errors == []


def test_physical_validator_rejects_unphysical():
    cfg = default_config()
    # 1. Impossible efficiency > 1.0
    cfg.detectors.detector_0.efficiency = 1.25
    errors = validate_scenario_config(cfg)
    assert any("efficiency" in e for e in errors)

    # 2. Negative rates
    cfg = default_config()
    cfg.detectors.detector_0.dark_count_rate_hz = -50.0
    errors = validate_scenario_config(cfg)
    assert any("dark count rate" in e for e in errors)

    # 3. Decoy inversion
    cfg = default_config()
    cfg.source.mu_signal = 0.2
    cfg.source.mu_decoy = 0.4
    errors = validate_scenario_config(cfg)
    assert any("mu_signal" in e for e in errors)
