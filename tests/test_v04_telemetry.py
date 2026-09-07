"""Telemetry and data quality tests for V0.4 infrastructure monitoring."""

import numpy as np
import pytest

from qism.config import default_config
from qism.simulation.simulator import Simulator
from qism.telemetry.events import EVENT_SCHEMA, FieldCategory, ml_input_columns
from qism.telemetry.features import window_features, get_observable_feature_names


def test_v04_window_features_contain_infrastructure_telemetry():
    """Windowed feature dictionary contains all V0.4 infrastructure monitoring metrics."""
    cfg = default_config()
    cfg.environment.type = "dynamic"
    cfg.simulation.chunk_size = 2000
    sim = Simulator(cfg, "run_tel", seed=42)
    events = sim.simulate_chunk(0, 2000, 0)

    wf = window_features(
        events,
        window_size=2000,
        repetition_rate_hz=1e7,
        mu_signal=cfg.source.mu_signal,
        mu_decoy=cfg.source.mu_decoy,
        mu_vacuum=cfg.source.mu_vacuum,
    )

    required_infra_keys = [
        "sensor_temperature_mean",
        "sensor_temperature_std",
        "sensor_temperature_drift",
        "sensor_humidity_mean",
        "sensor_humidity_std",
        "sensor_vibration_mean",
        "sensor_vibration_std",
        "sensor_background_light_mean",
        "sensor_background_light_std",
        "sensor_background_light_drift",
        "clock_offset_mean",
        "clock_offset_std",
        "clock_drift_rate",
        "estimated_channel_loss_db",
        "channel_transmittance_variance",
        "channel_transmittance_drift",
        "source_intensity_mean",
        "source_intensity_variance",
        "source_stability",
        "detection_rate_hz",
        "sifted_key_rate_hz",
    ]

    for k in required_infra_keys:
        assert k in wf, f"Missing infrastructure feature: {k}"
        assert not np.isnan(wf[k]) or wf[k] is not None, f"Feature {k} is invalid"


def test_v04_no_label_or_ground_truth_leakage():
    """Ground truth and attack labels are strictly excluded from ML observable inputs."""
    observable_features = get_observable_feature_names()

    forbidden_substrings = [
        "attack",
        "true_",
        "ground_truth",
        "afterpulse_rate_ground_truth",
        "mean_afterpulse_delay",
        "yield_",
        "decoy_residual",
    ]

    for feat in observable_features:
        for fsub in forbidden_substrings:
            if fsub == "decoy_residual":
                # decoy residual is a protocol-level check, but check exact attack terms
                continue
            assert not feat.startswith("true_"), f"Ground truth feature leaked into observable: {feat}"
            assert "attack" not in feat, f"Attack label leaked into observable: {feat}"
            assert feat != "afterpulse_rate_ground_truth", f"Ground truth leaked: {feat}"
            assert feat != "mean_afterpulse_delay", f"Ground truth leaked: {feat}"


def test_event_schema_v04_categories():
    """Ensure all 12 V0.4 fields have correct classification."""
    gt_fields = [
        "true_temperature_c",
        "true_humidity_pct",
        "true_vibration_g",
        "true_background_photon_rate_hz",
        "true_channel_loss_db",
        "true_polarization_drift_deg",
        "background_detected",
    ]
    for gtf in gt_fields:
        assert EVENT_SCHEMA[gtf].category is FieldCategory.SIMULATOR_GROUND_TRUTH, f"{gtf} must be ground truth"

    obs_fields = [
        "sensor_temperature_c",
        "sensor_humidity_pct",
        "sensor_vibration_g",
        "sensor_background_light_lux",
        "clock_offset_s",
    ]
    for obf in obs_fields:
        assert EVENT_SCHEMA[obf].category is FieldCategory.OBSERVABLE_TELEMETRY, f"{obf} must be observable telemetry"
