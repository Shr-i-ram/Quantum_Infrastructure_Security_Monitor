"""Event schema, window features, and physics-informed extraction."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import make_config, run_events, run_windows
from qism.telemetry.events import EVENT_SCHEMA, ml_input_columns
from qism.types import FieldCategory


def test_event_schema_columns() -> None:
    ev = run_events(make_config(), 20_000, seed=10)
    assert list(ev.columns) == list(EVENT_SCHEMA.keys())
    assert len(ev) == 20_000


def test_schema_field_categories() -> None:
    cats = {spec.category for spec in EVENT_SCHEMA.values()}
    assert FieldCategory.OBSERVABLE_TELEMETRY in cats
    assert FieldCategory.DERIVED_TELEMETRY in cats
    assert FieldCategory.SIMULATOR_GROUND_TRUTH in cats
    assert FieldCategory.FUTURE_ONLY in cats


def test_ml_input_excludes_ground_truth() -> None:
    cols = ml_input_columns()
    for forbidden in (
        "photon_number",
        "attack_type",
        "attack_strength",
        "attack_active",
        "channel_transmittance",
        "attack_id",
    ):
        assert forbidden not in cols, f"{forbidden} must never enter ML inputs"
    assert "alice_bit" in cols
    assert "timing_offset_s" in cols


def test_sifting_and_bit_error_semantics() -> None:
    ev = run_events(make_config(), 50_000, seed=11)
    sifted = ev["alice_basis"].to_numpy() == ev["bob_basis"].to_numpy()
    assert np.array_equal(ev["sifted"].to_numpy(), sifted)
    single = ev["detector_0_click"].to_numpy() != ev["detector_1_click"].to_numpy()
    be = ev["bit_error"].to_numpy(dtype=np.float64)
    # bit_error defined exactly on sifted single-click events
    defined = ~np.isnan(be)
    assert np.array_equal(defined, sifted & single)
    # values are 0/1 where defined
    assert set(np.unique(be[defined])) <= {0.0, 1.0}
    # double clicks excluded (V0.1 policy): bit_error NaN on doubles
    doubles = ev["double_click"].to_numpy()
    assert not np.any(defined & doubles)


def test_window_features_qber_reconstruction() -> None:
    cfg = make_config()
    ev = run_events(cfg, 50_000, seed=12)
    from qism.telemetry.features import window_features

    feats = window_features(
        ev,
        window_size=cfg.telemetry.window_size,
        max_photon_number=cfg.source.max_photon_number,
        mu_signal=cfg.source.mu_signal,
        mu_decoy=cfg.source.mu_decoy,
        mu_vacuum=cfg.source.mu_vacuum,
    )
    sifted = ev["alice_basis"].to_numpy() == ev["bob_basis"].to_numpy()
    single = ev["detector_0_click"].to_numpy() != ev["detector_1_click"].to_numpy()
    key = sifted & single
    n_s = int(key.sum())
    n_e = int(np.sum(ev["bit_error"].to_numpy(dtype=np.float64) == 1.0))
    assert feats["sifted_count"] == n_s
    assert feats["error_count"] == n_e
    assert feats["qber"] == pytest.approx(n_e / n_s)


def test_window_gains_reconstruction() -> None:
    cfg = make_config()
    ev = run_events(cfg, 50_000, seed=13)
    from qism.telemetry.features import window_features

    feats = window_features(
        ev,
        window_size=cfg.telemetry.window_size,
        max_photon_number=cfg.source.max_photon_number,
        mu_signal=cfg.source.mu_signal,
        mu_decoy=cfg.source.mu_decoy,
        mu_vacuum=cfg.source.mu_vacuum,
    )
    clicked = ev["detector_0_click"].to_numpy() | ev["detector_1_click"].to_numpy()
    cls = ev["intensity_class"].to_numpy()
    for name in ("signal", "decoy", "vacuum"):
        n_sent = int(np.sum(cls == name))
        n_click = int(np.sum((cls == name) & clicked))
        assert feats[f"{name}_gain"] == pytest.approx(n_click / n_sent)


def test_imbalance_nan_when_no_counts() -> None:
    from qism.telemetry.features import _safe_ratio

    assert np.isnan(_safe_ratio(1, 0))
    assert not np.isnan(_safe_ratio(3, 3))


def test_telemetry_rows_have_all_features() -> None:
    cfg = make_config()
    df = run_windows(cfg, 50_000, seed=14)
    assert len(df) == 5  # 50k pulses / 10k windows
    required = [
        "qber",
        "qber_z",
        "qber_x",
        "sifted_count",
        "sifted_rate",
        "error_count",
        "signal_gain",
        "decoy_gain",
        "vacuum_gain",
        "signal_decoy_residual",
        "decoy_decoy_residual",
        "vacuum_residual",
        "detector_0_count",
        "detector_1_count",
        "total_count",
        "detector_imbalance",
        "double_click_rate",
        "dark_count_rate",
        "mean_arrival_offset",
        "std_arrival_offset",
        "timing_skewness",
        "timing_kurtosis",
        "mean_photon_number",
        "photon_variance",
        "vacuum_fraction",
        "single_photon_fraction",
        "multiphoton_fraction",
        "mean_channel_transmittance",
    ]
    for col in required:
        assert col in df.columns, f"missing feature: {col}"
    assert df["window_id"].tolist() == [0, 1, 2, 3, 4]


def test_timing_histogram_length() -> None:
    cfg = make_config(**{"telemetry.timing_histogram_bins": 20})
    df = run_windows(cfg, 30_000, seed=15)
    assert len(df["timing_histogram_counts"].iloc[0]) == 20
    assert len(df["timing_histogram_edges"].iloc[0]) == 21


def test_yields_present() -> None:
    cfg = make_config()
    df = run_windows(cfg, 20_000, seed=16)
    for n in range(0, cfg.source.max_photon_number + 1):
        assert f"yield_{n}" in df.columns


def test_excess_kurtosis_convention() -> None:
    """Gaussian data has excess kurtosis ~ 0 (Fisher convention)."""
    cfg = make_config()
    df = run_windows(cfg, 30_000, seed=17)
    assert df["timing_kurtosis"].abs().max() < 0.5


def test_event_level_fields_observable() -> None:
    ev = run_events(make_config(), 5_000, seed=18)
    assert (ev["attack_active"] == False).all()  # noqa: E712
    assert (ev["attack_type"] == "none").all()
    assert (ev["environment_state"] == "stable").all()
    assert (ev["detector_state"] == "ready").all()
