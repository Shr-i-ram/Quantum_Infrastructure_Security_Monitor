"""Unit tests for V0.7 temporal feature engineering and trajectory isolation."""

import numpy as np
import pandas as pd
import pytest

from qism.temporal.features import TemporalFeatureConfig, TemporalFeatureEngine


def test_temporal_feature_generation():
    """Verify differences, rolling statistics, EWMA, and rate of change."""
    df = pd.DataFrame({
        "scenario_id": ["s1"] * 10,
        "run_id": ["r1"] * 10,
        "qber": [0.01 * (i + 1) for i in range(10)],
        "gain": [1.0] * 10,
    })

    cfg = TemporalFeatureConfig(
        rolling_windows=[3],
        compute_diff1=True,
        compute_diff2=True,
        compute_rolling_mean=True,
        compute_rolling_std=True,
        compute_ewma=True,
        compute_rate_of_change=True,
        compute_physical_couplings=False,
    )
    engine = TemporalFeatureEngine(cfg)
    df_feat = engine.fit_transform(df)

    assert "qber_diff_1" in df_feat.columns
    assert "qber_diff_2" in df_feat.columns
    assert "qber_rolling_mean_3" in df_feat.columns
    assert "qber_rolling_std_3" in df_feat.columns
    assert "qber_ewma_0.30" in df_feat.columns
    assert "qber_roc_1" in df_feat.columns

    # First difference of linear ramp [0.01, 0.02, 0.03, ...] is 0.01 (index 0 is 0.0)
    assert np.isclose(df_feat["qber_diff_1"].iloc[1], 0.01)
    assert np.isclose(df_feat["qber_diff_1"].iloc[5], 0.01)

    # Second difference of linear ramp should be 0.0
    assert np.isclose(df_feat["qber_diff_2"].iloc[2], 0.0)


def test_features_no_cross_scenario_leakage():
    """Verify rolling statistics reset at scenario boundaries."""
    df = pd.DataFrame({
        "scenario_id": ["scen_A"] * 5 + ["scen_B"] * 5,
        "run_id": ["run_0"] * 10,
        "signal": [10.0] * 5 + [100.0] * 5,
    })

    cfg = TemporalFeatureConfig(
        rolling_windows=[3],
        compute_diff1=True,
        compute_diff2=False,
        compute_rolling_mean=True,
        compute_rolling_std=False,
        compute_ewma=False,
        compute_rate_of_change=False,
        compute_physical_couplings=False,
    )
    engine = TemporalFeatureEngine(cfg)
    df_feat = engine.fit_transform(df)

    # At boundary (index 5, which is first item of scen_B), diff1 must be 0.0, NOT 100 - 10 = 90
    assert df_feat["signal_diff_1"].iloc[5] == 0.0

    # At boundary index 5, rolling mean must be 100.0, NOT (10 + 10 + 100)/3
    assert df_feat["signal_rolling_mean_3"].iloc[5] == 100.0

