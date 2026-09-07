"""Unit tests verifying strict temporal causality of rolling features (Requirements 39, 40, 244)."""

import numpy as np
import pandas as pd
import pytest

from qism.ml.features.extractor import FeatureExtractor


def test_causal_windowing_invariance_to_future_data():
    """Features computed at time t must be invariant to modifications to future values (t+1, ...)."""
    # Original sequence of 5 windows
    df1 = pd.DataFrame({
        "qber": [0.02, 0.03, 0.025, 0.04, 0.05],
        "signal_gain": [0.05, 0.052, 0.048, 0.051, 0.049],
        "run_id": ["run_01"] * 5,
        "window_id": [0, 1, 2, 3, 4],
    })

    # Altered sequence where values at time t >= 3 are drastically changed (e.g. attack injected)
    df2 = pd.DataFrame({
        "qber": [0.02, 0.03, 0.025, 0.25, 0.35],  # drastical shift at t=3, 4
        "signal_gain": [0.05, 0.052, 0.048, 0.010, 0.005],
        "run_id": ["run_01"] * 5,
        "window_id": [0, 1, 2, 3, 4],
    })

    ext = FeatureExtractor(feature_set="minimal", add_temporal_features=True, rolling_window_length=3)
    X1, _ = ext.extract(df1)
    X2, _ = ext.extract(df2)

    # Values at index 0, 1, 2 must be IDENTICAL between X1 and X2
    for col in X1.columns:
        vals1 = X1[col].to_numpy()[:3]
        vals2 = X2[col].to_numpy()[:3]
        assert np.allclose(vals1, vals2, equal_nan=True), f"Causality violation in feature '{col}': past features changed when future was modified!"


def test_causal_windowing_boundary_handling():
    """At sequence start (t < L), rolling statistics use available history without crashing or leaking future."""
    df = pd.DataFrame({
        "qber": [0.02, 0.04],
        "signal_gain": [0.05, 0.05],
        "run_id": ["run_01", "run_01"],
        "window_id": [0, 1],
    })
    ext = FeatureExtractor(feature_set="minimal", add_temporal_features=True, rolling_window_length=4)
    X, _ = ext.extract(df)

    # First observation rolling mean should equal the value itself
    assert np.isclose(X["qber_rolling_mean_4"].iloc[0], 0.02)
    # Second observation rolling mean should equal mean of first two
    assert np.isclose(X["qber_rolling_mean_4"].iloc[1], 0.03)
