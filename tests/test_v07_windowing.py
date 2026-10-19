"""Unit tests for V0.7 temporal windowing and boundary enforcement."""

import numpy as np
import pandas as pd
import pytest

from qism.temporal.windowing import (
    LabelingStrategy,
    TemporalWindowBuilder,
    TemporalWindowConfig,
)


def test_window_builder_basic_shape():
    """Verify window builder constructs expected (N_windows, L, d) tensor."""
    df_X = pd.DataFrame({
        "qber": [0.01 * i for i in range(10)],
        "gain": [0.05 * i for i in range(10)],
    })
    df_y = pd.DataFrame({
        "scenario_id": ["scen_01"] * 10,
        "run_id": ["run_01"] * 10,
        "pulse_id": [i * 2000 for i in range(10)],
        "attack_active": [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    })

    cfg = TemporalWindowConfig(window_length=3, stride=1, causal_padding=True, labeling_strategy=LabelingStrategy.TERMINAL)
    builder = TemporalWindowBuilder(cfg)
    X_win, meta, y_win = builder.build_windows(df_X, df_y)

    assert X_win.shape == (10, 3, 2)
    assert len(meta) == 10
    assert len(y_win) == 10
    # In terminal labeling, window label matches final sample label
    np.testing.assert_array_equal(y_win, df_y["attack_active"].values)


def test_no_crossing_scenario_boundaries():
    """Verify sliding windows never bridge across different scenario or run IDs."""
    df_X = pd.DataFrame({
        "feat1": np.arange(8, dtype=float),
    })
    df_y = pd.DataFrame({
        "scenario_id": ["scen_A"] * 4 + ["scen_B"] * 4,
        "run_id": ["run_0"] * 8,
        "attack_active": [0] * 8,
    })

    cfg = TemporalWindowConfig(window_length=3, stride=1, causal_padding=False)
    builder = TemporalWindowBuilder(cfg)
    X_win, meta, y_win = builder.build_windows(df_X, df_y)

    # Scen A has 4 samples -> with L=3, stride=1, no padding: (4-3+1) = 2 windows
    # Scen B has 4 samples -> (4-3+1) = 2 windows
    # Total windows = 4
    assert X_win.shape == (4, 3, 1)

    scen_ids = [m.scenario_id for m in meta]
    assert scen_ids == ["scen_A", "scen_A", "scen_B", "scen_B"]

    # First window of Scen B should NOT contain values from Scen A
    # Scen B values are 4, 5, 6, 7
    np.testing.assert_array_equal(X_win[2, :, 0], [4.0, 5.0, 6.0])


def test_labeling_strategies():
    """Verify ANY, MAJORITY, and FRACTION labeling strategies."""
    df_X = pd.DataFrame({"feat": np.zeros(5)})
    # Window of 5 steps with attack active at indices 2, 3
    df_y = pd.DataFrame({
        "scenario_id": ["s0"] * 5,
        "run_id": ["r0"] * 5,
        "attack_active": [0, 0, 1, 1, 0],
    })

    # 1. Any strategy
    b_any = TemporalWindowBuilder(TemporalWindowConfig(window_length=5, causal_padding=False, labeling_strategy=LabelingStrategy.ANY))
    _, _, y_any = b_any.build_windows(df_X, df_y)
    assert y_any[0] == 1  # Contains at least one anomaly

    # 2. Majority strategy (fraction is 2/5 = 0.4 < 0.5)
    b_maj = TemporalWindowBuilder(TemporalWindowConfig(window_length=5, causal_padding=False, labeling_strategy=LabelingStrategy.MAJORITY, majority_threshold=0.5))
    _, _, y_maj = b_maj.build_windows(df_X, df_y)
    assert y_maj[0] == 0  # 0.4 < 0.5

    # 3. Terminal strategy
    b_term = TemporalWindowBuilder(TemporalWindowConfig(window_length=5, causal_padding=False, labeling_strategy=LabelingStrategy.TERMINAL))
    _, _, y_term = b_term.build_windows(df_X, df_y)
    assert y_term[0] == 0  # Terminal index (4) is 0
