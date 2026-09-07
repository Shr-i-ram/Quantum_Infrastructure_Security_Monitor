"""Reproducibility and domain separation tests for V0.4."""

import numpy as np
import pytest

from qism.config import default_config
from qism.simulation.simulator import Simulator


def test_v04_same_seed_identical_simulation():
    """Identical seed and configuration produce bit-identical events."""
    cfg = default_config()
    cfg.environment.type = "dynamic"
    cfg.environment.temperature.sigma = 0.5
    cfg.simulation.chunk_size = 2000

    sim1 = Simulator(cfg, "run_rep", seed=12345)
    df1 = sim1.simulate_chunk(0, 2000, 0)

    sim2 = Simulator(cfg, "run_rep", seed=12345)
    df2 = sim2.simulate_chunk(0, 2000, 0)

    for col in df1.columns:
        if df1[col].dtype == object or col == "run_id" or str(df1[col].dtype) == "string":
            np.testing.assert_array_equal(df1[col], df2[col], err_msg=f"Mismatch in {col}")
        else:
            np.testing.assert_allclose(df1[col], df2[col], rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=f"Mismatch in {col}")


def test_v04_different_seeds_differ():
    """Different seeds produce different stochastic realizations."""
    cfg = default_config()
    cfg.environment.type = "dynamic"
    cfg.simulation.chunk_size = 2000

    sim1 = Simulator(cfg, "run_rep1", seed=12345)
    df1 = sim1.simulate_chunk(0, 2000, 0)

    sim2 = Simulator(cfg, "run_rep2", seed=54321)
    df2 = sim2.simulate_chunk(0, 2000, 0)

    # Clicks and photon counts should differ
    assert not np.array_equal(df1["detector_0_click"], df2["detector_0_click"])
    assert not np.array_equal(df1["photon_number"], df2["photon_number"])
