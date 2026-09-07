"""Shared test helpers: quick configs and in-memory simulation runs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from qism.config import QKDConfig, apply_overrides
from qism.random import derive_run_seed
from qism.simulation.simulator import Simulator
from qism.telemetry.windows import WindowAggregator


def make_config(**overrides: Any) -> QKDConfig:
    """Default (V0.2) config with dotted-path overrides, e.g.
    ``channel.distance_km=10``. V0.2 defaults enable detector dead time and
    afterpulsing."""
    return apply_overrides(QKDConfig(), overrides)


def make_legacy_config(**overrides: Any) -> QKDConfig:
    """V0.1-compatible config: detector dynamics (dead time + afterpulsing)
    disabled. Used by regression tests that must exercise the V0.1 vectorized
    engine and its statistics."""
    return apply_overrides(
        QKDConfig(),
        {"detectors.dead_time.enabled": False, "detectors.afterpulse.enabled": False, **overrides},
    )


def run_events(
    config: QKDConfig,
    n_pulses: int,
    seed: int = 12345,
    run_id: str = "test_run",
) -> pd.DataFrame:
    """Run a chunked simulation in memory and return the event DataFrame."""
    sim = Simulator(config, run_id=run_id, seed=derive_run_seed(seed, run_id))
    chunk_size = min(config.simulation.chunk_size, n_pulses)
    frames = []
    for start in range(0, n_pulses, chunk_size):
        n = min(chunk_size, n_pulses - start)
        chunk_index = start // chunk_size
        frames.append(sim.simulate_chunk(chunk_index, n, start))
    return pd.concat(frames, ignore_index=True)


def run_windows(config: QKDConfig, n_pulses: int, seed: int = 12345) -> pd.DataFrame:
    """Run a simulation and return one telemetry row per window."""
    events = run_events(config, n_pulses, seed)
    agg = WindowAggregator(config)
    rows = []
    for wid, sub in events.groupby("window_id", sort=False):
        rows.append(agg.aggregate(sub))
    return pd.DataFrame(rows)
