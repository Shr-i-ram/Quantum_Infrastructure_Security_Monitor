"""TEST 10 — reproducibility: identical config + seed => identical output."""

from __future__ import annotations

import pandas as pd
import pytest

from helpers import make_config, run_events, run_windows
from qism.config import QKDConfig


def test_identical_seed_identical_events() -> None:
    cfg = make_config()
    a = run_events(cfg, 20_000, seed=1234)
    b = run_events(cfg, 20_000, seed=1234)
    pd.testing.assert_frame_equal(a, b)


def test_identical_seed_identical_windows() -> None:
    cfg = make_config()
    a = run_windows(cfg, 30_000, seed=999)
    b = run_windows(cfg, 30_000, seed=999)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ() -> None:
    cfg = make_config()
    a = run_events(cfg, 20_000, seed=1)
    b = run_events(cfg, 20_000, seed=2)
    assert not a.equals(b)
    # statistically similar but not identical
    assert a["alice_bit"].mean() == pytest.approx(b["alice_bit"].mean(), abs=0.02)


def test_run_seed_derivation_deterministic() -> None:
    from qism.random import derive_run_seed, make_chunk_rng

    s1 = derive_run_seed(42, "run_0000")
    s2 = derive_run_seed(42, "run_0000")
    assert s1 == s2
    assert derive_run_seed(42, "run_0001") != s1

    # chunk RNGs are deterministic in (run_seed, chunk_index)
    r1 = make_chunk_rng(s1, 3)
    r2 = make_chunk_rng(s1, 3)
    assert r1.random(100).tolist() == r2.random(100).tolist()
    assert make_chunk_rng(s1, 4).random(10).tolist() != r1.random(10).tolist()


def test_config_hash_stable_across_instances() -> None:
    c1 = make_config()
    c2 = QKDConfig()
    assert c1.config_hash() == c2.config_hash()
