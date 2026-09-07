"""Centralized deterministic random-number generation.

Design rules:
  * Never use ``random.random()`` in simulation code.
  * Every random draw comes from a ``numpy.random.Generator`` backed by PCG64.
  * A run's RNG is derived deterministically from ``(master_seed, run_id)``.
  * Each chunk uses its own deterministic RNG derived from ``(run_seed, chunk_index)``.
    This makes chunk boundaries crash-safe: after an interrupted run, resuming
    only needs to know *how many chunks were committed* — the exact RNG state
    for the next chunk is recomputed from the seed and chunk index. No RNG
    state desynchronization is possible.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.random import PCG64, Generator

_MASK64 = (1 << 64) - 1
_SPLITMIX_CONSTANT = 0x9E3779B97F4A7C15
_DOMAIN_RUN = 0x6D2B79F5C0DE0001 & _MASK64
_DOMAIN_CHUNK = 0x6D2B79F5C0DE0002 & _MASK64
_DOMAIN_ATTACK = 0x6D2B79F5C0DE0003 & _MASK64
_DOMAIN_ENVIRONMENT = 0x6D2B79F5C0DE0004 & _MASK64
_DOMAIN_SENSOR = 0x6D2B79F5C0DE0005 & _MASK64
_DOMAIN_DATASET = 0x6D2B79F5C0DE0006 & _MASK64
_DOMAIN_SCENARIO = 0x6D2B79F5C0DE0007 & _MASK64
_DOMAIN_POPULATION = 0x6D2B79F5C0DE0008 & _MASK64


def _splitmix64(state: int) -> int:
    """splitmix64 finalizer: a fast, high-quality 64-bit integer mixer."""
    state = (state + _SPLITMIX_CONSTANT) & _MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def _mix_bytes(seed: int, data: bytes, domain: int) -> int:
    h = seed & _MASK64
    for byte in data:
        h = _splitmix64(h ^ byte)
    return _splitmix64(h ^ domain)


def derive_dataset_seed(master_seed: int, dataset_id: str) -> int:
    """Deterministic per-dataset seed from (master_seed, dataset_id)."""
    if not isinstance(master_seed, int) or master_seed < 0:
        raise ValueError(f"master_seed must be a non-negative integer, got {master_seed!r}")
    return _mix_bytes(master_seed, dataset_id.encode("utf-8"), _DOMAIN_DATASET)


def derive_scenario_seed(dataset_seed: int, scenario_id: str) -> int:
    """Deterministic per-scenario seed from (dataset_seed, scenario_id)."""
    if not isinstance(dataset_seed, int) or dataset_seed < 0:
        raise ValueError(f"dataset_seed must be a non-negative integer, got {dataset_seed!r}")
    return _mix_bytes(dataset_seed, scenario_id.encode("utf-8"), _DOMAIN_SCENARIO)


def derive_population_seed(master_seed: int, population_id: str = "default") -> int:
    """Deterministic seed for population parameter sampling."""
    if not isinstance(master_seed, int) or master_seed < 0:
        raise ValueError(f"master_seed must be a non-negative integer, got {master_seed!r}")
    return _mix_bytes(master_seed, population_id.encode("utf-8"), _DOMAIN_POPULATION)


def derive_run_seed(master_seed: int, run_id: str) -> int:
    """Deterministic per-run seed from ``(master_seed, run_id)``.

    Identical inputs always produce identical seeds; the mapping is a
    one-way cryptographic-style mixer (splitmix64), so run ids cannot be
    reverse-engineered from seeds.
    """
    if not isinstance(master_seed, int) or master_seed < 0:
        raise ValueError(f"master_seed must be a non-negative integer, got {master_seed!r}")
    return _mix_bytes(master_seed, run_id.encode("utf-8"), _DOMAIN_RUN)


def derive_chunk_seed(run_seed: int, chunk_index: int) -> int:
    """Deterministic per-chunk seed from ``(run_seed, chunk_index)``."""
    if chunk_index < 0:
        raise ValueError(f"chunk_index must be >= 0, got {chunk_index}")
    return _mix_bytes(run_seed, str(chunk_index).encode("utf-8"), _DOMAIN_CHUNK)


def derive_attack_seed(chunk_seed: int, attack_id: str | int = "attack") -> int:
    """Deterministic domain-separated seed for attack RNG stream."""
    ident = str(attack_id).encode("utf-8")
    return _mix_bytes(chunk_seed, ident, _DOMAIN_ATTACK)


def derive_environment_seed(chunk_seed: int, sub_id: str | int = "environment") -> int:
    """Deterministic domain-separated seed for environmental variation RNG stream."""
    ident = str(sub_id).encode("utf-8")
    return _mix_bytes(chunk_seed, ident, _DOMAIN_ENVIRONMENT)


def derive_sensor_seed(chunk_seed: int, sensor_id: str | int = "sensor") -> int:
    """Deterministic domain-separated seed for sensor measurement noise."""
    ident = str(sensor_id).encode("utf-8")
    return _mix_bytes(chunk_seed, ident, _DOMAIN_SENSOR)


def make_rng(seed: int) -> Generator:
    """Create a PCG64-backed Generator from an integer seed."""
    return np.random.Generator(PCG64(int(seed)))


def make_chunk_rng(run_seed: int, chunk_index: int) -> Generator:
    """RNG for one chunk of a run. Deterministic in (run_seed, chunk_index)."""
    return make_rng(derive_chunk_seed(run_seed, chunk_index))


def make_attack_rng(chunk_seed: int, attack_id: str = "attack") -> Generator:
    """Dedicated RNG for attacks in one chunk, isolated from physical RNG stream."""
    return make_rng(derive_attack_seed(chunk_seed, attack_id))


def make_environment_rng(chunk_seed: int, sub_id: str = "environment") -> Generator:
    """Dedicated RNG for environmental drift and fluctuations in one chunk."""
    return make_rng(derive_environment_seed(chunk_seed, sub_id))


def make_sensor_rng(chunk_seed: int, sensor_id: str = "sensor") -> Generator:
    """Dedicated RNG for sensor noise and measurement dropouts."""
    return make_rng(derive_sensor_seed(chunk_seed, sensor_id))


def serialize_rng_state(rng: Generator) -> dict[str, Any]:
    """Return the full RNG state (BitGenerator + internal state) as a dict."""
    return {
        "bit_generator": "PCG64",
        "state": rng.bit_generator.state,
    }


def deserialize_rng_state(state: Mapping[str, Any]) -> Generator:
    """Restore a Generator from ``serialize_rng_state`` output."""
    name = state.get("bit_generator", "PCG64")
    if name != "PCG64":
        raise ValueError(f"Unsupported BitGenerator in saved state: {name!r}")
    bg = PCG64()
    bg.state = dict(state["state"])
    return Generator(bg)

