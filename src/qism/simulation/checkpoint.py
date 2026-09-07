"""Checkpointing and crash recovery.

Checkpoints are written atomically (``file.tmp`` -> fsync -> rename), and the
previous valid checkpoint is preserved as ``checkpoint.json.bak`` /
``rng_state.npz.bak`` so a corrupt checkpoint falls back to the previous valid
one.

Recovery invariants:
  * The authoritative record of completed pulses is the set of committed event
    chunk files on disk (the checkpoint is advisory). Chunk RNGs are derived
    deterministically from (run_seed, chunk_index), so resuming only needs to
    know *which* chunks are committed — the exact RNG state is recomputed and
    no event duplication or loss is possible.
  * ``validate()`` verifies simulator version, configuration hash, run id and
    seed before a resume is allowed.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from datetime import datetime, timezone
from typing import Any

import numpy as np

from qism.random import deserialize_rng_state, make_rng, serialize_rng_state
from qism.types import RunStatus

log = logging.getLogger("qism.checkpoint")

CHECKPOINT_FORMAT_VERSION = 2

#: fields of the checkpoint payload that describe simulator physical state
#: (added in V0.2; V0.1 checkpoints had no runtime state and are rejected by
#: the simulator_version check anyway).


def _fsync_file(path: pathlib.Path) -> None:
    # Windows requires the handle opened for writing before fsync is allowed.
    with open(path, "r+b") as fh:
        os.fsync(fh.fileno())


def atomic_write_text(path: pathlib.Path, text: str) -> None:
    """Write text to ``path`` atomically (tmp + fsync + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    _fsync_file(tmp)
    os.replace(tmp, path)


def atomic_write_npz(path: pathlib.Path, payload: dict[str, Any]) -> None:
    # Note: np.savez appends '.npz' to its filename, so the tmp name must end
    # in '.npz' for the file to be created where we expect it.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **payload)
    _fsync_file(tmp)
    os.replace(tmp, path)


class CheckpointManager:
    """Saves/loads/validates per-run checkpoints."""

    def __init__(
        self,
        checkpoint_dir: pathlib.Path,
        *,
        config_hash: str,
        simulator_version: str,
        run_id: str,
        seed: int,
        master_seed: int,
        scenario_id: str,
        family: str,
        run_dir: pathlib.Path,
        total_pulses: int,
        window_size: int,
        output_root: str,
        checkpoint_root: str,
    ):
        self.dir = pathlib.Path(checkpoint_dir)
        self.config_hash = config_hash
        self.simulator_version = simulator_version
        self.run_id = run_id
        self.seed = seed
        self.master_seed = master_seed
        self.scenario_id = scenario_id
        self.family = family
        self.run_dir = str(run_dir)
        self.total_pulses = total_pulses
        self.window_size = window_size
        self.output_root = output_root
        self.checkpoint_root = checkpoint_root

    # ------------------------------------------------------------------ #
    @property
    def json_path(self) -> pathlib.Path:
        return self.dir / "checkpoint.json"

    @property
    def rng_path(self) -> pathlib.Path:
        return self.dir / "rng_state.npz"

    def exists(self) -> bool:
        return self.json_path.exists()

    # ------------------------------------------------------------------ #
    def save(
        self,
        *,
        pulses_committed: int,
        chunk_index: int,
        run_rng,
        status: RunStatus,
        runtime_state: dict[str, Any] | None = None,
    ) -> None:
        """Atomically persist a checkpoint.

        ``runtime_state`` is the simulator's full physical state (detector
        state machines, pending afterpulse queues, rate-estimation window) at
        the ``chunk_index`` grid boundary — i.e. the entering state of chunk
        ``chunk_index``. ``chunk_index`` counts fully simulated chunks whose
        events are committed on disk.
        """
        # preserve the previous valid checkpoint as .bak
        if self.json_path.exists():
            os.replace(self.json_path, self.json_path.with_name("checkpoint.json.bak"))
        if self.rng_path.exists():
            os.replace(self.rng_path, self.rng_path.with_name("rng_state.bak.npz"))

        payload: dict[str, Any] = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "simulator_version": self.simulator_version,
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "family": self.family,
            "master_seed": self.master_seed,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "output_dir": self.run_dir,
            "checkpoint_dir": str(self.dir),
            "output_root": self.output_root,
            "checkpoint_root": self.checkpoint_root,
            "total_pulses": self.total_pulses,
            "window_size": self.window_size,
            "pulses_committed": int(pulses_committed),
            "chunk_index": int(chunk_index),
            "status": status.value,
            "runtime_state": runtime_state,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_text(self.json_path, json.dumps(payload, indent=2, default=float))
        atomic_write_npz(
            self.rng_path, {"rng_state": np.asarray(serialize_rng_state(run_rng), dtype=object)}
        )
        log.info(
            "checkpoint saved: %d/%d pulses committed (status=%s)",
            pulses_committed,
            self.total_pulses,
            status.value,
        )

    # ------------------------------------------------------------------ #
    def load(self) -> dict[str, Any] | None:
        """Load the latest valid checkpoint (falls back to .bak)."""
        if not self.exists():
            return None
        payload = self._read_json(self.json_path)
        if payload is None and self.json_path.with_name("checkpoint.json.bak").exists():
            log.warning("checkpoint.json is corrupt; falling back to checkpoint.json.bak")
            payload = self._read_json(self.json_path.with_name("checkpoint.json.bak"))
        return payload

    @staticmethod
    def _read_json(path: pathlib.Path) -> dict[str, Any] | None:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return None
            return data
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            log.error("failed to read checkpoint %s: %s", path, exc)
            return None

    def load_rng(self, payload: dict[str, Any]):
        """Restore the run-level RNG from the checkpoint (fresh if absent)."""
        candidates = [self.rng_path, self.rng_path.with_name("rng_state.bak.npz")]
        for path in candidates:
            if not path.exists():
                continue
            try:
                with np.load(path, allow_pickle=True) as data:
                    state = dict(data["rng_state"].item())
                return deserialize_rng_state(state)
            except (OSError, KeyError, ValueError, TypeError) as exc:
                log.warning(
                    "could not read RNG state from %s (%s); trying next candidate", path, exc
                )
        log.warning("no usable RNG state found; starting fresh run RNG")
        return make_rng(self.seed)

    # ------------------------------------------------------------------ #
    def validate(self, payload: dict[str, Any]) -> list[str]:
        """Return a list of validation issues (empty means valid to resume)."""
        issues: list[str] = []
        if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            issues.append(
                f"checkpoint format_version {payload.get('format_version')!r} != expected {CHECKPOINT_FORMAT_VERSION}"
            )
        if payload.get("simulator_version") != self.simulator_version:
            issues.append(
                f"checkpoint simulator_version {payload.get('simulator_version')!r} != current {self.simulator_version}"
            )
        if payload.get("run_id") != self.run_id:
            issues.append(
                f"checkpoint run_id {payload.get('run_id')!r} != requested {self.run_id!r}"
            )
        if payload.get("seed") != self.seed:
            issues.append(f"checkpoint seed {payload.get('seed')!r} != derived seed {self.seed}")
        if payload.get("master_seed") != self.master_seed:
            issues.append(
                f"checkpoint master_seed {payload.get('master_seed')!r} != configured {self.master_seed}"
            )
        if payload.get("config_hash") != self.config_hash:
            issues.append("checkpoint config_hash does not match the current configuration")
        old_total = payload.get("total_pulses")
        if old_total != self.total_pulses:
            if self.total_pulses < old_total:
                issues.append(
                    f"checkpoint total_pulses {old_total!r} exceeds the requested total "
                    f"{self.total_pulses}; cannot truncate a run (use --fresh to restart)"
                )
            else:
                log.info(
                    "resuming run %s: extending from %s to %s pulses",
                    self.run_id,
                    old_total,
                    self.total_pulses,
                )
        return issues

    # ------------------------------------------------------------------ #
    def wipe(self) -> None:
        """Delete all checkpoint artifacts for this run."""
        for pattern in (
            "checkpoint.json",
            "checkpoint.json.bak",
            "checkpoint.json.tmp",
            "rng_state.npz",
            "rng_state.bak.npz",
            "rng_state.npz.tmp.npz",
        ):
            p = self.dir / pattern
            if p.exists():
                p.unlink()
