"""Run manager: orchestrates chunked simulation, dataset writing, checkpointing
and crash recovery for one run.

Flow per chunk:
    simulate chunk (memory) -> write events chunk (atomic) -> aggregate windows
    -> checkpoint (atomic). The checkpoint is advisory; committed event chunk
    files are the source of truth, so a crash at any point can be resumed
    without duplicating or losing events.

Detector dynamics (V0.2): the simulator carries physical state (READY/DEAD
state machines, pending afterpulse queues, the rate-estimation window) that is
serialized into every checkpoint. Chunks are always simulated at their full
size so that chunk RNG streams — and therefore the physics — are invariant to
the run length; a run interrupted mid-chunk is repaired by re-simulating that
one chunk and verifying the committed prefix bit-identically.
"""

from __future__ import annotations

import logging
import pathlib
import shutil
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from tqdm import tqdm

from qism import __version__ as SIMULATOR_VERSION
from qism.config import QKDConfig
from qism.datasets.manifest import build_manifest, write_manifest_atomic
from qism.datasets.writer import DatasetWriter
from qism.random import derive_run_seed, make_rng
from qism.simulation.checkpoint import CheckpointManager
from qism.simulation.simulator import Simulator
from qism.telemetry.windows import WindowAggregator
from qism.types import DatasetStatus, RunStatus

log = logging.getLogger("qism.runner")


@dataclass
class RunSummary:
    """Result of one simulation run."""

    run_id: str
    run_dir: pathlib.Path
    scenario_id: str
    family: str
    pulses_completed: int
    status: DatasetStatus
    qber: float = float("nan")
    qber_z: float = float("nan")
    qber_x: float = float("nan")
    signal_gain: float = float("nan")
    decoy_gain: float = float("nan")
    vacuum_gain: float = float("nan")
    detector_imbalance: float = float("nan")
    double_click_rate: float = float("nan")
    n_windows: int = 0
    issues: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["run_dir"] = str(d["run_dir"])
        d["status"] = d["status"].value if hasattr(d["status"], "value") else d["status"]
        return d


class RunManager:
    """Runs (or resumes) one simulation run end-to-end."""

    def __init__(
        self,
        config: QKDConfig,
        run_id: str,
        *,
        output_root: str | None = None,
        checkpoint_root: str | None = None,
    ):
        self.config = config
        self.run_id = run_id
        self.scenario_id = config.simulation.scenario_id
        self.family = config.dataset.family
        self.master_seed = config.simulation.master_seed
        self.seed = derive_run_seed(self.master_seed, run_id)
        self.output_root = pathlib.Path(output_root or config.simulation.output_root)
        self.checkpoint_root = pathlib.Path(checkpoint_root or config.simulation.checkpoint_root)
        self.run_dir = self.output_root / self.family / run_id
        self.chunk_size = config.simulation.chunk_size
        self.window_size = config.telemetry.window_size
        self.checkpoint_interval = config.simulation.checkpoint_interval
        self.config_hash = config.config_hash()

        self.simulator = Simulator(config, run_id, self.seed, scenario_id=self.scenario_id)
        self.writer = DatasetWriter(self.run_dir)
        self.aggregator = WindowAggregator(config)
        self.checkpoint_mgr = CheckpointManager(
            self.checkpoint_root / run_id,
            config_hash=self.config_hash,
            simulator_version=SIMULATOR_VERSION,
            run_id=run_id,
            seed=self.seed,
            master_seed=self.master_seed,
            scenario_id=self.scenario_id,
            family=self.family,
            run_dir=self.run_dir,
            total_pulses=config.protocol.n_pulses,
            window_size=self.window_size,
            output_root=str(self.output_root),
            checkpoint_root=str(self.checkpoint_root),
        )
        self.run_rng = make_rng(self.seed)
        self._total_pulses = self.config.protocol.n_pulses
        self.manifest = self._initial_manifest()

        # window accumulation state
        self._pending: pd.DataFrame | None = None
        self._pending_wid: int | None = None

    # ------------------------------------------------------------------ #
    def _initial_manifest(self) -> dict[str, Any]:
        return build_manifest(
            run_id=self.run_id,
            scenario=self.scenario_id,
            family=self.family,
            master_seed=self.master_seed,
            seed=self.seed,
            number_of_pulses=self._total_pulses,
            window_size=self.window_size,
            config_hash=self.config_hash,
            status=DatasetStatus.CREATED,
        )

    # ------------------------------------------------------------------ #
    def run(
        self, *, resume: bool = False, fresh: bool = False, max_pulses: int | None = None
    ) -> RunSummary:
        """Execute (or resume) the run. ``max_pulses`` is a runtime override of
        the total (useful for tests and quick experiments)."""
        t_start = time.time()
        total = max_pulses if max_pulses is not None else self._total_pulses
        if total <= 0:
            raise ValueError("total pulses must be > 0")

        if fresh:
            self._wipe()

        pulses_committed, chunk_index, repaired = self._start_or_resume(resume=resume, total=total)
        # the entering state of the next chunk to simulate (captured after
        # restore/replay and refreshed after every simulated chunk)
        state_at = self.simulator.runtime_state()

        if pulses_committed >= total:
            # Only skip when the resume did NOT repair a partial chunk (which
            # would leave the combined Parquet files stale) and the run is
            # genuinely complete.
            if not repaired and self._is_completed():
                log.info("run %s already completed; nothing to do", self.run_id)
                return self._summary_from_disk(total, time.time() - t_start)
            pulses_committed = self._reconcile_committed(pulses_committed, total)

        log.info(
            "run %s: scenario=%s seed=%d pulses=%d chunk_size=%d window_size=%d dynamics=%s",
            self.run_id,
            self.scenario_id,
            self.seed,
            total,
            self.chunk_size,
            self.window_size,
            self.simulator.dynamics_enabled,
        )

        try:
            with tqdm(
                total=total, initial=pulses_committed, desc=f"run {self.run_id}", unit="pulse"
            ) as pbar:
                while pulses_committed < total:
                    # Chunks are ALWAYS simulated at their full size so that
                    # chunk RNG streams — and therefore the stateful detector
                    # physics — are invariant to the run length: a partial
                    # chunk can be re-simulated bit-identically on resume (see
                    # _align_partial_chunk). Rows beyond the requested total
                    # are truncated at write time.
                    events_full = self.simulator.simulate_chunk(
                        chunk_index, self.chunk_size, pulses_committed
                    )
                    events = events_full.iloc[: min(self.chunk_size, total - pulses_committed)]
                    self.writer.write_events_chunk(events, chunk_index)
                    self._consume_events(events, total)
                    pulses_committed += len(events)
                    chunk_index += 1
                    # entering state of the next chunk (grid boundary)
                    state_at = self.simulator.runtime_state()
                    pbar.update(len(events))
                    last_full = len(events) == self.chunk_size
                    if last_full and pulses_committed % self.checkpoint_interval == 0:
                        self.checkpoint_mgr.save(
                            pulses_committed=pulses_committed,
                            chunk_index=chunk_index,
                            run_rng=self.run_rng,
                            status=RunStatus.CHECKPOINTED,
                            runtime_state=state_at,
                        )
        except KeyboardInterrupt:
            log.warning("run %s interrupted by user; checkpoint preserved", self.run_id)
            self.checkpoint_mgr.save(
                pulses_committed=pulses_committed,
                chunk_index=chunk_index,
                run_rng=self.run_rng,
                status=RunStatus.INTERRUPTED,
                runtime_state=state_at,
            )
            self._set_manifest_status(DatasetStatus.INTERRUPTED)
            return self._final_summary(
                pulses_committed, total, DatasetStatus.INTERRUPTED, time.time() - t_start
            )
        except Exception:
            log.exception("run %s failed; latest valid checkpoint preserved", self.run_id)
            if pulses_committed > 0:
                self.checkpoint_mgr.save(
                    pulses_committed=pulses_committed,
                    chunk_index=chunk_index,
                    run_rng=self.run_rng,
                    status=RunStatus.CHECKPOINTED,
                    runtime_state=state_at,
                )
            self._set_manifest_status(DatasetStatus.FAILED)
            raise

        # ---- finalize ----
        issues = self._finalize(total)
        status = DatasetStatus.FAILED if issues else DatasetStatus.COMPLETED
        self._set_manifest_status(status)
        elapsed = time.time() - t_start
        summary = self._final_summary(total, total, status, elapsed)
        summary.issues = issues
        log.info("run %s finished in %.1fs: status=%s", self.run_id, elapsed, status.value)
        return summary

    # ------------------------------------------------------------------ #
    def _wipe(self) -> None:
        log.warning("--fresh: wiping run directory %s and checkpoints", self.run_dir)
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)
        self.checkpoint_mgr.wipe()

    # ------------------------------------------------------------------ #
    def _start_or_resume(self, *, resume: bool, total: int) -> tuple[int, int, bool]:
        """Returns (pulses_committed, next_chunk_index, repaired_partial_chunk).

        When resuming with detector dynamics, the simulator's physical state
        (dead time, afterpulse queues, rate window) is restored from the
        checkpoint and advanced by re-simulating any committed chunks that
        were checkpointed but not yet reflected in the saved state. The
        committed event chunk files on disk remain the authoritative record.
        """
        checkpoint = (
            self.checkpoint_mgr.load() if (resume or self.checkpoint_mgr.exists()) else None
        )

        if checkpoint is not None:
            issues = self.checkpoint_mgr.validate(checkpoint)
            if issues:
                raise RuntimeError(
                    "Existing checkpoint is incompatible with the requested run:\n  - "
                    + "\n  - ".join(issues)
                    + "\nUse --fresh to discard the checkpoint and start over."
                )
            log.info(
                "checkpoint found for run %s (status=%s)", self.run_id, checkpoint.get("status")
            )
            self.run_rng = self.checkpoint_mgr.load_rng(checkpoint)
            pulses_committed = self._committed_from_disk(total)
            # a partial final chunk is repaired (re-simulated and verified)
            # BEFORE telemetry is rebuilt from the committed event files
            pulses_committed, repaired = self._restore_and_align(
                pulses_committed, total, checkpoint
            )
            # telemetry is a pure function of events: rebuild it from committed events
            self.writer.clear_telemetry_chunks()
            self._pending, self._pending_wid = None, None
            self._rebuild_telemetry_from_events(total)
            self._set_manifest_status(DatasetStatus.RUNNING)
            log.info("resuming run %s from pulse %d", self.run_id, pulses_committed)
        else:
            pulses_committed = 0
            repaired = False
            # a fresh manager may have a dirty simulator from a previous run
            self.simulator.reset_runtime_state()
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.writer.write_config(self.config)
            self.writer.write_manifest(self.manifest)
            self._set_manifest_status(DatasetStatus.RUNNING)

        chunk_index = pulses_committed // self.chunk_size
        return pulses_committed, chunk_index, repaired

    # ------------------------------------------------------------------ #
    def _restore_and_align(
        self, pulses_committed: int, total: int, checkpoint: dict[str, Any]
    ) -> tuple[int, bool]:
        """Restore simulator physical state and align to a chunk boundary.

        Returns (pulses_committed, repaired_partial_chunk).
        """
        ckpt_state = checkpoint.get("runtime_state")
        ckpt_grid = int(checkpoint.get("chunk_index", 0))

        # The committed prefix may end mid-chunk (partial final file): its
        # chunk index is the one whose *entering* state we need.
        need_grid = pulses_committed // self.chunk_size

        # Every checkpoint stores the entering state of `chunk_index`; grid
        # gaps (crash before the next scheduled save) are closed by replaying
        # committed chunks, which is bounded by the checkpoint interval.
        if ckpt_state is not None:
            self.simulator.restore_runtime_state(ckpt_state)
        if ckpt_grid < need_grid:
            if ckpt_state is None:
                log.warning(
                    "checkpoint has no runtime state; replaying %d committed chunk(s) to "
                    "rebuild detector state",
                    need_grid,
                )
            log.info(
                "replaying %d committed chunk(s) to rebuild detector state (grid %d -> %d)",
                need_grid - ckpt_grid,
                ckpt_grid,
                need_grid,
            )
            for gi in range(ckpt_grid, need_grid):
                # full-size chunk simulation reproduces the committed events
                self.simulator.simulate_chunk(gi, self.chunk_size, gi * self.chunk_size)
        elif ckpt_grid > need_grid:
            # The checkpoint claims more progress than the files on disk (state
            # is only saved after events are committed, so this should not
            # happen); rebuild from scratch.
            log.warning(
                "checkpoint grid %d exceeds committed files (%d pulses); re-simulating from 0",
                ckpt_grid,
                pulses_committed,
            )
            self.simulator.reset_runtime_state()
            for gi in range(0, need_grid):
                self.simulator.simulate_chunk(gi, self.chunk_size, gi * self.chunk_size)

        if pulses_committed % self.chunk_size != 0:
            return self._align_partial_chunk(pulses_committed, total)
        return pulses_committed, False

    def _committed_from_disk(self, total: int) -> int:
        """Authoritative committed pulse count from event chunk files on disk."""
        files = self.writer.event_chunk_files()
        if not files:
            return 0
        committed = 0
        expected_index = 0
        for f in files:
            index = self.writer.chunk_index_of(f)
            if index != expected_index:
                raise RuntimeError(
                    f"event chunk files are not contiguous: expected chunk {expected_index}, "
                    f"found {index} in {f}"
                )
            rows = self.writer.chunk_rows(f)
            if rows <= 0:
                raise RuntimeError(f"event chunk file {f} is empty or corrupt")
            committed += rows
            expected_index += 1
        if committed > total:
            raise RuntimeError(
                f"committed events ({committed}) exceed the requested total ({total}); "
                "the run configuration has changed. Use --fresh to restart."
            )
        return committed

    def _align_partial_chunk(self, pulses_committed: int, total: int) -> tuple[int, bool]:
        """Repair a committed run that ends mid-chunk.

        The committed events end at a non-multiple of ``chunk_size`` (a partial
        final chunk — e.g. a run whose total was not a multiple of the chunk
        size, or a crash inside the last chunk). Because chunk RNGs are derived
        deterministically from (run_seed, chunk_index) and the detector state
        entering the chunk has been restored/replayed, re-simulating the whole
        chunk reproduces the committed prefix **bit-identically**; we verify
        that and then rewrite the chunk file with the full chunk, continuing
        from the aligned boundary. Work is bounded to one chunk, never
        proportional to run length.
        """
        if pulses_committed % self.chunk_size == 0:
            return pulses_committed, False
        chunk_index = pulses_committed // self.chunk_size
        chunk_start = chunk_index * self.chunk_size
        # Full-size chunk simulation (never truncated): the RNG stream for the
        # chunk is then identical to the one that produced the committed events.
        events_full = self.simulator.simulate_chunk(chunk_index, self.chunk_size, chunk_start)
        events = events_full.iloc[: min(self.chunk_size, total - chunk_start)]
        path = self.writer.chunks_dir / f"events_{chunk_index:06d}.parquet"
        prefix_len = pulses_committed - chunk_start
        if path.exists():
            existing = self.writer.read_events_chunk(path)
            if len(existing) != prefix_len:
                raise RuntimeError(
                    f"partial chunk {path} has {len(existing)} rows, expected {prefix_len}"
                )
            try:
                pd.testing.assert_frame_equal(existing, events_full.iloc[:prefix_len])
            except AssertionError as exc:
                raise RuntimeError(
                    f"re-simulated chunk {chunk_index} does not match the committed events "
                    f"({exc}); the simulator version or configuration changed. "
                    "Use --fresh to restart."
                ) from exc
            log.info(
                "repairing partial chunk %d: %d committed pulses verified, re-simulating to %d",
                chunk_index,
                prefix_len,
                chunk_start + len(events),
            )
        self.writer.write_events_chunk(events, chunk_index)
        return chunk_start + len(events), True

    def _reconcile_committed(self, pulses_committed: int, total: int) -> int:
        from_disk = self._committed_from_disk(total)
        if from_disk > pulses_committed:
            log.warning(
                "event files on disk (%d pulses) exceed the checkpoint (%d); trusting the files",
                from_disk,
                pulses_committed,
            )
        return from_disk

    def _rebuild_telemetry_from_events(self, total: int) -> None:
        """Re-aggregate telemetry for all committed event chunk files."""
        for f in self.writer.event_chunk_files():
            events = self.writer.read_events_chunk(f)
            self._consume_events(events, total)

    # ------------------------------------------------------------------ #
    # Window accumulation
    # ------------------------------------------------------------------ #
    def _consume_events(self, events: pd.DataFrame, total: int) -> None:
        for wid, sub in events.groupby("window_id", sort=False):
            wid = int(wid)
            if self._pending is not None:
                if self._pending_wid == wid:
                    sub = pd.concat([self._pending, sub], ignore_index=True)
                else:
                    self._flush_pending()
            self._pending, self._pending_wid = sub, wid
            self._emit_if_complete(sub, wid, total)

    def _emit_if_complete(self, sub: pd.DataFrame, wid: int, total: int) -> None:
        expected_end = min((wid + 1) * self.window_size, total)
        window_start = wid * self.window_size
        complete = (
            int(sub["pulse_id"].max()) == expected_end - 1
            and len(sub) == expected_end - window_start
        )
        if complete:
            row = self.aggregator.aggregate(sub)
            self.writer.write_telemetry_chunk(pd.DataFrame([row]), wid)
            self._pending, self._pending_wid = None, None

    def _flush_pending(self) -> None:
        if self._pending is not None and len(self._pending) > 0:
            row = self.aggregator.aggregate(self._pending)
            self.writer.write_telemetry_chunk(pd.DataFrame([row]), self._pending_wid)
            self._pending, self._pending_wid = None, None

    # ------------------------------------------------------------------ #
    def _finalize(self, total: int) -> list[str]:
        """Flush trailing window, write combined Parquet, run quality checks."""
        self._flush_pending()
        issues = self.writer.finalize(
            expected_pulses=total,
            config_hash=self.config_hash,
            simulator_version=SIMULATOR_VERSION,
            window_size=self.window_size,
        )
        return issues

    def _set_manifest_status(self, status: DatasetStatus) -> None:
        # Merge with the on-disk manifest first: the writer may have updated it
        # during finalize (pulses_completed, file inventory), and we must not
        # clobber that provenance with the in-memory (stale) copy.
        import json as _json

        disk = self.run_dir / "manifest.json"
        if disk.exists():
            try:
                self.manifest.update(_json.loads(disk.read_text(encoding="utf-8")))
            except (_json.JSONDecodeError, OSError):
                log.warning("could not read existing manifest %s; using in-memory copy", disk)
        self.manifest["status"] = status.value
        self.manifest["timestamp"] = build_manifest(
            run_id=self.run_id,
            scenario=self.scenario_id,
            family=self.family,
            master_seed=self.master_seed,
            seed=self.seed,
            number_of_pulses=self._total_pulses,
            window_size=self.window_size,
            config_hash=self.config_hash,
            status=status,
        )["timestamp"]
        write_manifest_atomic(self.manifest, self.run_dir / "manifest.json")

    # ------------------------------------------------------------------ #
    def _summary_from_disk(self, total: int, elapsed: float) -> RunSummary:
        from_disk = self._committed_from_disk(total)
        if from_disk == 0:
            # chunks were removed by a successful finalize; trust the manifest
            m = self.run_dir / "manifest.json"
            if m.exists():
                import json

                try:
                    manifest = json.loads(m.read_text(encoding="utf-8"))
                    from_disk = int(manifest.get("pulses_completed", 0))
                except (json.JSONDecodeError, ValueError):
                    from_disk = 0
        return self._final_summary(from_disk, total, DatasetStatus.COMPLETED, elapsed)

    def _final_summary(
        self, pulses_completed: int, total: int, status: DatasetStatus, elapsed: float
    ) -> RunSummary:
        summary = RunSummary(
            run_id=self.run_id,
            run_dir=self.run_dir,
            scenario_id=self.scenario_id,
            family=self.family,
            pulses_completed=pulses_completed,
            status=status,
            elapsed_s=elapsed,
        )
        telemetry = self.writer.read_final_telemetry()
        if telemetry is not None and len(telemetry) > 0:
            summary.n_windows = len(telemetry)
            n_s = int(telemetry["sifted_count"].sum())
            n_e = int(telemetry["error_count"].sum())
            if n_s:
                summary.qber = n_e / n_s
            n_sz = int(telemetry["n_sifted_z"].sum())
            n_ez = int(telemetry["n_errors_z"].sum())
            n_sx = int(telemetry["n_sifted_x"].sum())
            n_ex = int(telemetry["n_errors_x"].sum())
            if n_sz:
                summary.qber_z = n_ez / n_sz
            if n_sx:
                summary.qber_x = n_ex / n_sx
            for cls, attr in (
                ("signal", "signal_gain"),
                ("decoy", "decoy_gain"),
                ("vacuum", "vacuum_gain"),
            ):
                n_sent = int(telemetry[f"n_{cls}"].sum())
                n_click = int((telemetry[f"{cls}_gain"] * telemetry[f"n_{cls}"]).sum())
                if n_sent:
                    setattr(summary, attr, n_click / n_sent)
            c0 = int(telemetry["detector_0_count"].sum())
            c1 = int(telemetry["detector_1_count"].sum())
            if c0 + c1:
                summary.detector_imbalance = (c0 - c1) / (c0 + c1)
            n_det = int(telemetry["n_detection"].sum())
            n_dbl = int(telemetry["n_double"].sum())
            if n_det:
                summary.double_click_rate = n_dbl / n_det
        return summary

    def _is_completed(self) -> bool:
        m = self.run_dir / "manifest.json"
        if not m.exists():
            return False
        return self.writer.final_events_exists()
