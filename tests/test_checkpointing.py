"""TEST 12 — checkpoint recovery: an interrupted + resumed run matches a clean
uninterrupted run (same configuration and seed)."""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pyarrow.parquet as pq
import pytest

from helpers import make_config
from qism.simulation.runner import RunManager

RUN_ID = "checkpoint_test"


def _make_manager(tmp_path, pulses=1000, chunk=500, interval=500):
    cfg = make_config(
        **{
            "protocol.n_pulses": pulses,
            "simulation.chunk_size": chunk,
            "simulation.checkpoint_interval": interval,
            "simulation.master_seed": 777,
        }
    )
    return RunManager(cfg, RUN_ID, output_root=tmp_path / "data", checkpoint_root=tmp_path / "ckpt")


def _read_events(run_dir: pathlib.Path) -> pd.DataFrame:
    return pq.read_table(run_dir / "events.parquet").to_pandas()


def _read_telemetry(run_dir: pathlib.Path) -> pd.DataFrame:
    return pq.read_table(run_dir / "telemetry.parquet").to_pandas()


def test_interrupt_resume_matches_clean_run(tmp_path) -> None:
    # clean uninterrupted run
    fresh_mgr = _make_manager(tmp_path / "fresh", pulses=1000)
    fresh_mgr.run()
    fresh_events = _read_events(fresh_mgr.run_dir)
    fresh_tel = _read_telemetry(fresh_mgr.run_dir)

    # interrupted run: stop after 500 pulses, then resume to 1000
    mgr = _make_manager(tmp_path / "resumed")
    summary = mgr.run(max_pulses=500)
    assert summary.pulses_completed == 500
    assert mgr.checkpoint_mgr.exists()

    summary2 = mgr.run(resume=True, max_pulses=1000)
    assert summary2.pulses_completed == 1000
    assert summary2.status.value == "COMPLETED"

    resumed_events = _read_events(mgr.run_dir)
    resumed_tel = _read_telemetry(mgr.run_dir)

    # identical events and telemetry, no duplication
    pd.testing.assert_frame_equal(resumed_events, fresh_events)
    pd.testing.assert_frame_equal(resumed_tel, fresh_tel)
    assert len(resumed_events) == 1000
    assert resumed_events["pulse_id"].tolist() == list(range(1000))


def test_resume_partial_chunk_matches_clean_run(tmp_path) -> None:
    """A run interrupted inside a chunk (partial chunk committed) resumes
    correctly: the partial chunk is re-simulated and verified bit-identically."""
    fresh_mgr = _make_manager(tmp_path / "fresh_partial", pulses=1000, chunk=1000, interval=1000)
    fresh_mgr.run()
    fresh_events = _read_events(fresh_mgr.run_dir)

    mgr = _make_manager(tmp_path / "resumed_partial", pulses=1000, chunk=1000, interval=1000)
    summary = mgr.run(max_pulses=500)  # ends mid-chunk (500 of 1000)
    assert summary.pulses_completed == 500

    summary2 = mgr.run(resume=True, max_pulses=1000)
    assert summary2.pulses_completed == 1000
    assert summary2.status.value == "COMPLETED"
    pd.testing.assert_frame_equal(_read_events(mgr.run_dir), fresh_events)
    pd.testing.assert_frame_equal(_read_telemetry(mgr.run_dir), _read_telemetry(fresh_mgr.run_dir))


def test_fresh_restart_clears_old_data(tmp_path) -> None:
    mgr = _make_manager(tmp_path / "fresh2")
    mgr.run(max_pulses=500)
    mgr.run(resume=True, fresh=True, max_pulses=1000)
    events = _read_events(mgr.run_dir)
    assert len(events) == 1000
    assert events["pulse_id"].tolist() == list(range(1000))
    # chunk files retained as the authoritative cursor for future resumes
    chunks = mgr.writer.event_chunk_files()
    assert len(chunks) == 2
    assert sum(mgr.writer.chunk_rows(f) for f in chunks) == 1000


def test_corrupt_checkpoint_falls_back_to_backup(tmp_path) -> None:
    # chunk/interval=250 -> saves at 250 and 500, so a .bak exists after the 2nd save
    mgr = _make_manager(tmp_path / "corrupt", chunk=250, interval=250)
    mgr.run(max_pulses=500)
    assert mgr.checkpoint_mgr.json_path.with_name("checkpoint.json.bak").exists()

    # corrupt the checkpoint file
    mgr.checkpoint_mgr.json_path.write_text("{ not valid json !!!", encoding="utf-8")

    summary = mgr.run(resume=True, max_pulses=1000)
    assert summary.pulses_completed == 1000
    events = _read_events(mgr.run_dir)
    assert len(events) == 1000
    assert events["pulse_id"].tolist() == list(range(1000))


def test_resume_after_completion_is_noop(tmp_path) -> None:
    mgr = _make_manager(tmp_path / "completed")
    mgr.run(max_pulses=1000)
    events_before = _read_events(mgr.run_dir)
    summary = mgr.run(resume=True, max_pulses=1000)
    assert summary.status.value == "COMPLETED"
    events_after = _read_events(mgr.run_dir)
    pd.testing.assert_frame_equal(events_before, events_after)


def test_checkpoint_validation_rejects_mismatched_config(tmp_path) -> None:
    mgr = _make_manager(tmp_path / "mismatch", pulses=1000)
    mgr.run(max_pulses=500)

    # same run id but a different configuration hash (distance changed)
    cfg = make_config(
        **{
            "protocol.n_pulses": 1000,
            "simulation.chunk_size": 500,
            "simulation.checkpoint_interval": 500,
            "simulation.master_seed": 777,
            "channel.distance_km": 10.0,
        }
    )
    # same run id / checkpoint location, different configuration
    bad_mgr = RunManager(
        cfg,
        RUN_ID,
        output_root=tmp_path / "mismatch",
        checkpoint_root=tmp_path / "mismatch" / "ckpt",
    )
    with pytest.raises(RuntimeError, match="checkpoint config_hash"):
        bad_mgr.run(resume=True, max_pulses=1000)


def test_manifest_records_status(tmp_path) -> None:
    mgr = _make_manager(tmp_path / "manifest", pulses=1000)
    mgr.run(max_pulses=500)
    manifest = json.loads((mgr.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETED"
    from qism import __version__
    from qism.datasets.manifest import DATASET_VERSION
    assert manifest["simulator_version"] == __version__
    assert manifest["dataset_version"] == DATASET_VERSION


def test_stateful_interrupt_resume_matches_clean_run(tmp_path) -> None:
    """V0.2: with detector dynamics enabled, detector state machines and the
    afterpulse queue survive checkpoints — an interrupted + resumed run is
    bit-identical to a clean uninterrupted run."""
    base = {
        "protocol.n_pulses": 2000,
        "simulation.chunk_size": 500,
        "simulation.checkpoint_interval": 500,
        "simulation.master_seed": 777,
        # V0.2 defaults already enable dynamics; be explicit here
        "detectors.dead_time.enabled": True,
        "detectors.afterpulse.enabled": True,
    }

    fresh_cfg = make_config(**base)
    fresh_mgr = RunManager(
        fresh_cfg,
        RUN_ID,
        output_root=tmp_path / "fresh_stateful",
        checkpoint_root=tmp_path / "ckpt_stateful",
    )
    fresh_mgr.run()
    fresh_events = _read_events(fresh_mgr.run_dir)

    mgr_cfg = make_config(**base)
    mgr = RunManager(
        mgr_cfg,
        RUN_ID,
        output_root=tmp_path / "resumed_stateful",
        checkpoint_root=tmp_path / "ckpt_stateful2",
    )
    summary = mgr.run(max_pulses=1000)  # stop mid-run at a chunk boundary
    assert summary.pulses_completed == 1000
    assert mgr.checkpoint_mgr.exists()

    summary2 = mgr.run(resume=True, max_pulses=2000)
    assert summary2.pulses_completed == 2000
    assert summary2.status.value == "COMPLETED"

    resumed_events = _read_events(mgr.run_dir)
    pd.testing.assert_frame_equal(resumed_events, fresh_events)
    pd.testing.assert_frame_equal(_read_telemetry(mgr.run_dir), _read_telemetry(fresh_mgr.run_dir))


def test_stateful_partial_chunk_resume_matches_clean_run(tmp_path) -> None:
    """V0.2: resume from a run interrupted INSIDE a chunk (partial chunk
    committed) reproduces the uninterrupted run bit-identically even with
    stateful detector dynamics. chunk=500/interval=500 with an interrupt at
    700 pulses exercises the real repair path: checkpoint at 500, committed
    events end mid-chunk-1, the full chunk is re-simulated and verified."""
    base = {
        "protocol.n_pulses": 2000,
        "simulation.chunk_size": 500,
        "simulation.checkpoint_interval": 500,
        "simulation.master_seed": 777,
        "detectors.dead_time.enabled": True,
        "detectors.afterpulse.enabled": True,
    }
    fresh_cfg = make_config(**base)
    fresh_mgr = RunManager(
        fresh_cfg,
        RUN_ID,
        output_root=tmp_path / "fresh_partial2",
        checkpoint_root=tmp_path / "ckpt_partial2",
    )
    fresh_mgr.run()
    fresh_events = _read_events(fresh_mgr.run_dir)

    mgr_cfg = make_config(**base)
    mgr = RunManager(
        mgr_cfg,
        RUN_ID,
        output_root=tmp_path / "resumed_partial2",
        checkpoint_root=tmp_path / "ckpt_partial2b",
    )
    summary = mgr.run(max_pulses=700)  # mid-chunk (700 of 500+500; 200 into chunk 1)
    assert summary.pulses_completed == 700

    summary2 = mgr.run(resume=True, max_pulses=2000)
    assert summary2.pulses_completed == 2000
    assert summary2.status.value == "COMPLETED"
    pd.testing.assert_frame_equal(_read_events(mgr.run_dir), fresh_events)
    pd.testing.assert_frame_equal(_read_telemetry(mgr.run_dir), _read_telemetry(fresh_mgr.run_dir))


def test_checkpoint_contains_runtime_state(tmp_path) -> None:
    """V0.2 checkpoints serialize the detector runtime state (state machine,
    afterpulse queue, rate window)."""
    mgr = _make_manager(tmp_path / "rtstate", pulses=2000, chunk=500, interval=500)
    mgr.run(max_pulses=500)
    ckpt = mgr.checkpoint_mgr.load()
    assert ckpt is not None
    rt = ckpt.get("runtime_state")
    assert rt is not None
    dets = rt.get("detectors")
    assert dets is not None and len(dets) == 2
    for d in dets:
        assert "state" in d
        assert "dead_until" in d
        assert "last_click_time" in d
        assert "pending_afterpulses" in d
        assert "busy_intervals" in d
        assert "tau_eff_s" in d
    assert "rate_cand_counts" in rt
    assert "rate_window_start" in rt
