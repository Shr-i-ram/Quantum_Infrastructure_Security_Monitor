"""Tests for QISM V0.9 orchestrator, streaming replay, checkpointing, and evaluation."""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from qism.drift.scenarios import DriftScenarioGenerator, DriftScenarioType, ScenarioSpec
from qism.monitor.evaluation import evaluate_monitor_run
from qism.monitor.orchestrator import MonitorConfig, QISMMonitor
from qism.monitor.types import IncidentState, SeverityLevel


def test_monitor_sequential_streaming_step():
    spec = ScenarioSpec(scenario_type=DriftScenarioType.STATIONARY, total_steps=20, seed=42)
    df, feat_cols = DriftScenarioGenerator(spec).generate_stream()

    cfg = MonitorConfig(run_id="test_stream")
    mon = QISMMonitor(cfg, feature_names=feat_cols)

    rec_dict = df.iloc[0].to_dict()
    step_res = mon.update(rec_dict)

    assert step_res.telemetry_valid is True
    assert step_res.step_index == 1
    assert step_res.anomaly_score >= 0.0
    assert 0.0 <= step_res.evidence_score <= 1.0
    assert step_res.current_severity == SeverityLevel.INFO
    assert step_res.processing_latency_ms > 0.0


def test_monitor_batch_run_and_evaluation():
    spec = ScenarioSpec(
        scenario_type=DriftScenarioType.ATTACK_ONLY,
        total_steps=50,
        attack_start_step=20,
        attack_end_step=35,
        attack_type="intercept_resend",
        attack_strength=0.85,
        seed=42,
    )
    df, feat_cols = DriftScenarioGenerator(spec).generate_stream()

    cfg = MonitorConfig(run_id="test_batch_run")
    mon = QISMMonitor(cfg, feature_names=feat_cols)

    run_result = mon.run(df, feature_cols=feat_cols)
    assert run_result.total_steps == 50
    assert len(run_result.step_results) == 50
    assert len(run_result.all_incidents) >= 1

    # Evaluate against offline ground truth
    eval_rep = evaluate_monitor_run(run_result, df)
    assert eval_rep.total_incidents >= 1
    assert eval_rep.incident_recall == 1.0  # Successfully detected true attack event
    assert eval_rep.alert_compression_ratio >= 1.0
    assert eval_rep.mean_detection_latency_s is not None


def test_monitor_checkpoint_and_restore(tmp_path: Path):
    spec = ScenarioSpec(
        scenario_type=DriftScenarioType.DRIFT_THEN_ATTACK,
        total_steps=40,
        drift_start_step=15,
        attack_start_step=25,
        attack_end_step=35,
        seed=42,
    )
    df, feat_cols = DriftScenarioGenerator(spec).generate_stream()
    ckpt_file = tmp_path / "monitor_ckpt.json"

    # Step 1: Run uninterrupted
    cfg1 = MonitorConfig(run_id="run_continuous")
    mon_cont = QISMMonitor(cfg1, feature_names=feat_cols)
    res_cont = mon_cont.run(df, feature_cols=feat_cols)

    # Step 2: Run half-way, checkpoint, restore, continue
    cfg2 = MonitorConfig(run_id="run_checkpointed")
    mon_split = QISMMonitor(cfg2, feature_names=feat_cols)
    records = list(mon_split.ingestor.stream_from_dataframe(df, feature_cols=feat_cols))

    for rec in records[:20]:
        mon_split.update(rec)

    # Save state
    mon_split.checkpoint(ckpt_file)
    assert ckpt_file.exists()

    # Create new monitor and restore
    mon_resumed = QISMMonitor(cfg2, feature_names=feat_cols)
    mon_resumed.restore(ckpt_file)

    # Continue remaining stream
    for rec in records[20:]:
        mon_resumed.update(rec)

    all_incs = mon_resumed.incident_manager.finalize(final_timestamp_s=records[-1].timestamp_s)

    # Validate state recovery: incident counts should match continuous run
    assert len(all_incs) == len(res_cont.all_incidents)
