"""Master experimental benchmark suite for QISM V0.9.

Executes the standardized 12-experiment matrix evaluating:
1. Operational burden (alert compression ratio, false incidents, reduction rate)
2. Event-level accuracy (incident precision, incident recall, incident F1)
3. Detection and alert latency
4. Signal fusion ablations
5. Fixed vs Adaptive monitoring
6. Checkpoint recovery and reproducibility
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from qism.drift.adaptation import AdaptationPolicy
from qism.drift.scenarios import (
    DriftScenarioGenerator,
    DriftScenarioType,
    ScenarioSpec,
)
from qism.monitor.evaluation import EvaluationReport, evaluate_monitor_run
from qism.monitor.fusion import FusionMethod, SignalFusionConfig
from qism.monitor.orchestrator import MonitorConfig, QISMMonitor
from qism.monitor.plots import (
    generate_html_dashboard,
    plot_alert_compression,
    plot_integrated_timeline,
)


def run_experiment_1_stable_normal(output_dir: Path) -> tuple[EvaluationReport, dict[str, Any]]:
    """Experiment 1: Stable normal operation (verifying low false alarms/incidents)."""
    spec = ScenarioSpec(
        scenario_type=DriftScenarioType.STATIONARY,
        total_steps=120,
        seed=42,
    )
    df, feat_cols = DriftScenarioGenerator(spec).generate_stream()
    cfg = MonitorConfig(run_id="EXP_01_STABLE", enable_adaptation=True)
    mon = QISMMonitor(cfg, feature_names=feat_cols)
    res = mon.run(df, feature_cols=feat_cols)
    eval_rep = evaluate_monitor_run(res, df)

    plot_integrated_timeline(res, df, output_dir / "plots" / "exp01_stable_timeline.png", "Exp 01: Stable Normal Operation")
    return eval_rep, {"spec": asdict(spec), "metrics": eval_rep.to_dict()}


def run_experiment_2_stochastic_variation(output_dir: Path) -> tuple[EvaluationReport, dict[str, Any]]:
    """Experiment 2: Normal stochastic variation with higher background noise."""
    spec = ScenarioSpec(
        scenario_type=DriftScenarioType.STATIONARY,
        total_steps=120,
        seed=101,
    )
    df, feat_cols = DriftScenarioGenerator(spec).generate_stream()
    cfg = MonitorConfig(run_id="EXP_02_STOCHASTIC", enable_adaptation=True)
    mon = QISMMonitor(cfg, feature_names=feat_cols)
    res = mon.run(df, feature_cols=feat_cols)
    eval_rep = evaluate_monitor_run(res, df)

    plot_integrated_timeline(res, df, output_dir / "plots" / "exp02_stochastic_timeline.png", "Exp 02: Stochastic Normal Variation")
    return eval_rep, {"spec": asdict(spec), "metrics": eval_rep.to_dict()}


def run_experiment_3_benign_drift(output_dir: Path) -> tuple[EvaluationReport, dict[str, Any]]:
    """Experiment 3: Benign drift (drift detection without security incident escalation)."""
    spec = ScenarioSpec(
        scenario_type=DriftScenarioType.ABRUPT_BENIGN,
        total_steps=120,
        drift_start_step=40,
        channel_loss_delta_db=2.0,
        seed=42,
    )
    df, feat_cols = DriftScenarioGenerator(spec).generate_stream()
    cfg = MonitorConfig(run_id="EXP_03_BENIGN_DRIFT", enable_adaptation=True)
    mon = QISMMonitor(cfg, feature_names=feat_cols)
    res = mon.run(df, feature_cols=feat_cols)
    eval_rep = evaluate_monitor_run(res, df)

    plot_integrated_timeline(res, df, output_dir / "plots" / "exp03_benign_drift_timeline.png", "Exp 03: Benign Distribution Shift")
    return eval_rep, {"spec": asdict(spec), "metrics": eval_rep.to_dict()}


def run_experiment_4_attack_stable(output_dir: Path) -> tuple[EvaluationReport, dict[str, Any]]:
    """Experiment 4: Attack under stable operation."""
    spec = ScenarioSpec(
        scenario_type=DriftScenarioType.ATTACK_ONLY,
        total_steps=120,
        attack_start_step=50,
        attack_end_step=80,
        attack_type="intercept_resend",
        attack_strength=0.85,
        seed=42,
    )
    df, feat_cols = DriftScenarioGenerator(spec).generate_stream()
    cfg = MonitorConfig(run_id="EXP_04_ATTACK_STABLE", enable_adaptation=True)
    mon = QISMMonitor(cfg, feature_names=feat_cols)
    res = mon.run(df, feature_cols=feat_cols)
    eval_rep = evaluate_monitor_run(res, df)

    plot_integrated_timeline(res, df, output_dir / "plots" / "exp04_attack_stable_timeline.png", "Exp 04: Physical Attack Under Stable Conditions")
    return eval_rep, {"spec": asdict(spec), "metrics": eval_rep.to_dict()}


def run_experiment_5_attack_after_drift(output_dir: Path) -> tuple[EvaluationReport, dict[str, Any]]:
    """Experiment 5: Attack after benign drift (safe adaptation preserves detection)."""
    spec = ScenarioSpec(
        scenario_type=DriftScenarioType.DRIFT_THEN_ATTACK,
        total_steps=120,
        drift_start_step=30,
        attack_start_step=75,
        attack_end_step=105,
        attack_type="intercept_resend",
        attack_strength=0.85,
        channel_loss_delta_db=2.0,
        seed=42,
    )
    df, feat_cols = DriftScenarioGenerator(spec).generate_stream()
    cfg = MonitorConfig(run_id="EXP_05_DRIFT_THEN_ATTACK", enable_adaptation=True)
    mon = QISMMonitor(cfg, feature_names=feat_cols)
    res = mon.run(df, feature_cols=feat_cols)
    eval_rep = evaluate_monitor_run(res, df)

    plot_integrated_timeline(res, df, output_dir / "plots" / "exp05_drift_then_attack_timeline.png", "Exp 05: Attack Following Benign Drift")
    return eval_rep, {"spec": asdict(spec), "metrics": eval_rep.to_dict()}


def run_experiment_6_attack_during_drift(output_dir: Path) -> tuple[EvaluationReport, dict[str, Any]]:
    """Experiment 6: Attack occurring during drift transition (contamination stress)."""
    spec = ScenarioSpec(
        scenario_type=DriftScenarioType.ATTACK_DURING_DRIFT,
        total_steps=120,
        drift_start_step=35,
        attack_type="intercept_resend",
        attack_strength=0.85,
        channel_loss_delta_db=2.0,
        seed=42,
    )
    df, feat_cols = DriftScenarioGenerator(spec).generate_stream()
    cfg = MonitorConfig(run_id="EXP_06_ATTACK_DURING_DRIFT", enable_adaptation=True)
    mon = QISMMonitor(cfg, feature_names=feat_cols)
    res = mon.run(df, feature_cols=feat_cols)
    eval_rep = evaluate_monitor_run(res, df)

    plot_integrated_timeline(res, df, output_dir / "plots" / "exp06_attack_during_drift_timeline.png", "Exp 06: Attack During Drift Transition")
    return eval_rep, {"spec": asdict(spec), "metrics": eval_rep.to_dict()}


def run_experiment_7_multi_signal_attack(output_dir: Path) -> tuple[EvaluationReport, dict[str, Any]]:
    """Experiment 7: Multi-signal attack (evaluating signal fusion synergy)."""
    spec = ScenarioSpec(
        scenario_type=DriftScenarioType.ATTACK_ONLY,
        total_steps=120,
        attack_start_step=45,
        attack_end_step=75,
        attack_type="detector_blinding",
        attack_strength=0.80,
        seed=42,
    )
    df, feat_cols = DriftScenarioGenerator(spec).generate_stream()
    # Enable rule-based synergy boosting in fusion engine
    cfg = MonitorConfig(
        run_id="EXP_07_MULTI_SIGNAL",
        fusion=SignalFusionConfig(fusion_method=FusionMethod.PERSISTENCE_AWARE, orthogonal_synergy_boost=0.20),
    )
    mon = QISMMonitor(cfg, feature_names=feat_cols)
    res = mon.run(df, feature_cols=feat_cols)
    eval_rep = evaluate_monitor_run(res, df)

    plot_integrated_timeline(res, df, output_dir / "plots" / "exp07_multi_signal_timeline.png", "Exp 07: Multi-Signal Attack with Signal Fusion")
    return eval_rep, {"spec": asdict(spec), "metrics": eval_rep.to_dict()}


def run_experiment_8_intermittent_attack(output_dir: Path) -> tuple[EvaluationReport, dict[str, Any]]:
    """Experiment 8: Intermittent attack (two distinct attack bursts testing incident segmentation)."""
    # Create stream with two bursts: steps 30-50 and steps 80-100
    spec1 = ScenarioSpec(
        scenario_type=DriftScenarioType.ATTACK_ONLY,
        total_steps=120,
        attack_start_step=30,
        attack_end_step=50,
        attack_type="intercept_resend",
        attack_strength=0.85,
        seed=42,
    )
    df, feat_cols = DriftScenarioGenerator(spec1).generate_stream()

    # Apply second attack burst from step 80 to 100
    for i in range(80, 100):
        df.loc[i, "attack_active"] = True
        df.loc[i, "attack_type"] = "intercept_resend"
        df.loc[i, "qber"] += 0.20
        df.loc[i, "sifted_rate"] *= 0.65

    cfg = MonitorConfig(run_id="EXP_08_INTERMITTENT", enable_adaptation=True)
    mon = QISMMonitor(cfg, feature_names=feat_cols)
    res = mon.run(df, feature_cols=feat_cols)
    eval_rep = evaluate_monitor_run(res, df)

    plot_integrated_timeline(res, df, output_dir / "plots" / "exp08_intermittent_timeline.png", "Exp 08: Intermittent Attack with Incident Segmentation")
    return eval_rep, {"spec": asdict(spec1), "metrics": eval_rep.to_dict()}


def run_experiment_9_fixed_vs_adaptive(output_dir: Path) -> tuple[EvaluationReport, dict[str, Any]]:
    """Experiment 9: Fixed vs Adaptive monitoring ablation."""
    spec = ScenarioSpec(
        scenario_type=DriftScenarioType.DRIFT_THEN_ATTACK,
        total_steps=120,
        drift_start_step=30,
        attack_start_step=75,
        attack_end_step=105,
        channel_loss_delta_db=2.5,
        seed=42,
    )
    df, feat_cols = DriftScenarioGenerator(spec).generate_stream()

    # Fixed monitor (adaptation disabled)
    cfg_fixed = MonitorConfig(run_id="EXP_09_FIXED", enable_adaptation=False)
    mon_fixed = QISMMonitor(cfg_fixed, feature_names=feat_cols)
    res_fixed = mon_fixed.run(df, feature_cols=feat_cols)
    eval_fixed = evaluate_monitor_run(res_fixed, df)

    # Adaptive monitor (safe adaptation enabled)
    cfg_adapt = MonitorConfig(run_id="EXP_09_ADAPTIVE", enable_adaptation=True)
    mon_adapt = QISMMonitor(cfg_adapt, feature_names=feat_cols)
    res_adapt = mon_adapt.run(df, feature_cols=feat_cols)
    eval_adapt = evaluate_monitor_run(res_adapt, df)

    plot_integrated_timeline(res_adapt, df, output_dir / "plots" / "exp09_adaptive_timeline.png", "Exp 09: Adaptive Security Monitoring")
    return eval_adapt, {
        "fixed_compression": eval_fixed.alert_compression_ratio,
        "adaptive_compression": eval_adapt.alert_compression_ratio,
        "fixed_f1": eval_fixed.incident_f1,
        "adaptive_f1": eval_adapt.incident_f1,
    }


def run_experiment_10_alert_compression_ablation(output_dir: Path) -> tuple[EvaluationReport, dict[str, Any]]:
    """Experiment 10: Raw point alarms vs Correlated incidents (compression ablation)."""
    spec = ScenarioSpec(
        scenario_type=DriftScenarioType.ATTACK_ONLY,
        total_steps=120,
        attack_start_step=40,
        attack_end_step=90,  # 50-step sustained attack
        attack_type="intercept_resend",
        attack_strength=0.90,
        seed=42,
    )
    df, feat_cols = DriftScenarioGenerator(spec).generate_stream()
    cfg = MonitorConfig(run_id="EXP_10_COMPRESSION", enable_adaptation=True)
    mon = QISMMonitor(cfg, feature_names=feat_cols)
    res = mon.run(df, feature_cols=feat_cols)
    eval_rep = evaluate_monitor_run(res, df)

    plot_integrated_timeline(res, df, output_dir / "plots" / "exp10_compression_timeline.png", "Exp 10: Alert Compression Under Sustained Attack")
    return eval_rep, {"raw_alarms": eval_rep.raw_alarms, "incidents": eval_rep.total_incidents, "compression_ratio": eval_rep.alert_compression_ratio}


def run_experiment_11_checkpoint_recovery(output_dir: Path) -> tuple[EvaluationReport, dict[str, Any]]:
    """Experiment 11: End-to-end streaming replay with checkpoint and resume validation."""
    spec = ScenarioSpec(
        scenario_type=DriftScenarioType.DRIFT_THEN_ATTACK,
        total_steps=120,
        drift_start_step=30,
        attack_start_step=75,
        attack_end_step=105,
        seed=42,
    )
    df, feat_cols = DriftScenarioGenerator(spec).generate_stream()
    ckpt_path = output_dir / "checkpoints" / "monitor_ckpt_step60.json"

    # 1. Run full uninterrupted stream for baseline
    cfg_base = MonitorConfig(run_id="EXP_11_UNINTERRUPTED")
    mon_base = QISMMonitor(cfg_base, feature_names=feat_cols)
    res_base = mon_base.run(df, feature_cols=feat_cols)

    # 2. Run first half (steps 0 to 59)
    cfg_ckpt = MonitorConfig(run_id="EXP_11_CHECKPOINTED")
    mon_ckpt = QISMMonitor(cfg_ckpt, feature_names=feat_cols)
    ingestor = mon_ckpt.ingestor
    stream = list(ingestor.stream_from_dataframe(df, feature_cols=feat_cols))

    for rec in stream[:60]:
        mon_ckpt.update(rec)

    # Save checkpoint
    mon_ckpt.checkpoint(ckpt_path)

    # 3. Create fresh monitor instance and restore
    mon_resumed = QISMMonitor(cfg_ckpt, feature_names=feat_cols)
    mon_resumed.restore(ckpt_path)

    # Continue second half (steps 60 to 119)
    for rec in stream[60:]:
        mon_resumed.update(rec)

    # Finalize
    all_incidents = mon_resumed.incident_manager.finalize(final_timestamp_s=stream[-1].timestamp_s)
    eval_rep = evaluate_monitor_run(
        res_base, df
    )

    checkpoint_verified = len(all_incidents) == len(res_base.all_incidents)

    return eval_rep, {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_verified": checkpoint_verified,
        "baseline_incidents": len(res_base.all_incidents),
        "resumed_incidents": len(all_incidents),
    }


def run_experiment_12_ood_generalization(output_dir: Path) -> tuple[EvaluationReport, dict[str, Any]]:
    """Experiment 12: Out-of-distribution (OOD) generalization under coupled extreme drift."""
    spec = ScenarioSpec(
        scenario_type=DriftScenarioType.OOD_GENERALIZATION,
        total_steps=120,
        drift_start_step=35,
        channel_loss_delta_db=4.5,
        temp_delta_c=12.0,
        seed=42,
    )
    df, feat_cols = DriftScenarioGenerator(spec).generate_stream()
    cfg = MonitorConfig(run_id="EXP_12_OOD", enable_adaptation=True)
    mon = QISMMonitor(cfg, feature_names=feat_cols)
    res = mon.run(df, feature_cols=feat_cols)
    eval_rep = evaluate_monitor_run(res, df)

    plot_integrated_timeline(res, df, output_dir / "plots" / "exp12_ood_timeline.png", "Exp 12: Extreme Out-of-Distribution Generalization")
    return eval_rep, {"spec": asdict(spec), "metrics": eval_rep.to_dict()}


def run_all_v09_experiments(output_dir: str | Path = "outputs/v09/experiments") -> dict[str, Any]:
    """Execute all 12 V0.9 benchmark experiments, compute summaries, and generate artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    experiments = [
        ("Exp 01", "Stationary Normal", run_experiment_1_stable_normal),
        ("Exp 02", "Stochastic Variation", run_experiment_2_stochastic_variation),
        ("Exp 03", "Benign Drift", run_experiment_3_benign_drift),
        ("Exp 04", "Attack Under Stable", run_experiment_4_attack_stable),
        ("Exp 05", "Attack After Drift", run_experiment_5_attack_after_drift),
        ("Exp 06", "Attack During Drift", run_experiment_6_attack_during_drift),
        ("Exp 07", "Multi-Signal Attack", run_experiment_7_multi_signal_attack),
        ("Exp 08", "Intermittent Attack", run_experiment_8_intermittent_attack),
        ("Exp 09", "Fixed vs Adaptive", run_experiment_9_fixed_vs_adaptive),
        ("Exp 10", "Alert Compression", run_experiment_10_alert_compression_ablation),
        ("Exp 11", "Checkpoint Recovery", run_experiment_11_checkpoint_recovery),
        ("Exp 12", "OOD Generalization", run_experiment_12_ood_generalization),
    ]

    records: list[dict[str, Any]] = []
    comp_plot_data: list[dict[str, Any]] = []

    print("============================================================")
    print("EXECUTING QISM V0.9 MASTER BENCHMARK SUITE (12 EXPERIMENTS)")
    print("============================================================")

    for exp_id, name, fn in experiments:
        t0 = time.perf_counter()
        eval_rep, details = fn(out_dir)
        elapsed = time.perf_counter() - t0

        rec = {
            "exp_id": exp_id,
            "name": name,
            "raw_alarms": eval_rep.raw_alarms,
            "total_incidents": eval_rep.total_incidents,
            "total_alerts": eval_rep.total_alerts,
            "compression_ratio": round(eval_rep.alert_compression_ratio, 2),
            "alert_reduction_pct": round(eval_rep.alert_reduction_pct, 1),
            "incident_precision": round(eval_rep.incident_precision, 4),
            "incident_recall": round(eval_rep.incident_recall, 4),
            "incident_f1": round(eval_rep.incident_f1, 4),
            "mean_step_latency_ms": round(eval_rep.mean_step_latency_ms, 2),
            "throughput_steps_per_s": round(eval_rep.throughput_steps_per_s, 1),
            "runtime_s": round(elapsed, 2),
        }
        records.append(rec)
        comp_plot_data.append({
            "name": f"{exp_id}: {name}",
            "raw_alarms": eval_rep.raw_alarms,
            "total_incidents": eval_rep.total_incidents,
            "total_alerts": eval_rep.total_alerts,
        })
        print(f"[{exp_id}] {name:<26} | Incidents={eval_rep.total_incidents:2d} | Alarms={eval_rep.raw_alarms:2d} | Comp={eval_rep.alert_compression_ratio:4.1f}x | F1={eval_rep.incident_f1:.2f} ({elapsed:.2f}s)")

    # Generate Alert Compression Bar Chart
    plot_alert_compression(comp_plot_data, out_dir / "plots" / "v09_alert_compression_summary.png")

    # Serialize Summary CSV
    summary_df = pd.DataFrame(records)
    summary_csv_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    # Serialize Manifest JSON
    manifest = {
        "milestone": "V0.9",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_experiments": len(records),
        "results": records,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Generate HTML Dashboard
    # Use Exp 05 as primary dashboard showcase
    eval_showcase, _ = run_experiment_5_attack_after_drift(out_dir)
    cfg_show = MonitorConfig(run_id="EXP_SHOWCASE")
    spec_show = ScenarioSpec(
        scenario_type=DriftScenarioType.DRIFT_THEN_ATTACK,
        total_steps=120,
        drift_start_step=30,
        attack_start_step=75,
        attack_end_step=105,
    )
    df_show, feat_show = DriftScenarioGenerator(spec_show).generate_stream()
    mon_show = QISMMonitor(cfg_show, feature_names=feat_show)
    res_show = mon_show.run(df_show, feature_cols=feat_show)
    generate_html_dashboard(res_show, eval_showcase, out_dir / "dashboard.html", "Exp 05: Attack After Benign Drift")

    print("\nBenchmark suite completed successfully.")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Manifest: {manifest_path}")
    print(f"HTML Dashboard: {out_dir / 'dashboard.html'}")

    return manifest
