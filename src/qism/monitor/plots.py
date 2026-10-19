"""Integrated visualization and monitoring report generator for QISM V0.9.

Generates:
1. Publication-quality multi-panel timeline figure (telemetry, anomaly scores, drift, evidence, incidents, ground truth)
2. Operational alert compression and fatigue reduction charts
3. Feature contribution rankings
4. High-resolution standalone HTML monitoring dashboard / report
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from qism.monitor.evaluation import EvaluationReport
from qism.monitor.types import Incident, MonitorRunResult, SeverityLevel


def plot_integrated_timeline(
    run_result: MonitorRunResult,
    ground_truth_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "QISM V0.9 Integrated Security Monitoring Timeline",
) -> None:
    """Generate comprehensive 5-panel operational timeline figure."""
    steps = run_result.step_results
    if not steps:
        return

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    times = [s.timestamp_s for s in steps]
    anom_scores = [s.anomaly_score for s in steps]
    anom_thresh = [s.anomaly_threshold for s in steps]
    drift_scores = [s.drift_score for s in steps]
    drift_thresh = [s.drift_threshold for s in steps]
    evidence_scores = [s.evidence_score for s in steps]

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True, dpi=300)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)

    # Panel 1: Key Telemetry Features
    ax1 = axes[0]
    if "qber" in ground_truth_df.columns:
        ax1.plot(times, ground_truth_df["qber"] * 100.0, label="QBER (%)", color="#d62728", lw=1.5)
    if "signal_gain" in ground_truth_df.columns:
        ax1.plot(times, ground_truth_df["signal_gain"] * 100.0, label="Signal Gain (%)", color="#1f77b4", lw=1.2)
    ax1.set_ylabel("Telemetry (%)")
    ax1.set_title("1. Physical Telemetry Observations (Decoy-State BB84)", fontsize=10, loc="left")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Temporal Anomaly Detection
    ax2 = axes[1]
    ax2.plot(times, anom_scores, label="Anomaly Score", color="#ff7f0e", lw=1.5)
    ax2.plot(times, anom_thresh, label="Threshold", color="#333333", linestyle="--", lw=1.2)
    ax2.set_ylabel("Anomaly Score")
    ax2.set_title("2. Temporal Anomaly Detection Engine (V0.7)", fontsize=10, loc="left")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Statistical Drift Detection
    ax3 = axes[2]
    ax3.plot(times, drift_scores, label="Wasserstein Drift Score", color="#2ca02c", lw=1.5)
    ax3.plot(times, drift_thresh, label="Drift Threshold", color="#333333", linestyle="--", lw=1.2)
    ax3.set_ylabel("Drift Distance")
    ax3.set_title("3. Statistical Drift Tracking & Reference Management (V0.8)", fontsize=10, loc="left")
    ax3.legend(loc="upper right", framealpha=0.9)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Panel 4: Signal Fusion Evidence Score
    ax4 = axes[3]
    ax4.plot(times, evidence_scores, label="Unified Evidence Score", color="#9467bd", lw=1.8)
    ax4.axhline(0.45, color="#e377c2", linestyle=":", label="Incident Trigger (0.45)", lw=1.2)
    ax4.axhline(0.75, color="#d62728", linestyle=":", label="Escalation Threshold (0.75)", lw=1.2)
    ax4.set_ylabel("Evidence Score [0, 1]")
    ax4.set_title("4. Multivariate Signal Fusion & Correlation (V0.9)", fontsize=10, loc="left")
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(loc="upper right", framealpha=0.9)
    ax4.grid(True, linestyle="--", alpha=0.5)

    # Panel 5: Incident Lifecycles & Ground Truth
    ax5 = axes[4]
    ax5.set_ylabel("Security State")
    ax5.set_xlabel("Time (s)")
    ax5.set_title("5. Operational Incidents vs Offline Ground Truth", fontsize=10, loc="left")

    # Draw Ground Truth (Shaded grey/red regions)
    if "attack_active" in ground_truth_df.columns:
        atk_mask = ground_truth_df["attack_active"].to_numpy(dtype=bool)
        ax5.fill_between(times, 0, 1, where=atk_mask, color="#ff9896", alpha=0.4, label="True Attack Window (Offline GT)")

    # Draw Incidents
    sev_colors = {
        SeverityLevel.INFO: "#17becf",
        SeverityLevel.LOW: "#2ca02c",
        SeverityLevel.MEDIUM: "#bcbd22",
        SeverityLevel.HIGH: "#ff7f0e",
        SeverityLevel.CRITICAL: "#d62728",
    }
    for inc in run_result.all_incidents:
        t_start = inc.start_time_s
        t_end = inc.end_time_s if inc.end_time_s is not None else t_start + inc.duration_s
        c = sev_colors.get(inc.peak_severity, "#ff7f0e")
        ax5.axvspan(t_start, t_end, ymin=0.3, ymax=0.8, color=c, alpha=0.7, label=f"Incident {inc.incident_id} ({inc.peak_severity.value})")

    ax5.set_ylim(0, 1)
    ax5.set_yticks([])
    ax5.grid(True, linestyle="--", alpha=0.5)

    handles, labels = ax5.get_legend_handles_labels()
    # Deduplicate legend entries
    by_label = dict(zip(labels, handles))
    ax5.legend(by_label.values(), by_label.keys(), loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(p, bbox_inches="tight")
    plt.close(fig)


def plot_alert_compression(
    experiments_data: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Generate comparative bar chart showing Raw Alarms vs Correlated Incidents."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    names = [e["name"] for e in experiments_data]
    raw_alarms = [e["raw_alarms"] for e in experiments_data]
    incidents = [e["total_incidents"] for e in experiments_data]
    alerts = [e["total_alerts"] for e in experiments_data]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    rects1 = ax.bar(x - width, raw_alarms, width, label="Raw Point Alarms", color="#ff7f0e", alpha=0.85)
    rects2 = ax.bar(x, alerts, width, label="Emitted Alerts (Deduplicated)", color="#2ca02c", alpha=0.85)
    rects3 = ax.bar(x + width, incidents, width, label="Correlated Incidents", color="#1f77b4", alpha=0.85)

    ax.set_ylabel("Event Count")
    ax.set_title("Operational Alert Compression & Operator Fatigue Reduction in QISM V0.9", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.tight_layout()
    plt.savefig(p, bbox_inches="tight")
    plt.close(fig)


def generate_html_dashboard(
    run_result: MonitorRunResult,
    eval_report: EvaluationReport,
    output_path: str | Path,
    scenario_name: str = "Benchmark Scenario",
) -> None:
    """Generate standalone HTML monitoring dashboard."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    incidents_rows = ""
    for inc in run_result.all_incidents:
        reasons_list = "".join(f"<li>{r}</li>" for r in inc.explanation.get("reasons", []))
        rec_text = inc.recommendation.advisory_text if inc.recommendation else "Nominal monitoring"
        incidents_rows += f"""
        <tr>
            <td><strong>{inc.incident_id}</strong></td>
            <td><span class="badge badge-{inc.peak_severity.value}">{inc.peak_severity.value.upper()}</span></td>
            <td>{inc.start_time_s:.3f}s</td>
            <td>{inc.duration_s * 1000.0:.1f} ms</td>
            <td>{inc.peak_evidence_score:.2f}</td>
            <td>{inc.alarm_count}</td>
            <td>{', '.join(inc.subsystems_involved)}</td>
            <td><small>{rec_text}</small></td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>QISM V0.9 Operational Security Dashboard</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #0f172a; color: #f8fafc; }}
        h1, h2, h3 {{ color: #38bdf8; }}
        .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-bottom: 25px; }}
        .card {{ background: #1e293b; padding: 18px; border-radius: 8px; border: 1px solid #334155; }}
        .card-title {{ font-size: 0.85em; color: #94a3b8; text-transform: uppercase; }}
        .card-value {{ font-size: 1.8em; font-weight: bold; margin-top: 5px; color: #f1f5f9; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; background: #1e293b; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #0f172a; color: #38bdf8; font-size: 0.9em; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; }}
        .badge-critical {{ background: #ef4444; color: white; }}
        .badge-high {{ background: #f97316; color: white; }}
        .badge-medium {{ background: #eab308; color: black; }}
        .badge-low {{ background: #22c55e; color: white; }}
        .badge-info {{ background: #06b6d4; color: white; }}
    </style>
</head>
<body>
    <h1>QISM V0.9 Security Monitoring Dashboard</h1>
    <p>Operational incident correlation, multi-signal fusion, and safe adaptive telemetry analysis for QKD infrastructure.</p>
    <div class="cards">
        <div class="card">
            <div class="card-title">Scenario</div>
            <div class="card-value" style="font-size: 1.2em;">{scenario_name}</div>
        </div>
        <div class="card">
            <div class="card-title">Alert Compression</div>
            <div class="card-value" style="color: #22c55e;">{eval_report.alert_compression_ratio:.1f}x</div>
            <small>{eval_report.alert_reduction_pct:.1f}% reduction</small>
        </div>
        <div class="card">
            <div class="card-title">Correlated Incidents</div>
            <div class="card-value">{eval_report.total_incidents}</div>
            <small>From {eval_report.raw_alarms} raw alarms</small>
        </div>
        <div class="card">
            <div class="card-title">Incident Recall / F1</div>
            <div class="card-value" style="color: #38bdf8;">{eval_report.incident_recall * 100.0:.0f}% / {eval_report.incident_f1:.2f}</div>
        </div>
        <div class="card">
            <div class="card-title">Mean Step Latency</div>
            <div class="card-value">{eval_report.mean_step_latency_ms:.2f} ms</div>
            <small>{eval_report.throughput_steps_per_s:.0f} steps/s</small>
        </div>
    </div>

    <h2>Correlated Security Incidents</h2>
    <table>
        <thead>
            <tr>
                <th>Incident ID</th>
                <th>Peak Severity</th>
                <th>Start Time</th>
                <th>Duration</th>
                <th>Peak Evidence</th>
                <th>Alarms</th>
                <th>Subsystems</th>
                <th>Advisory Guidance</th>
            </tr>
        </thead>
        <tbody>
            {incidents_rows if incidents_rows else "<tr><td colspan='8'>No security incidents detected. Nominal operation.</td></tr>"}
        </tbody>
    </table>
</body>
</html>
"""
    with open(p, "w", encoding="utf-8") as f:
        f.write(html)
