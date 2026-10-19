"""Offline evaluation, latency profiling, and operational burden metrics for QISM V0.9.

Compares operational monitoring inferences against offline ground-truth telemetry
strictly for post-hoc scientific evaluation (preserving the operational security boundary).

Calculates:
1. Operational metrics: Alert compression ratio, alert reduction rate, false incident rate
2. Event-level metrics: Incident precision, incident recall, incident F1, fragmentation index
3. Latency metrics: Telemetry-to-detection latency, telemetry-to-alert latency
4. Computational performance: Mean/p95 step latency and throughput
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from qism.monitor.types import Alert, Incident, MonitorRunResult


@dataclass
class GroundTruthEvent:
    """An offline ground truth episode (attack or drift) from simulation metadata."""
    event_type: str  # e.g. "intercept_resend", "channel_loss", "detector_blinding"
    start_time_s: float
    end_time_s: float
    start_step: int
    end_step: int


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report for a monitoring run."""
    total_steps: int
    total_duration_s: float
    # Operational Burden
    raw_alarms: int
    total_incidents: int
    total_alerts: int
    alert_compression_ratio: float
    alert_reduction_pct: float
    false_incident_count: int
    false_incident_rate_per_min: float
    # Event-Level Accuracy
    ground_truth_events_count: int
    detected_events_count: int
    incident_precision: float
    incident_recall: float
    incident_f1: float
    fragmentation_index: float
    # Latency (seconds and steps)
    mean_detection_latency_s: float | None
    mean_alert_latency_s: float | None
    mean_step_latency_ms: float
    p95_step_latency_ms: float
    throughput_steps_per_s: float
    # Summary Details
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "total_duration_s": self.total_duration_s,
            "raw_alarms": self.raw_alarms,
            "total_incidents": self.total_incidents,
            "total_alerts": self.total_alerts,
            "alert_compression_ratio": float(self.alert_compression_ratio),
            "alert_reduction_pct": float(self.alert_reduction_pct),
            "false_incident_count": self.false_incident_count,
            "false_incident_rate_per_min": float(self.false_incident_rate_per_min),
            "ground_truth_events_count": self.ground_truth_events_count,
            "detected_events_count": self.detected_events_count,
            "incident_precision": float(self.incident_precision),
            "incident_recall": float(self.incident_recall),
            "incident_f1": float(self.incident_f1),
            "fragmentation_index": float(self.fragmentation_index),
            "mean_detection_latency_s": self.mean_detection_latency_s,
            "mean_alert_latency_s": self.mean_alert_latency_s,
            "mean_step_latency_ms": float(self.mean_step_latency_ms),
            "p95_step_latency_ms": float(self.p95_step_latency_ms),
            "throughput_steps_per_s": float(self.throughput_steps_per_s),
            "details": self.details,
        }


def extract_ground_truth_events(df: pd.DataFrame) -> list[GroundTruthEvent]:
    """Extract discrete attack intervals from ground truth columns in evaluation DataFrame."""
    atk_col = None
    for candidate in ("attack_active", "ground_truth_attack_active", "ground_truth_is_attack", "is_attack"):
        if candidate in df.columns:
            atk_col = candidate
            break

    if atk_col is None:
        return []

    events: list[GroundTruthEvent] = []
    in_event = False
    start_step = 0
    t_start = 0.0
    atk_type = "unknown"

    t_col = "timestamp_s" if "timestamp_s" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    type_col = "attack_type" if "attack_type" in df.columns else ("ground_truth_attack_type" if "ground_truth_attack_type" in df.columns else None)

    for idx, row in df.iterrows():
        is_active = bool(row[atk_col])
        curr_t = float(row[t_col]) if t_col else float(idx) * 0.002

        if is_active and not in_event:
            in_event = True
            start_step = int(idx)
            t_start = curr_t
            atk_type = str(row.get(type_col, "attack")) if type_col else "attack"
        elif not is_active and in_event:
            in_event = False
            prev_t = float(df.iloc[idx - 1][t_col]) if t_col else float(idx - 1) * 0.002
            events.append(
                GroundTruthEvent(
                    event_type=atk_type,
                    start_time_s=t_start,
                    end_time_s=prev_t,
                    start_step=start_step,
                    end_step=int(idx - 1),
                )
            )

    if in_event:
        last_idx = len(df) - 1
        last_t = float(df.iloc[last_idx][t_col]) if t_col else float(last_idx) * 0.002
        events.append(
            GroundTruthEvent(
                event_type=atk_type,
                start_time_s=t_start,
                end_time_s=last_t,
                start_step=start_step,
                end_step=last_idx,
            )
        )

    return events


def evaluate_monitor_run(
    run_result: MonitorRunResult,
    ground_truth_df: pd.DataFrame,
) -> EvaluationReport:
    """Evaluate operational performance against offline ground truth."""
    gt_events = extract_ground_truth_events(ground_truth_df)
    incidents = run_result.all_incidents
    alerts = run_result.all_alerts
    raw_alarms = run_result.metrics.get("total_raw_alarms", 0)

    # 1. Operational Burden Metrics
    total_incidents = len(incidents)
    total_alerts = len(alerts)
    compression_ratio = float(raw_alarms / total_incidents) if total_incidents > 0 else 1.0
    alert_reduction_pct = float((1.0 - total_alerts / max(1, raw_alarms)) * 100.0) if raw_alarms > 0 else 0.0

    # 2. Event-Level Overlap & Accuracy
    true_incidents = 0
    false_incidents = 0
    detected_events = set()
    event_overlap_counts: dict[int, int] = {i: 0 for i in range(len(gt_events))}

    detection_delays_s: list[float] = []
    alert_delays_s: list[float] = []

    for inc in incidents:
        inc_start = inc.start_time_s
        inc_end = inc.end_time_s if inc.end_time_s is not None else inc_start + inc.duration_s

        matched_gt = False
        for e_idx, event in enumerate(gt_events):
            # Check for temporal overlap: max(start) <= min(end)
            if max(inc_start, event.start_time_s) <= min(inc_end, event.end_time_s) + 0.010:  # 10ms tolerance
                matched_gt = True
                detected_events.add(e_idx)
                event_overlap_counts[e_idx] += 1

                # Latency from event onset to incident start
                det_delay = max(0.0, inc_start - event.start_time_s)
                detection_delays_s.append(det_delay)
                break

        if matched_gt:
            true_incidents += 1
        else:
            false_incidents += 1

    # Match first alert per event
    for e_idx, event in enumerate(gt_events):
        matching_alerts = [
            a for a in alerts if a.timestamp_s >= event.start_time_s and a.timestamp_s <= event.end_time_s + 0.010
        ]
        if matching_alerts:
            first_alt = min(matching_alerts, key=lambda a: a.timestamp_s)
            alert_delays_s.append(max(0.0, first_alt.timestamp_s - event.start_time_s))

    # Precision, Recall, F1
    n_gt = len(gt_events)
    n_det_events = len(detected_events)

    precision = float(true_incidents / total_incidents) if total_incidents > 0 else (1.0 if n_gt == 0 else 0.0)
    recall = float(n_det_events / n_gt) if n_gt > 0 else (1.0 if total_incidents == 0 else 0.0)
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Fragmentation index: mean number of incidents per detected event
    frag_counts = [count for count in event_overlap_counts.values() if count > 0]
    frag_index = float(np.mean(frag_counts)) if frag_counts else (1.0 if n_gt == 0 else 0.0)

    # Durations & Rates
    total_duration_s = run_result.total_duration_s
    sim_duration_s = float(ground_truth_df["timestamp_s"].iloc[-1] - ground_truth_df["timestamp_s"].iloc[0]) if "timestamp_s" in ground_truth_df.columns and len(ground_truth_df) > 1 else len(ground_truth_df) * 0.002
    sim_mins = max(1e-4, sim_duration_s / 60.0)
    false_incident_rate_per_min = float(false_incidents / sim_mins)

    return EvaluationReport(
        total_steps=run_result.total_steps,
        total_duration_s=sim_duration_s,
        raw_alarms=raw_alarms,
        total_incidents=total_incidents,
        total_alerts=total_alerts,
        alert_compression_ratio=compression_ratio,
        alert_reduction_pct=alert_reduction_pct,
        false_incident_count=false_incidents,
        false_incident_rate_per_min=false_incident_rate_per_min,
        ground_truth_events_count=n_gt,
        detected_events_count=n_det_events,
        incident_precision=precision,
        incident_recall=recall,
        incident_f1=f1,
        fragmentation_index=frag_index,
        mean_detection_latency_s=float(np.mean(detection_delays_s)) if detection_delays_s else None,
        mean_alert_latency_s=float(np.mean(alert_delays_s)) if alert_delays_s else None,
        mean_step_latency_ms=run_result.metrics.get("mean_latency_ms", 0.0),
        p95_step_latency_ms=run_result.metrics.get("p95_latency_ms", 0.0),
        throughput_steps_per_s=run_result.metrics.get("throughput_steps_per_s", 0.0),
        details={
            "true_incidents": true_incidents,
            "false_incidents": false_incidents,
            "gt_events": [
                {"type": e.event_type, "start_s": e.start_time_s, "end_s": e.end_time_s} for e in gt_events
            ],
            "incidents_summary": [
                {
                    "id": inc.incident_id,
                    "state": inc.state.value,
                    "severity": inc.severity.value,
                    "evidence": inc.evidence_score,
                    "duration_s": inc.duration_s,
                }
                for inc in incidents
            ],
        },
    )
