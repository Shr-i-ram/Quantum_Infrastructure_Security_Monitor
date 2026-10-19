"""Metrics and comprehensive evaluation framework for concept drift and adaptive security (V0.8).

Computes:
1. Sample-level metrics: AUROC, AUPRC, F1, Recall, Precision, False Positive Rate (FAR)
2. Event-level attack metrics: Episode Detection Rate (EDR), Detection Delay (seconds and pulses)
3. Drift detection metrics: Drift Detection Delay, Drift Recall, False Drift Alarms
4. Adaptation safety metrics: Attack recall before vs after adaptation, Contamination events, Rollbacks
5. Operational efficiency: Average and P95 streaming latency, Adaptation acceptance rate
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics

from qism.drift.streaming import AdaptiveStreamingResult


@dataclass
class DriftBenchmarkMetrics:
    """Consolidated performance metrics for a V0.8 experiment run."""
    # Experiment metadata
    experiment_id: str
    scenario_type: str
    adaptation_policy: str
    total_steps: int

    # Security & Anomaly Detection Performance
    auroc: float
    auprc: float
    f1: float
    recall: float
    precision: float
    fpr: float  # False Positive Rate (FAR)
    episode_recall: float
    median_attack_delay_s: float
    attack_recall_before_adaptation: float | None = None
    attack_recall_after_adaptation: float | None = None
    attack_recall_during_drift: float | None = None

    # Drift Detection Performance
    drift_detection_delay_s: float | None = None
    drift_detected: bool = False
    false_drift_alarms: int = 0

    # Adaptation Dynamics & Stability
    adaptations_attempted: int = 0
    adaptations_accepted: int = 0
    adaptations_rejected: int = 0
    rollbacks_executed: int = 0
    contamination_events: int = 0
    final_model_version: str = "model_v001"

    # Computational Cost
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    max_latency_ms: float = 0.0


class DriftEvaluator:
    """Evaluates adaptive streaming inference results against ground-truth labels."""

    def __init__(self, step_duration_s: float = 0.002, pulses_per_window: int = 2000):
        self.step_duration_s = step_duration_s
        self.pulses_per_window = pulses_per_window

    def evaluate(
        self,
        results: list[AdaptiveStreamingResult],
        df_ground_truth: pd.DataFrame,
        experiment_id: str = "exp_01",
        scenario_type: str = "stationary",
        adaptation_policy: str = "safe",
    ) -> DriftBenchmarkMetrics:
        """Compute full suite of security, drift, and adaptation metrics."""
        N = len(results)
        if N == 0:
            raise ValueError("No streaming results provided for evaluation.")

        scores = np.array([r.anomaly_score for r in results], dtype=np.float64)
        alarms = np.array([int(r.filtered_alarm) for r in results], dtype=int)
        drift_scores = np.array([r.drift_score for r in results], dtype=np.float64)
        drift_flags = np.array([int(r.is_drift) for r in results], dtype=int)
        timestamps = np.array(
            [r.timestamp_s if r.timestamp_s is not None else i * self.step_duration_s for i, r in enumerate(results)],
            dtype=np.float64,
        )

        y_attack = df_ground_truth["ground_truth_is_attack"].values[:N].astype(int)
        y_drift = df_ground_truth["ground_truth_is_drift"].values[:N].astype(int)

        # 1. Sample-level classification metrics
        # Replace non-finite scores safely
        clean_scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)

        if len(np.unique(y_attack)) > 1:
            try:
                auroc_val = float(metrics.roc_auc_score(y_attack, clean_scores))
            except Exception:
                auroc_val = 0.50
            try:
                precision_curve, recall_curve, _ = metrics.precision_recall_curve(y_attack, clean_scores)
                auprc_val = float(metrics.auc(recall_curve, precision_curve))
            except Exception:
                auprc_val = float(np.mean(y_attack))
        else:
            auroc_val = 1.0 if y_attack[0] == 0 else 0.5
            auprc_val = 1.0 if y_attack[0] == 0 else 0.5

        # Confusion matrix
        n_pos = int(np.sum(y_attack == 1))
        n_neg = int(np.sum(y_attack == 0))

        tp = int(np.sum((alarms == 1) & (y_attack == 1)))
        fp = int(np.sum((alarms == 1) & (y_attack == 0)))
        tn = int(np.sum((alarms == 0) & (y_attack == 0)))
        fn = int(np.sum((alarms == 0) & (y_attack == 1)))

        recall_val = float(tp / n_pos) if n_pos > 0 else 0.0
        precision_val = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        fpr_val = float(fp / n_neg) if n_neg > 0 else 0.0
        f1_val = (
            float(2 * precision_val * recall_val / (precision_val + recall_val))
            if (precision_val + recall_val) > 0
            else 0.0
        )

        # 2. Event-level attack metrics (Episode Detection Rate & Detection Delay)
        episode_recall = 0.0
        attack_delays: list[float] = []

        if n_pos > 0:
            # Find contiguous attack episodes
            in_attack = False
            ep_start = 0
            for i in range(N):
                if y_attack[i] == 1 and not in_attack:
                    in_attack = True
                    ep_start = i
                elif (y_attack[i] == 0 or i == N - 1) and in_attack:
                    in_attack = False
                    ep_end = i if y_attack[i] == 0 else i + 1
                    ep_alarms = alarms[ep_start:ep_end]
                    if np.any(ep_alarms == 1):
                        episode_recall = 1.0
                        first_alarm_idx = ep_start + int(np.argmax(ep_alarms == 1))
                        delay_s = float(timestamps[first_alarm_idx] - timestamps[ep_start])
                        attack_delays.append(max(0.0, delay_s))
                    else:
                        attack_delays.append(float((ep_end - ep_start) * self.step_duration_s))

        med_attack_delay = float(np.median(attack_delays)) if attack_delays else 0.0

        # 3. Drift detection metrics
        drift_detected = False
        drift_delay_s = None
        false_drift_alarms = 0

        drift_indices = np.where(y_drift == 1)[0]
        if len(drift_indices) > 0:
            first_drift_true_step = int(drift_indices[0])
            first_drift_flag_after = np.where((drift_flags == 1) & (np.arange(N) >= first_drift_true_step))[0]
            if len(first_drift_flag_after) > 0:
                drift_detected = True
                drift_detected_step = int(first_drift_flag_after[0])
                drift_delay_s = float((drift_detected_step - first_drift_true_step) * self.step_duration_s)

        # False drift alarms in stationary normal steps
        stationary_normal_mask = (y_drift == 0) & (y_attack == 0)
        false_drift_alarms = int(np.sum((drift_flags == 1) & stationary_normal_mask))

        # 4. Security retention: attack recall before vs after vs during drift
        rec_before = None
        rec_after = None
        rec_during = None

        if n_pos > 0 and len(drift_indices) > 0:
            d_start = int(drift_indices[0])
            d_end = int(drift_indices[-1])

            before_mask = (y_attack == 1) & (np.arange(N) < d_start)
            during_mask = (y_attack == 1) & (np.arange(N) >= d_start) & (np.arange(N) <= d_end)
            after_mask = (y_attack == 1) & (np.arange(N) > d_end)

            if np.sum(before_mask) > 0:
                rec_before = float(np.sum((alarms == 1) & before_mask) / np.sum(before_mask))
            if np.sum(during_mask) > 0:
                rec_during = float(np.sum((alarms == 1) & during_mask) / np.sum(during_mask))
            if np.sum(after_mask) > 0:
                rec_after = float(np.sum((alarms == 1) & after_mask) / np.sum(after_mask))

        # 5. Adaptation metrics from results
        adapt_promoted = sum(1 for r in results if r.adaptation_decision.value == "PROMOTED")
        adapt_rejected_anom = sum(1 for r in results if r.adaptation_decision.value == "REJECTED_ANOMALY")
        adapt_rejected_val = sum(1 for r in results if r.adaptation_decision.value == "REJECTED_VALIDATION_FAILED")
        adapt_rejected_cd = sum(1 for r in results if r.adaptation_decision.value == "REJECTED_COOLDOWN")
        rollbacks = sum(1 for r in results if r.adaptation_decision.value == "ROLLBACK_EXECUTED")

        # Contamination events: adaptations promoted during active attacks
        contamination_events = 0
        for i, r in enumerate(results):
            if r.adaptation_decision.value == "PROMOTED" and y_attack[i] == 1:
                contamination_events += 1

        final_ver = results[-1].active_model_version

        # 6. Computational latency profiling
        latencies = [r.computational_latency_ms for r in results if r.computational_latency_ms > 0]
        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
        max_lat = float(np.max(latencies)) if latencies else 0.0

        return DriftBenchmarkMetrics(
            experiment_id=experiment_id,
            scenario_type=scenario_type,
            adaptation_policy=adaptation_policy,
            total_steps=N,
            auroc=auroc_val,
            auprc=auprc_val,
            f1=f1_val,
            recall=recall_val,
            precision=precision_val,
            fpr=fpr_val,
            episode_recall=episode_recall,
            median_attack_delay_s=med_attack_delay,
            attack_recall_before_adaptation=rec_before,
            attack_recall_after_adaptation=rec_after,
            attack_recall_during_drift=rec_during,
            drift_detection_delay_s=drift_delay_s,
            drift_detected=drift_detected,
            false_drift_alarms=false_drift_alarms,
            adaptations_attempted=adapt_promoted + adapt_rejected_anom + adapt_rejected_val,
            adaptations_accepted=adapt_promoted,
            adaptations_rejected=adapt_rejected_anom + adapt_rejected_val + adapt_rejected_cd,
            rollbacks_executed=rollbacks,
            contamination_events=contamination_events,
            final_model_version=final_ver,
            mean_latency_ms=mean_lat,
            p95_latency_ms=p95_lat,
            max_latency_ms=max_lat,
        )
