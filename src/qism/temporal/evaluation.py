"""Dual sample-level and event-level evaluation framework for temporal security monitoring (V0.7).

Provides:
- Sample-level classification and curve metrics (Precision, Recall, F1, FPR, AUROC, AUPRC)
- Event-level evaluation (Episode Detection Rate, Detection Delay distribution, Alarm Duration)
- Operational false-alarm burden (False Alarm Episodes, False Alarms per Simulated Hour, MTBFA)
- Score distribution separation ratio
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from qism.ml.evaluation.metrics import calculate_classification_metrics


@dataclass
class EventEvaluationRecord:
    """Detailed record of an attack episode evaluation."""
    scenario_id: str
    run_id: str
    episode_index: int
    attack_type: str
    attack_severity: float
    start_pulse: int
    end_pulse: int
    start_time_s: float
    end_time_s: float
    detected: bool
    detection_delay_pulses: int | None
    detection_delay_s: float | None
    alarms_count: int
    alarm_duration_windows: int


@dataclass
class TemporalEvaluationSummary:
    """Consolidated summary of sample-level, event-level, and operational metrics."""
    # Sample-level metrics (raw and filtered)
    raw_window_metrics: dict[str, Any]
    filtered_window_metrics: dict[str, Any]
    curve_metrics: dict[str, Any]

    # Event-level metrics
    total_attack_episodes: int
    detected_attack_episodes: int
    missed_attack_episodes: int
    episode_detection_rate: float
    detection_delay_summary: dict[str, float | None]

    # Operational false-alarm metrics
    false_alarm_episodes: int
    normal_observation_hours: float
    false_alarms_per_hour: float
    mean_time_between_false_alarms_s: float | None

    # Anomaly score separation
    score_separation_ratio: float
    scores_summary: dict[str, float]

    # Episode records
    episode_records: list[EventEvaluationRecord] = field(default_factory=list)


# Convenient aliases
EpisodeMetadata = EventEvaluationRecord
EvaluationMetrics = TemporalEvaluationSummary


class TemporalEvaluator:

    """Evaluates temporal anomaly detectors across discrete events and continuous timelines."""

    def __init__(self, repetition_rate_hz: float = 1_000_000.0, default_pulses_per_window: int = 2000):
        self.repetition_rate_hz = repetition_rate_hz
        self.default_pulses_per_window = default_pulses_per_window

    def evaluate(
        self,
        scores: np.ndarray,
        raw_alarms: np.ndarray,
        filtered_alarms: np.ndarray,
        y_true_active: np.ndarray,
        y_meta_df: pd.DataFrame | None = None,
        timestamps_s: np.ndarray | None = None,
    ) -> TemporalEvaluationSummary:
        """Execute comprehensive temporal and operational evaluation."""
        n_samples = len(scores)
        y_true = np.asarray(y_true_active, dtype=int).ravel()
        raw_pred = np.asarray(raw_alarms, dtype=int).ravel()
        filt_pred = np.asarray(filtered_alarms, dtype=int).ravel()

        # 1. Sample-level classification metrics
        raw_m = calculate_classification_metrics(y_true, raw_pred)
        filt_m = calculate_classification_metrics(y_true, filt_pred)

        # 2. Continuous curve metrics (AUROC, AUPRC)
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))

        if n_pos > 0 and n_neg > 0:
            try:
                auroc = float(roc_auc_score(y_true, scores))
            except Exception:
                auroc = 0.5
            try:
                auprc = float(average_precision_score(y_true, scores))
            except Exception:
                auprc = float(n_pos / n_samples)
        else:
            auroc = 0.5
            auprc = float(n_pos / max(1, n_samples))

        curve_m = {"auroc": auroc, "auprc": auprc}

        # 3. Score distribution separation ratio
        norm_scores = scores[y_true == 0]
        atk_scores = scores[y_true == 1]
        if len(norm_scores) > 0 and len(atk_scores) > 0:
            med_diff = float(np.median(atk_scores) - np.median(norm_scores))
            std_norm = float(np.std(norm_scores))
            score_sep = float(med_diff / (std_norm + 1e-6))
        else:
            score_sep = 0.0

        scores_summary = {
            "min": float(np.min(scores)) if len(scores) else 0.0,
            "max": float(np.max(scores)) if len(scores) else 0.0,
            "mean": float(np.mean(scores)) if len(scores) else 0.0,
            "std": float(np.std(scores)) if len(scores) else 0.0,
            "normal_median": float(np.median(norm_scores)) if len(norm_scores) else 0.0,
            "attack_median": float(np.median(atk_scores)) if len(atk_scores) else 0.0,
        }

        # 4. Event-level episode evaluation
        # Group by trajectory (scenario_id, run_id)
        if y_meta_df is not None and "scenario_id" in y_meta_df.columns and "run_id" in y_meta_df.columns:
            traj_groups = y_meta_df.groupby(["scenario_id", "run_id"])
        else:
            # Fallback to single trajectory
            temp_df = pd.DataFrame({"idx": range(n_samples)})
            traj_groups = [("default", temp_df)]

        episode_records: list[EventEvaluationRecord] = []
        total_false_alarms = 0
        total_normal_pulses = 0

        for traj_key, sub_df in traj_groups:
            indices = sub_df.index.to_numpy()
            sub_y_true = y_true[indices]
            sub_alarms = filt_pred[indices]

            scen_name = str(traj_key[0]) if isinstance(traj_key, tuple) else "default"
            run_name = str(traj_key[1]) if isinstance(traj_key, tuple) else "default"

            # Recover pulse boundaries if present
            if y_meta_df is not None and "window_start_pulse" in y_meta_df.columns:
                sub_start_pulses = y_meta_df.loc[indices, "window_start_pulse"].to_numpy()
            else:
                sub_start_pulses = np.arange(len(indices)) * self.default_pulses_per_window

            if y_meta_df is not None and "window_end_pulse" in y_meta_df.columns:
                sub_end_pulses = y_meta_df.loc[indices, "window_end_pulse"].to_numpy()
            else:
                sub_end_pulses = sub_start_pulses + (self.default_pulses_per_window - 1)

            # Extract attack metadata if present
            sub_atk_types = (
                y_meta_df.loc[indices, "attack_type"].to_numpy()
                if (y_meta_df is not None and "attack_type" in y_meta_df.columns)
                else ["unknown"] * len(indices)
            )
            sub_atk_sevs = (
                y_meta_df.loc[indices, "attack_severity"].to_numpy()
                if (y_meta_df is not None and "attack_severity" in y_meta_df.columns)
                else [0.0] * len(indices)
            )

            # Identify contiguous attack episodes
            in_attack = False
            ep_start = 0
            ep_idx = 0

            for i, act in enumerate(sub_y_true):
                if act and not in_attack:
                    in_attack = True
                    ep_start = i
                elif not act and in_attack:
                    in_attack = False
                    ep_rec = self._evaluate_single_episode(
                        ep_start, i - 1, ep_idx, scen_name, run_name,
                        sub_alarms, sub_start_pulses, sub_end_pulses,
                        sub_atk_types, sub_atk_sevs,
                    )
                    episode_records.append(ep_rec)
                    ep_idx += 1

            if in_attack:
                ep_rec = self._evaluate_single_episode(
                    ep_start, len(sub_y_true) - 1, ep_idx, scen_name, run_name,
                    sub_alarms, sub_start_pulses, sub_end_pulses,
                    sub_atk_types, sub_atk_sevs,
                )
                episode_records.append(ep_rec)

            # Identify false alarm episodes during normal operating intervals
            in_fa = False
            for alm, act in zip(sub_alarms, sub_y_true):
                if not act:
                    total_normal_pulses += self.default_pulses_per_window
                    if alm == 1 and not in_fa:
                        in_fa = True
                        total_false_alarms += 1
                    elif alm == 0 and in_fa:
                        in_fa = False
                else:
                    in_fa = False

        # Summary calculations
        n_episodes = len(episode_records)
        detected_eps = [r for r in episode_records if r.detected]
        n_det = len(detected_eps)
        n_miss = n_episodes - n_det
        det_rate = float(n_det / n_episodes) if n_episodes > 0 else 1.0

        delays = [r.detection_delay_s for r in detected_eps if r.detection_delay_s is not None]
        delays_pulses = [r.detection_delay_pulses for r in detected_eps if r.detection_delay_pulses is not None]
        if delays:
            delay_summary = {
                "mean_delay_s": float(np.mean(delays)),
                "median_delay_s": float(np.median(delays)),
                "p90_delay_s": float(np.percentile(delays, 90)),
                "p95_delay_s": float(np.percentile(delays, 95)),
                "max_delay_s": float(np.max(delays)),
                "mean_delay_pulses": float(np.mean(delays_pulses)) if delays_pulses else None,
                "median_delay_pulses": float(np.median(delays_pulses)) if delays_pulses else None,
            }
        else:
            delay_summary = {
                "mean_delay_s": None,
                "median_delay_s": None,
                "p90_delay_s": None,
                "p95_delay_s": None,
                "max_delay_s": None,
                "mean_delay_pulses": None,
                "median_delay_pulses": None,
            }


        normal_sec = float(total_normal_pulses / self.repetition_rate_hz)
        normal_hours = float(normal_sec / 3600.0)
        fa_per_hour = float(total_false_alarms / normal_hours) if normal_hours > 0 else 0.0
        mtbfa_s = float(normal_sec / total_false_alarms) if total_false_alarms > 0 else None

        return TemporalEvaluationSummary(
            raw_window_metrics=raw_m,
            filtered_window_metrics=filt_m,
            curve_metrics=curve_m,
            total_attack_episodes=n_episodes,
            detected_attack_episodes=n_det,
            missed_attack_episodes=n_miss,
            episode_detection_rate=det_rate,
            detection_delay_summary=delay_summary,
            false_alarm_episodes=total_false_alarms,
            normal_observation_hours=normal_hours,
            false_alarms_per_hour=fa_per_hour,
            mean_time_between_false_alarms_s=mtbfa_s,
            score_separation_ratio=score_sep,
            scores_summary=scores_summary,
            episode_records=episode_records,
        )

    def _evaluate_single_episode(
        self,
        start_idx: int,
        end_idx: int,
        ep_num: int,
        scen_id: str,
        run_id: str,
        alarms: np.ndarray,
        start_pulses: np.ndarray,
        end_pulses: np.ndarray,
        atk_types: np.ndarray,
        atk_sevs: np.ndarray,
    ) -> EventEvaluationRecord:
        """Analyze a single discrete attack episode."""
        ep_alarms = alarms[start_idx : end_idx + 1]
        alarm_hits = np.where(ep_alarms == 1)[0]
        detected = len(alarm_hits) > 0
        s_pulse = int(start_pulses[start_idx])
        e_pulse = int(end_pulses[end_idx])
        s_time = float(s_pulse / self.repetition_rate_hz)
        e_time = float(e_pulse / self.repetition_rate_hz)

        if detected:
            first_hit_offset = alarm_hits[0]
            first_hit_pulse = int(start_pulses[start_idx + first_hit_offset])
            delay_pulses = max(0, first_hit_pulse - s_pulse)
            delay_s = float(delay_pulses / self.repetition_rate_hz)
        else:
            delay_pulses = None
            delay_s = None

        return EventEvaluationRecord(
            scenario_id=scen_id,
            run_id=run_id,
            episode_index=ep_num,
            attack_type=str(atk_types[start_idx]),
            attack_severity=float(atk_sevs[start_idx]),
            start_pulse=s_pulse,
            end_pulse=e_pulse,
            start_time_s=s_time,
            end_time_s=e_time,
            detected=detected,
            detection_delay_pulses=delay_pulses,
            detection_delay_s=delay_s,
            alarms_count=int(np.sum(ep_alarms)),
            alarm_duration_windows=int(len(alarm_hits)),
        )
