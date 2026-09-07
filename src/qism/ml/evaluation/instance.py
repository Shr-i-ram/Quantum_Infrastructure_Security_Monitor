"""Attack-instance matching, detection delay, and operational false alarm rates (Requirements 74–81, 171–175, 252, 253)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class AttackInstanceEvaluator:
    """Evaluates detection performance at the discrete attack episode level."""

    def __init__(self, repetition_rate_hz: float = 1_000_000.0):
        self.repetition_rate_hz = repetition_rate_hz

    def evaluate_instances(
        self,
        df_eval: pd.DataFrame,
        alarm_col: str = "alarm",
    ) -> dict[str, Any]:
        """Match alarm sequences against ground-truth attack intervals.

        Computes:
        - Attack detection rate (percentage of attack episodes detected)
        - Missed attack rate
        - Detection delay (D = t_d - t_a) distribution: mean, median, P90, P95
        - False alarm episodes and rate per simulated operational hour
        """
        if df_eval.empty:
            return {
                "total_attack_episodes": 0,
                "detected_attack_episodes": 0,
                "missed_attack_episodes": 0,
                "episode_detection_rate": 0.0,
                "detection_delay_summary": {},
                "false_alarm_episodes": 0,
                "false_alarms_per_hour": 0.0,
                "episode_records": [],
            }

        group_cols = [c for c in ["scenario_id", "run_id"] if c in df_eval.columns]
        if not group_cols:
            groups = [("default", df_eval)]
        else:
            groups = df_eval.groupby(group_cols)

        episode_records: list[dict[str, Any]] = []
        total_false_alarms = 0
        total_normal_pulses = 0

        for g_id, g_df in groups:
            g_df = g_df.sort_values(by="window_id" if "window_id" in g_df.columns else g_df.index)
            atk_active = g_df["attack_active"].to_numpy(dtype=bool) if "attack_active" in g_df.columns else np.zeros(len(g_df), dtype=bool)
            alarms = g_df[alarm_col].to_numpy(dtype=int) if alarm_col in g_df.columns else np.zeros(len(g_df), dtype=int)
            start_pulses = g_df["window_start_pulse"].to_numpy() if "window_start_pulse" in g_df.columns else np.arange(len(g_df)) * 1000

            # 1. Identify true attack episodes (contiguous blocks of attack_active == True)
            in_attack = False
            ep_start_idx = 0
            episodes_in_run: list[dict[str, Any]] = []

            for i, active in enumerate(atk_active):
                if active and not in_attack:
                    in_attack = True
                    ep_start_idx = i
                elif not active and in_attack:
                    in_attack = False
                    episodes_in_run.append({
                        "start_idx": ep_start_idx,
                        "end_idx": i - 1,
                        "start_pulse": int(start_pulses[ep_start_idx]),
                        "end_pulse": int(start_pulses[i - 1]),
                        "attack_type": str(g_df.iloc[ep_start_idx].get("attack_type", "unknown")),
                        "attack_severity": float(g_df.iloc[ep_start_idx].get("attack_severity", 0.0)),
                    })

            if in_attack:
                episodes_in_run.append({
                    "start_idx": ep_start_idx,
                    "end_idx": len(atk_active) - 1,
                    "start_pulse": int(start_pulses[ep_start_idx]),
                    "end_pulse": int(start_pulses[-1]),
                    "attack_type": str(g_df.iloc[ep_start_idx].get("attack_type", "unknown")),
                    "attack_severity": float(g_df.iloc[ep_start_idx].get("attack_severity", 0.0)),
                })

            # 2. Match alarms to attack episodes
            for ep in episodes_in_run:
                # Alarms falling between start_idx and end_idx
                sub_alarms = alarms[ep["start_idx"] : ep["end_idx"] + 1]
                alarm_hits = np.where(sub_alarms == 1)[0]
                detected = len(alarm_hits) > 0
                if detected:
                    first_alarm_offset = alarm_hits[0]
                    first_alarm_pulse = int(start_pulses[ep["start_idx"] + first_alarm_offset])
                    delay_pulses = max(0, first_alarm_pulse - ep["start_pulse"])
                    delay_seconds = float(delay_pulses / self.repetition_rate_hz)
                else:
                    delay_pulses = None
                    delay_seconds = None

                episode_records.append({
                    "group": str(g_id),
                    "attack_type": ep["attack_type"],
                    "attack_severity": ep["attack_severity"],
                    "start_pulse": ep["start_pulse"],
                    "end_pulse": ep["end_pulse"],
                    "detected": detected,
                    "delay_pulses": delay_pulses,
                    "delay_seconds": delay_seconds,
                })

            # 3. Identify false alarm episodes (contiguous alarm blocks during normal periods)
            in_false_alarm = False
            for i, (alm, act) in enumerate(zip(alarms, atk_active)):
                if not act:
                    total_normal_pulses += 1000  # Default window pulse count
                    if alm == 1 and not in_false_alarm:
                        in_false_alarm = True
                        total_false_alarms += 1
                    elif alm == 0 and in_false_alarm:
                        in_false_alarm = False
                else:
                    in_false_alarm = False

        # Summary calculations
        n_attacks = len(episode_records)
        detected_attacks = [r for r in episode_records if r["detected"]]
        n_detected = len(detected_attacks)
        n_missed = n_attacks - n_detected
        det_rate = float(n_detected / n_attacks) if n_attacks > 0 else 1.0

        delays = [r["delay_seconds"] for r in detected_attacks if r["delay_seconds"] is not None]
        if delays:
            delay_summary = {
                "mean_delay_s": float(np.mean(delays)),
                "median_delay_s": float(np.median(delays)),
                "p90_delay_s": float(np.percentile(delays, 90)),
                "p95_delay_s": float(np.percentile(delays, 95)),
            }
        else:
            delay_summary = {
                "mean_delay_s": None,
                "median_delay_s": None,
                "p90_delay_s": None,
                "p95_delay_s": None,
            }

        total_normal_seconds = total_normal_pulses / self.repetition_rate_hz
        total_normal_hours = total_normal_seconds / 3600.0
        false_alarm_rate_per_hour = (
            float(total_false_alarms / total_normal_hours) if total_normal_hours > 0 else 0.0
        )

        return {
            "total_attack_episodes": n_attacks,
            "detected_attack_episodes": n_detected,
            "missed_attack_episodes": n_missed,
            "episode_detection_rate": det_rate,
            "detection_delay_summary": delay_summary,
            "false_alarm_episodes": total_false_alarms,
            "normal_observation_hours": total_normal_hours,
            "false_alarms_per_hour": false_alarm_rate_per_hour,
            "episode_records": episode_records,
        }
