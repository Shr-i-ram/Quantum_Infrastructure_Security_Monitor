"""Event-level and operational metrics computation for PI-TQSVDD."""

from __future__ import annotations

from typing import Any
import numpy as np


def compute_event_level_metrics(
    predictions: list[int] | np.ndarray,
    labels: list[int] | np.ndarray,
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute incident-level detection metrics, delay to first alarm, and false alarm rates."""
    preds = np.asarray(predictions).astype(int)
    y_true = np.asarray(labels).astype(int)

    # Group records by scenario_id
    by_scen: dict[str, list[dict[str, Any]]] = {}
    for i, meta in enumerate(metadata):
        sc_id = meta.get("scenario_id", "default_scenario")
        by_scen.setdefault(sc_id, []).append({
            "pred": preds[i],
            "label": y_true[i],
            "window_id": meta.get("window_id", i),
            "attack_type": meta.get("attack_type", "normal"),
        })

    total_incidents = 0
    detected_incidents = 0
    delays = []
    normal_windows = 0
    false_alarms = 0

    for sc_id, items in by_scen.items():
        in_attack = False
        incident_detected = False
        attack_start_idx = 0

        for idx, it in enumerate(items):
            is_atk = (it["label"] == 1)
            is_alarm = (it["pred"] == 1)

            if is_atk and not in_attack:
                # Attack onset
                in_attack = True
                total_incidents += 1
                attack_start_idx = idx
                incident_detected = False

            if in_attack:
                if is_alarm and not incident_detected:
                    incident_detected = True
                    detected_incidents += 1
                    delays.append(idx - attack_start_idx)

                if not is_atk:
                    # Attack ended
                    in_attack = False

            else:
                normal_windows += 1
                if is_alarm:
                    false_alarms += 1

    incident_recall = float(detected_incidents / total_incidents) if total_incidents > 0 else 1.0
    mean_delay = float(np.mean(delays)) if delays else 0.0
    far = float(false_alarms / normal_windows) if normal_windows > 0 else 0.0

    return {
        "total_attack_incidents": total_incidents,
        "detected_attack_incidents": detected_incidents,
        "incident_recall": incident_recall,
        "mean_detection_delay_windows": mean_delay,
        "false_alarm_rate_per_normal_window": far,
        "total_false_alarms": false_alarms,
        "total_normal_windows": normal_windows,
    }
