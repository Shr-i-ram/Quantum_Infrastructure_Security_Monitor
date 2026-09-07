"""Telemetry windowing.

Events are partitioned into fixed-size windows by pulse ID (not by number of
detected events — QKD systems experience loss, so detected-event counts vary
while pulse counts are fixed):

    pulses [k*window_size, (k+1)*window_size)  ->  window k

Every window has a stable (run_id, window_id) combination. The last window of
a run may be partial; its features are computed over the pulses actually
present and ``n_pulses`` records the true count.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from qism.config import QKDConfig
from qism.telemetry.features import window_features


class WindowAggregator:
    """Computes window feature rows from event DataFrames."""

    def __init__(self, config: QKDConfig):
        self.config = config
        self.window_size = config.telemetry.window_size
        self.max_photon_number = config.source.max_photon_number
        self.hist_bins = config.telemetry.timing_histogram_bins

    def aggregate(self, events: pd.DataFrame) -> dict[str, Any]:
        """Feature row for one complete window (from its events)."""
        cfg = self.config
        features = window_features(
            events,
            window_size=self.window_size,
            max_photon_number=self.max_photon_number,
            timing_histogram_bins=self.hist_bins,
            mu_signal=cfg.source.mu_signal,
            mu_decoy=cfg.source.mu_decoy,
            mu_vacuum=cfg.source.mu_vacuum,
            repetition_rate_hz=cfg.protocol.repetition_rate_hz,
            interarrival_histogram_bins=cfg.telemetry.interarrival_histogram_bins,
            interarrival_histogram_scale=cfg.telemetry.interarrival_histogram_scale,
            fano_sub_bins=cfg.telemetry.fano_sub_bins,
        )
        window_id = int(events["window_id"].iloc[0])
        run_id = str(events["run_id"].iloc[0])
        scenario_id = str(events["scenario_id"].iloc[0])
        start = int(events["pulse_id"].min())
        end = int(events["pulse_id"].max())

        # Ground truth attack labels (separated from observable features)
        has_attack_col = "attack_active" in events.columns
        if has_attack_col:
            import numpy as np
            from collections import Counter
            atk_active = events["attack_active"].to_numpy(dtype=bool)
            attack_present = bool(np.any(atk_active))
            attack_fraction = float(np.mean(atk_active)) if len(events) else 0.0
            if attack_present:
                types = events.loc[atk_active, "attack_type"].dropna().tolist()
                dominant_type = Counter(types).most_common(1)[0][0] if types else "none"
                if "attack_phase" in events.columns:
                    phases = events.loc[atk_active, "attack_phase"].dropna().tolist()
                    dominant_phase = Counter(phases).most_common(1)[0][0] if phases else "active"
                else:
                    dominant_phase = "active"
                mean_sev = (
                    float(events.loc[atk_active, "attack_strength"].mean())
                    if "attack_strength" in events.columns
                    else 0.0
                )
                ids = [i for i in events.loc[atk_active, "attack_id"].tolist() if i is not None]
                dominant_id = ids[0] if ids else None
            else:
                dominant_type = "none"
                dominant_phase = "normal"
                mean_sev = 0.0
                dominant_id = "none"
        else:
            attack_present = False
            dominant_type = "none"
            dominant_phase = "normal"
            attack_fraction = 0.0
            mean_sev = 0.0
            dominant_id = "none"

        gt_attack = {
            "ground_truth_attack_active": attack_present,
            "ground_truth_attack_type": dominant_type,
            "ground_truth_attack_phase": dominant_phase,
            "ground_truth_attack_fraction": attack_fraction,
            "ground_truth_mean_severity": mean_sev,
            "ground_truth_attack_id": dominant_id,
        }

        return {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "window_id": window_id,
            "window_start_pulse": start,
            "window_end_pulse": end,
            **features,
            **gt_attack,
        }

    def aggregate_partial(self, events: pd.DataFrame) -> dict[str, Any] | None:
        """Aggregate a trailing partial window (used at run finalization)."""
        if events is None or len(events) == 0:
            return None
        return self.aggregate(events)


def make_telemetry_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert aggregated window rows into a DataFrame."""
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
