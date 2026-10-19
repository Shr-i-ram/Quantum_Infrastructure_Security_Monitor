"""Operational temporal alarm state machine with persistence, hysteresis, cooldown, and minimum alarm duration (V0.7).

Transforms noisy, point-wise anomaly scores into operationally robust alarms.
Filters out transient single-window noise spikes while maintaining rapid alert
escalation for sustained physical attacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import numpy as np


class OperationalState(str, Enum):
    """Operational monitoring states."""
    NORMAL = "NORMAL"     # Healthy baseline operation
    SUSPECT = "SUSPECT"   # Anomaly detected, evaluating persistence
    ALARM = "ALARM"       # Confirmed persistent anomaly alert
    COOLDOWN = "COOLDOWN" # Post-alarm stabilization period


# Convenient alias
AlarmState = OperationalState


@dataclass
class AlarmConfig:
    """Configuration for temporal operational alarm state machine."""
    tau_high: float = 0.0                      # Primary trigger threshold
    tau_low: float | None = None         # Hysteresis recovery threshold (default 0.90 * tau_high)
    persistence_k: int = 2               # Required anomalous observations K
    persistence_n: int = 3               # In the last N observations (K-of-N logic)
    cooldown_windows: int = 0            # Cooldown duration after recovery
    min_alarm_duration: int = 1          # Minimum windows an ALARM state persists once triggered

    def __init__(
        self,
        tau_high: float | None = None,
        tau_low: float | None = None,
        persistence_k: int | None = None,
        persistence_n: int | None = None,
        cooldown_windows: int | None = None,
        min_alarm_duration: int = 1,
        threshold: float | None = None,
        recovery_threshold: float | None = None,
        k_consecutive: int | None = None,
        n_window: int | None = None,
        cooldown_steps: int | None = None,
        **kwargs: Any,
    ):
        high = tau_high if tau_high is not None else (threshold if threshold is not None else 0.0)
        low = tau_low if tau_low is not None else recovery_threshold
        k = persistence_k if persistence_k is not None else (k_consecutive if k_consecutive is not None else 2)
        n = persistence_n if persistence_n is not None else (n_window if n_window is not None else 3)
        cd = cooldown_windows if cooldown_windows is not None else (cooldown_steps if cooldown_steps is not None else 0)

        self.tau_high = float(high)
        self.tau_low = float(low) if low is not None else None
        self.persistence_k = int(k)
        self.persistence_n = int(n)
        self.cooldown_windows = int(cd)
        self.min_alarm_duration = int(min_alarm_duration)


@dataclass
class AlarmTransitionRecord:
    """Historical record of an alarm transition or state."""
    window_idx: int
    score: float
    state: OperationalState
    raw_alarm: bool
    filtered_alarm: bool
    consecutive_high_count: int
    recent_k_of_n_count: int
    duration_in_state: int
    timestamp_s: float | None = None
    scenario_id: str = "default"
    run_id: str = "default"


# Convenient alias
AlarmOutput = AlarmTransitionRecord


class TemporalAlarmStateMachine:
    """Stateful operational alarm generator supporting K-of-N persistence, hysteresis, and cooldown."""

    def __init__(self, config: AlarmConfig):
        self.config = config
        self.tau_high = float(config.tau_high)
        self.tau_low = float(config.tau_low) if config.tau_low is not None else float(self.tau_high * 0.90)
        self.k = max(1, config.persistence_k)
        self.n = max(self.k, config.persistence_n)
        self.cooldown_windows = max(0, config.cooldown_windows)
        self.min_duration = max(1, config.min_alarm_duration)

        # State tracking
        self.current_state = OperationalState.NORMAL
        self.history_scores: list[float] = []
        self.history_raw_alarms: list[bool] = []
        self.state_duration: int = 0
        self.alarm_duration: int = 0
        self.cooldown_counter: int = 0

    def update(self, score: float, **kwargs: Any) -> AlarmTransitionRecord:
        """Alias for step() for unified streaming interface."""
        return self.step(score, **kwargs)

    def reset(self) -> None:
        """Reset state machine for an independent simulation run."""
        self.current_state = OperationalState.NORMAL
        self.history_scores.clear()
        self.history_raw_alarms.clear()
        self.state_duration = 0
        self.alarm_duration = 0
        self.cooldown_counter = 0

    def step(
        self,
        score: float,
        window_idx: int = 0,
        timestamp_s: float | None = None,
        scenario_id: str = "default",
        run_id: str = "default",
    ) -> AlarmTransitionRecord:
        """Process a single incoming telemetry window anomaly score.

        Returns:
            AlarmTransitionRecord describing state, raw alarm, and filtered alarm.
        """
        score = float(score)
        raw_alarm = bool(score > self.tau_high)

        # Update circular sliding window of recent raw alarms
        self.history_scores.append(score)
        self.history_raw_alarms.append(raw_alarm)
        if len(self.history_raw_alarms) > self.n:
            self.history_raw_alarms.pop(0)
            self.history_scores.pop(0)

        recent_k_count = sum(self.history_raw_alarms)
        consecutive_high = 0
        for val in reversed(self.history_raw_alarms):
            if val:
                consecutive_high += 1
            else:
                break

        prev_state = self.current_state

        if self.current_state == OperationalState.NORMAL:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                self.current_state = OperationalState.COOLDOWN
            elif raw_alarm:
                if self.k == 1:
                    self.current_state = OperationalState.ALARM
                    self.alarm_duration = 1
                else:
                    self.current_state = OperationalState.SUSPECT

        elif self.current_state == OperationalState.COOLDOWN:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
            else:
                if raw_alarm:
                    self.current_state = OperationalState.SUSPECT
                else:
                    self.current_state = OperationalState.NORMAL

        elif self.current_state == OperationalState.SUSPECT:
            # Check K-of-N persistence condition
            if recent_k_count >= self.k:
                self.current_state = OperationalState.ALARM
                self.alarm_duration = 1
            elif score < self.tau_low and not raw_alarm:
                # De-escalate back to normal if score falls below recovery threshold
                self.current_state = OperationalState.NORMAL

        elif self.current_state == OperationalState.ALARM:
            self.alarm_duration += 1
            # Check recovery condition: must satisfy minimum duration AND fall below tau_low
            can_recover = self.alarm_duration >= self.min_duration
            if can_recover and score < self.tau_low:
                if self.cooldown_windows > 0:
                    self.current_state = OperationalState.COOLDOWN
                    self.cooldown_counter = self.cooldown_windows
                else:
                    self.current_state = OperationalState.NORMAL
                self.alarm_duration = 0

        # Duration in state tracking
        if self.current_state == prev_state:
            self.state_duration += 1
        else:
            self.state_duration = 1

        is_filtered_alarm = (self.current_state == OperationalState.ALARM)

        return AlarmTransitionRecord(
            window_idx=window_idx,
            score=score,
            state=self.current_state,
            raw_alarm=raw_alarm,
            filtered_alarm=is_filtered_alarm,
            consecutive_high_count=consecutive_high,
            recent_k_of_n_count=recent_k_count,
            duration_in_state=self.state_duration,
            timestamp_s=timestamp_s,
            scenario_id=scenario_id,
            run_id=run_id,
        )

    def process_sequence(
        self,
        scores: np.ndarray,
        timestamps_s: np.ndarray | None = None,
        scenario_id: str = "default",
        run_id: str = "default",
    ) -> tuple[np.ndarray, np.ndarray, list[AlarmTransitionRecord]]:
        """Process an entire trajectory of scores.

        Returns:
            raw_alarms: 1D binary array (score > tau_high).
            filtered_alarms: 1D binary array (operational ALARM state).
            records: List of AlarmTransitionRecord objects.
        """
        self.reset()
        n = len(scores)
        raw_alarms = np.zeros(n, dtype=np.int64)
        filtered_alarms = np.zeros(n, dtype=np.int64)
        records: list[AlarmTransitionRecord] = []

        for i, s in enumerate(scores):
            t_s = float(timestamps_s[i]) if timestamps_s is not None else None
            rec = self.step(s, window_idx=i, timestamp_s=t_s, scenario_id=scenario_id, run_id=run_id)
            raw_alarms[i] = int(rec.raw_alarm)
            filtered_alarms[i] = int(rec.filtered_alarm)
            records.append(rec)

        return raw_alarms, filtered_alarms, records
