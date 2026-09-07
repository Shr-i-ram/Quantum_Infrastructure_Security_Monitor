"""Alarm state machine, hysteresis filtering, and response interfaces (Requirements 167–170, 235, 236)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np


class AlarmState(str, Enum):
    """Operational alarm state machine phases."""
    NORMAL = "NORMAL"
    SUSPECT = "SUSPECT"
    ALARM = "ALARM"


class ResponseInterface(ABC):
    """Extensible interface for future automated QKD attack prevention and mitigation (V0.9)."""

    @abstractmethod
    def handle_alarm(
        self,
        timestamp: float | None,
        scenario_id: str,
        window_id: int,
        score: float,
        state: AlarmState,
    ) -> None:
        """Invoked upon an alarm state change or sustained alarm."""
        pass


class LoggingResponseHandler(ResponseInterface):
    """Default response handler recording alarm transitions for inspection."""

    def __init__(self):
        self.log: list[dict[str, Any]] = []

    def handle_alarm(
        self,
        timestamp: float | None,
        scenario_id: str,
        window_id: int,
        score: float,
        state: AlarmState,
    ) -> None:
        self.log.append({
            "timestamp": timestamp,
            "scenario_id": scenario_id,
            "window_id": window_id,
            "score": score,
            "state": state.value,
        })


class AlarmStateMachine:
    """Stateful post-processing layer converting raw continuous anomaly scores into operational alarms.

    Features:
    - Three-phase lifecycle: NORMAL -> SUSPECT -> ALARM -> NORMAL
    - Persistence filter: requires K consecutive anomalous windows before raising ALARM
    - Hysteresis: activation at tau_high, deactivation at tau_low
    """

    def __init__(
        self,
        tau_high: float,
        tau_low: float | None = None,
        consecutive_k: int = 2,
        response_handler: ResponseInterface | None = None,
    ):
        self.tau_high = float(tau_high)
        self.tau_low = float(tau_low) if tau_low is not None else float(tau_high * 0.90)
        self.consecutive_k = max(1, consecutive_k)
        self.response_handler = response_handler or LoggingResponseHandler()

        self.current_state = AlarmState.NORMAL
        self.consecutive_count = 0

    def reset(self) -> None:
        """Reset state machine between independent runs."""
        self.current_state = AlarmState.NORMAL
        self.consecutive_count = 0

    def step(
        self,
        score: float,
        scenario_id: str = "default",
        window_id: int = 0,
        timestamp: float | None = None,
    ) -> tuple[AlarmState, bool]:
        """Process a single incoming telemetry window score.

        Returns:
            (current_state, is_active_alarm)
        """
        score = float(score)

        if self.current_state == AlarmState.NORMAL:
            if score > self.tau_high:
                self.consecutive_count = 1
                if self.consecutive_count >= self.consecutive_k:
                    self.current_state = AlarmState.ALARM
                else:
                    self.current_state = AlarmState.SUSPECT
            else:
                self.consecutive_count = 0

        elif self.current_state == AlarmState.SUSPECT:
            if score > self.tau_high:
                self.consecutive_count += 1
                if self.consecutive_count >= self.consecutive_k:
                    self.current_state = AlarmState.ALARM
            elif score < self.tau_low:
                self.consecutive_count = 0
                self.current_state = AlarmState.NORMAL

        elif self.current_state == AlarmState.ALARM:
            if score < self.tau_low:
                self.consecutive_count = 0
                self.current_state = AlarmState.NORMAL
            else:
                self.consecutive_count += 1

        is_alarm = (self.current_state == AlarmState.ALARM)
        if is_alarm:
            self.response_handler.handle_alarm(
                timestamp=timestamp,
                scenario_id=scenario_id,
                window_id=window_id,
                score=score,
                state=self.current_state,
            )

        return self.current_state, is_alarm

    def process_sequence(
        self,
        scores: np.ndarray,
        scenario_id: str = "default",
    ) -> tuple[list[AlarmState], np.ndarray]:
        """Process an entire sequence of scores for evaluation."""
        self.reset()
        states = []
        alarms = np.zeros(len(scores), dtype=int)
        for i, s in enumerate(scores):
            st, is_alm = self.step(s, scenario_id=scenario_id, window_id=i)
            states.append(st)
            alarms[i] = int(is_alm)
        return states, alarms
