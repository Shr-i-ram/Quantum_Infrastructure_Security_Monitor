"""Sensor observation models: noise, ADC quantization, missingness, and multi-rate sampling (V0.4).

Models the difference between the true underlying physical state and the observable
sensor telemetry collected by a monitoring infrastructure:
1. True physical state x(t).
2. Sensor noise: eps_s ~ N(0, sigma_s^2).
3. ADC / display quantization: x_q = Delta_q * round((x + eps_s) / Delta_q).
4. Missingness / dropout: M ~ Bernoulli(p_missing) -> NaN.
5. Multi-rate sampling: holds last sampled value for sampling_interval_s.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
from numpy.random import Generator


class PhysicalSensor:
    """Simulates a physical sensor observing an underlying continuous quantity.

    Parameters
    ----------
    label : str
        Sensor identifier (e.g. "temperature_sensor", "humidity_sensor").
    unit : str
        Physical unit (e.g. "C", "%", "g", "lux").
    noise_std : float
        Standard deviation of Gaussian sensor measurement noise (sigma_s).
    quantization_step : float | None
        Finite sensor resolution / quantization step Delta_q (e.g. 0.1 C, 1 lux).
        If None or <= 0, quantization is disabled.
    missing_probability : float
        Probability p_missing of sensor dropout / missing value [0, 1].
    sampling_interval_s : float
        Time interval between physical sensor conversions / updates [s].
        A value of 0.0 means the sensor updates continuously.
    min_reading : float | None
        Hard minimum sensor reading (ADC lower clamp).
    max_reading : float | None
        Hard maximum sensor reading (ADC saturation clamp).
    """

    def __init__(
        self,
        label: str,
        unit: str = "",
        noise_std: float = 0.0,
        quantization_step: float | None = None,
        missing_probability: float = 0.0,
        sampling_interval_s: float = 0.0,
        min_reading: float | None = None,
        max_reading: float | None = None,
    ) -> None:
        self.label = label
        self.unit = unit
        self.noise_std = max(0.0, float(noise_std))
        self.quantization_step = (
            float(quantization_step) if quantization_step is not None and quantization_step > 0.0 else None
        )
        self.missing_probability = max(0.0, min(1.0, float(missing_probability)))
        self.sampling_interval_s = max(0.0, float(sampling_interval_s))
        self.min_reading = float(min_reading) if min_reading is not None else None
        self.max_reading = float(max_reading) if max_reading is not None else None

        self.last_sample_time = -float("inf")
        self.last_reading = float("nan")

    def measure(self, true_val: float, t: float, rng: Generator) -> float:
        """Sample the sensor reading at time t given true physical value ``true_val``.

        Respects sampling_interval_s: if elapsed time since last sample is less
        than sampling_interval_s, returns the held previous reading (zero-order hold).
        """
        if self.sampling_interval_s > 0.0 and self.last_sample_time != -float("inf"):
            if t - self.last_sample_time < self.sampling_interval_s:
                return self.last_reading

        # Sensor dropout / missingness
        if self.missing_probability > 0.0 and float(rng.random()) < self.missing_probability:
            self.last_reading = float("nan")
            self.last_sample_time = t
            return float("nan")

        # Gaussian measurement noise
        noisy_val = true_val
        if self.noise_std > 0.0:
            noisy_val += float(rng.normal(0.0, self.noise_std))

        # Quantization / ADC resolution
        if self.quantization_step is not None and self.quantization_step > 0.0:
            noisy_val = self.quantization_step * round(noisy_val / self.quantization_step)

        # ADC saturation clamps
        if self.min_reading is not None and noisy_val < self.min_reading:
            noisy_val = self.min_reading
        if self.max_reading is not None and noisy_val > self.max_reading:
            noisy_val = self.max_reading

        self.last_reading = noisy_val
        self.last_sample_time = t
        return noisy_val

    def get_state(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "unit": self.unit,
            "noise_std": self.noise_std,
            "quantization_step": self.quantization_step,
            "missing_probability": self.missing_probability,
            "sampling_interval_s": self.sampling_interval_s,
            "min_reading": self.min_reading,
            "max_reading": self.max_reading,
            "last_sample_time": self.last_sample_time,
            "last_reading": self.last_reading,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        self.label = state["label"]
        self.unit = state.get("unit", "")
        self.noise_std = float(state["noise_std"])
        self.quantization_step = (
            float(state["quantization_step"]) if state.get("quantization_step") is not None else None
        )
        self.missing_probability = float(state["missing_probability"])
        self.sampling_interval_s = float(state["sampling_interval_s"])
        self.min_reading = float(state["min_reading"]) if state.get("min_reading") is not None else None
        self.max_reading = float(state["max_reading"]) if state.get("max_reading") is not None else None
        self.last_sample_time = float(state["last_sample_time"])
        self.last_reading = float(state["last_reading"])
