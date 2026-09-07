"""Environment abstraction and state definitions for QISM V0.4.

    Environment.get_state(t)  ->  EnvironmentState at time t
    Environment.update(t)     ->  advance internal dynamics to time t
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qism.config import EnvironmentConfig


@dataclass
class EnvironmentState:
    """Snapshot of the operating environment at time t.

    Includes true physical variables as well as observable sensor readings.
    """

    label: str = "stable"
    temperature_c: float | None = None
    humidity_pct: float | None = None
    vibration_g: float | None = None
    background_light_lux: float | None = None
    background_photon_rate_hz: float | None = None
    polarization_drift_deg: float | None = None
    clock_offset_s: float | None = None
    clock_drift_ppm: float | None = None
    channel_loss_drift_db: float | None = None
    source_intensity_drift: float | None = None
    detector_efficiency_drift: float | None = None

    # Observable sensor telemetry (subject to sensor noise, ADC quantization, dropout)
    sensor_temperature_c: float | None = None
    sensor_humidity_pct: float | None = None
    sensor_vibration_g: float | None = None
    sensor_background_light_lux: float | None = None

    # Legacy fields
    bias_voltage_v: float | None = None
    optical_power_dbm: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "environment_state": self.label,
            "temperature_c": self.temperature_c,
            "humidity_pct": self.humidity_pct,
            "vibration_g": self.vibration_g,
            "background_light_lux": self.background_light_lux,
            "background_photon_rate_hz": self.background_photon_rate_hz,
            "polarization_drift_deg": self.polarization_drift_deg,
            "clock_offset_s": self.clock_offset_s,
            "clock_drift_ppm": self.clock_drift_ppm,
            "channel_loss_drift_db": self.channel_loss_drift_db,
            "source_intensity_drift": self.source_intensity_drift,
            "detector_efficiency_drift": self.detector_efficiency_drift,
            "sensor_temperature_c": self.sensor_temperature_c,
            "sensor_humidity_pct": self.sensor_humidity_pct,
            "sensor_vibration_g": self.sensor_vibration_g,
            "sensor_background_light_lux": self.sensor_background_light_lux,
            "bias_voltage_v": self.bias_voltage_v,
            "optical_power_dbm": self.optical_power_dbm,
        }
        d.update(self.extra)
        return d


class Environment(ABC):
    """Base class for operational/environmental variation models."""

    name: str = "base"

    @abstractmethod
    def get_state(self, t: float) -> EnvironmentState:
        """State of the environment at absolute time t [s]."""

    @abstractmethod
    def update(self, t: float) -> None:
        """Advance internal dynamics to time t [s]."""

    def get_runtime_state(self) -> dict[str, Any]:
        """Serialize internal state for checkpointing."""
        return {}

    def restore_runtime_state(self, state: dict[str, Any]) -> None:
        """Restore internal state from checkpoint."""
        pass


class StableEnvironment(Environment):
    """Constant environment: no drift, no variation (V0.1 default)."""

    name = "stable"

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self._state = EnvironmentState(label="stable")

    def get_state(self, t: float) -> EnvironmentState:
        return self._state

    def update(self, t: float) -> None:
        return None

    def get_runtime_state(self) -> dict[str, Any]:
        return {"type": "stable"}

    def restore_runtime_state(self, state: dict[str, Any]) -> None:
        pass


def make_environment(config: EnvironmentConfig) -> Environment:
    """Factory selecting the environment model from configuration."""
    if config.type == "stable":
        return StableEnvironment(config)
    elif config.type in ("dynamic", "ou_drift", "diurnal", "composite"):
        from qism.environment.dynamic import DynamicEnvironment

        return DynamicEnvironment(config)
    raise NotImplementedError(
        f"Environment model {config.type!r} is not implemented. Supported types: 'stable', 'dynamic'."
    )
