"""Dynamic environment model with coupled physical processes and sensor telemetry (V0.4).

Models continuous, temporally correlated environmental variables:
- Temperature [C]: baseline, diurnal sinusoidal cycle, Ornstein-Uhlenbeck drift.
- Humidity [%]: relative humidity with OU drift in [0, 100]%.
- Vibration [g]: mechanical acceleration RMS with OU drift >= 0.
- Background optical light [lux / rate]: ambient light coupling to stray photon arrivals.
- Polarization drift [deg]: slow optical fiber birefringence / alignment drift.
- Clock drift [ppm / s]: oscillator frequency error and accumulated phase/time offset.
- Channel loss drift [dB]: fiber strain/thermal perturbation.
- Source intensity drift: laser driver fluctuation.
- Detector efficiency drift: SPAD sensitivity drift.

Also simulates the monitoring sensor suite (noise, quantization, missingness, multi-rate).
"""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.random import Generator

from qism.config import EnvironmentConfig
from qism.environment.base import Environment, EnvironmentState
from qism.environment.processes import EnvironmentalDrift
from qism.environment.sensors import PhysicalSensor


class DynamicEnvironment(Environment):
    """Dynamic operational environment with continuous physical processes and sensor suite."""

    name = "dynamic"

    def __init__(self, config: EnvironmentConfig, seed: int = 0) -> None:
        self.config = config
        self.t = 0.0
        self.clock_offset_s = 0.0

        # Physical drift processes
        tc = config.temperature
        self.temperature_proc = EnvironmentalDrift(
            baseline=tc.baseline,
            theta=tc.theta,
            sigma=tc.sigma,
            diurnal_amplitude=tc.diurnal_amplitude,
            diurnal_period_s=tc.diurnal_period_s,
            diurnal_phase_rad=tc.diurnal_phase_rad,
            min_val=tc.min_val,
            max_val=tc.max_val,
        )

        hc = config.humidity
        self.humidity_proc = EnvironmentalDrift(
            baseline=hc.baseline,
            theta=hc.theta,
            sigma=hc.sigma,
            diurnal_amplitude=hc.diurnal_amplitude,
            diurnal_period_s=hc.diurnal_period_s,
            diurnal_phase_rad=hc.diurnal_phase_rad,
            min_val=max(0.0, hc.min_val if hc.min_val is not None else 0.0),
            max_val=min(100.0, hc.max_val if hc.max_val is not None else 100.0),
        )

        vc = config.vibration
        self.vibration_proc = EnvironmentalDrift(
            baseline=vc.baseline,
            theta=vc.theta,
            sigma=vc.sigma,
            diurnal_amplitude=vc.diurnal_amplitude,
            diurnal_period_s=vc.diurnal_period_s,
            diurnal_phase_rad=vc.diurnal_phase_rad,
            min_val=max(0.0, vc.min_val if vc.min_val is not None else 0.0),
            max_val=vc.max_val,
        )

        bc = config.background_light
        self.bg_light_proc = EnvironmentalDrift(
            baseline=bc.baseline,
            theta=bc.theta,
            sigma=bc.sigma,
            diurnal_amplitude=bc.diurnal_amplitude,
            diurnal_period_s=bc.diurnal_period_s,
            diurnal_phase_rad=bc.diurnal_phase_rad,
            min_val=max(0.0, bc.min_val if bc.min_val is not None else 0.0),
            max_val=bc.max_val,
        )

        pol = config.polarization_drift
        self.pol_drift_proc = EnvironmentalDrift(
            baseline=pol.baseline,
            theta=pol.theta,
            sigma=pol.sigma,
            diurnal_amplitude=pol.diurnal_amplitude,
            diurnal_period_s=pol.diurnal_period_s,
            diurnal_phase_rad=pol.diurnal_phase_rad,
            min_val=pol.min_val,
            max_val=pol.max_val,
        )

        clk = config.clock_drift
        self.clock_drift_proc = EnvironmentalDrift(
            baseline=clk.baseline,  # in ppm
            theta=clk.theta,
            sigma=clk.sigma,
            diurnal_amplitude=clk.diurnal_amplitude,
            diurnal_period_s=clk.diurnal_period_s,
            diurnal_phase_rad=clk.diurnal_phase_rad,
            min_val=clk.min_val,
            max_val=clk.max_val,
        )

        loss_c = config.channel_loss_drift
        self.loss_drift_proc = EnvironmentalDrift(
            baseline=loss_c.baseline,  # delta loss in dB
            theta=loss_c.theta,
            sigma=loss_c.sigma,
            diurnal_amplitude=loss_c.diurnal_amplitude,
            diurnal_period_s=loss_c.diurnal_period_s,
            diurnal_phase_rad=loss_c.diurnal_phase_rad,
            min_val=loss_c.min_val,
            max_val=loss_c.max_val,
        )

        src_c = config.source_intensity_drift
        self.source_drift_proc = EnvironmentalDrift(
            baseline=src_c.baseline,  # relative delta
            theta=src_c.theta,
            sigma=src_c.sigma,
            diurnal_amplitude=src_c.diurnal_amplitude,
            diurnal_period_s=src_c.diurnal_period_s,
            diurnal_phase_rad=src_c.diurnal_phase_rad,
            min_val=src_c.min_val,
            max_val=src_c.max_val,
        )

        det_c = config.detector_efficiency_drift
        self.detector_drift_proc = EnvironmentalDrift(
            baseline=det_c.baseline,  # relative delta
            theta=det_c.theta,
            sigma=det_c.sigma,
            diurnal_amplitude=det_c.diurnal_amplitude,
            diurnal_period_s=det_c.diurnal_period_s,
            diurnal_phase_rad=det_c.diurnal_phase_rad,
            min_val=det_c.min_val,
            max_val=det_c.max_val,
        )

        # Monitoring sensor suite
        sc = config.sensors
        self.temp_sensor = PhysicalSensor(
            label="temperature_sensor",
            unit="C",
            noise_std=sc.temperature.noise_std,
            quantization_step=sc.temperature.quantization_step,
            missing_probability=sc.temperature.missing_probability,
            sampling_interval_s=sc.temperature.sampling_interval_s,
            min_reading=sc.temperature.min_reading,
            max_reading=sc.temperature.max_reading,
        )
        self.humidity_sensor = PhysicalSensor(
            label="humidity_sensor",
            unit="%",
            noise_std=sc.humidity.noise_std,
            quantization_step=sc.humidity.quantization_step,
            missing_probability=sc.humidity.missing_probability,
            sampling_interval_s=sc.humidity.sampling_interval_s,
            min_reading=0.0,
            max_reading=100.0,
        )
        self.vibration_sensor = PhysicalSensor(
            label="vibration_sensor",
            unit="g",
            noise_std=sc.vibration.noise_std,
            quantization_step=sc.vibration.quantization_step,
            missing_probability=sc.vibration.missing_probability,
            sampling_interval_s=sc.vibration.sampling_interval_s,
            min_reading=0.0,
            max_reading=sc.vibration.max_reading,
        )
        self.bg_sensor = PhysicalSensor(
            label="background_light_sensor",
            unit="lux",
            noise_std=sc.background_light.noise_std,
            quantization_step=sc.background_light.quantization_step,
            missing_probability=sc.background_light.missing_probability,
            sampling_interval_s=sc.background_light.sampling_interval_s,
            min_reading=0.0,
            max_reading=sc.background_light.max_reading,
        )

        # Current cached state
        self._current_state = self._build_state()

    def _build_state(self) -> EnvironmentState:
        """Construct the EnvironmentState snapshot from current physical and sensor values."""
        true_t = self.temperature_proc.current_value()
        true_h = self.humidity_proc.current_value()
        true_v = self.vibration_proc.current_value()
        true_bg = self.bg_light_proc.current_value()
        true_pol = self.pol_drift_proc.current_value()
        true_clk_ppm = self.clock_drift_proc.current_value()
        true_loss_db = self.loss_drift_proc.current_value()
        true_src = self.source_drift_proc.current_value()
        true_det = self.detector_drift_proc.current_value()

        # Background stray photon rate coupling: R_bg = R_bg,0 + k_bg * lux
        k_bg = self.config.coupling.ambient_light_to_bg_rate
        base_bg_rate = self.config.coupling.base_background_rate_hz
        bg_photon_rate = max(0.0, base_bg_rate + k_bg * true_bg)

        return EnvironmentState(
            label=self.config.type,
            temperature_c=true_t,
            humidity_pct=true_h,
            vibration_g=true_v,
            background_light_lux=true_bg,
            background_photon_rate_hz=bg_photon_rate,
            polarization_drift_deg=true_pol,
            clock_offset_s=self.clock_offset_s,
            clock_drift_ppm=true_clk_ppm,
            channel_loss_drift_db=true_loss_db,
            source_intensity_drift=true_src,
            detector_efficiency_drift=true_det,
            sensor_temperature_c=self.temp_sensor.last_reading,
            sensor_humidity_pct=self.humidity_sensor.last_reading,
            sensor_vibration_g=self.vibration_sensor.last_reading,
            sensor_background_light_lux=self.bg_sensor.last_reading,
        )

    def get_state(self, t: float) -> EnvironmentState:
        return self._current_state

    def update(self, t: float) -> None:
        """Base interface: advance to time t (without explicit external RNG)."""
        dt = t - self.t
        if dt > 0.0:
            # Deterministic internal dummy step if called without RNG
            dummy_rng = np.random.default_rng(0)
            self.step_dynamics(dt, dummy_rng, dummy_rng)

    def step_dynamics(self, dt: float, env_rng: Generator, sensor_rng: Generator) -> EnvironmentState:
        """Advance physical environment and sensors by elapsed duration ``dt`` [s]."""
        if dt <= 0.0:
            return self._current_state

        self.t += dt

        # Step continuous physical processes
        self.temperature_proc.step(dt, env_rng)
        self.humidity_proc.step(dt, env_rng)
        self.vibration_proc.step(dt, env_rng)
        self.bg_light_proc.step(dt, env_rng)
        self.pol_drift_proc.step(dt, env_rng)
        clk_ppm = self.clock_drift_proc.step(dt, env_rng)
        self.loss_drift_proc.step(dt, env_rng)
        self.source_drift_proc.step(dt, env_rng)
        self.detector_drift_proc.step(dt, env_rng)

        # Accumulate clock phase offset (ppm = 1e-6)
        self.clock_offset_s += (clk_ppm * 1e-6) * dt

        # Sample sensors with sensor_rng
        self.temp_sensor.measure(self.temperature_proc.current_value(), self.t, sensor_rng)
        self.humidity_sensor.measure(self.humidity_proc.current_value(), self.t, sensor_rng)
        self.vibration_sensor.measure(self.vibration_proc.current_value(), self.t, sensor_rng)
        self.bg_sensor.measure(self.bg_light_proc.current_value(), self.t, sensor_rng)

        self._current_state = self._build_state()
        return self._current_state

    def get_runtime_state(self) -> dict[str, Any]:
        """Serialize complete state of all processes and sensors for checkpointing."""
        return {
            "t": self.t,
            "clock_offset_s": self.clock_offset_s,
            "temperature": self.temperature_proc.get_state(),
            "humidity": self.humidity_proc.get_state(),
            "vibration": self.vibration_proc.get_state(),
            "background_light": self.bg_light_proc.get_state(),
            "polarization_drift": self.pol_drift_proc.get_state(),
            "clock_drift": self.clock_drift_proc.get_state(),
            "loss_drift": self.loss_drift_proc.get_state(),
            "source_drift": self.source_drift_proc.get_state(),
            "detector_drift": self.detector_drift_proc.get_state(),
            "sensor_temp": self.temp_sensor.get_state(),
            "sensor_humidity": self.humidity_sensor.get_state(),
            "sensor_vibration": self.vibration_sensor.get_state(),
            "sensor_bg": self.bg_sensor.get_state(),
        }

    def restore_runtime_state(self, state: dict[str, Any]) -> None:
        """Restore all processes and sensors from checkpoint."""
        self.t = float(state["t"])
        self.clock_offset_s = float(state["clock_offset_s"])
        self.temperature_proc.restore_state(state["temperature"])
        self.humidity_proc.restore_state(state["humidity"])
        self.vibration_proc.restore_state(state["vibration"])
        self.bg_light_proc.restore_state(state["background_light"])
        self.pol_drift_proc.restore_state(state["polarization_drift"])
        self.clock_drift_proc.restore_state(state["clock_drift"])
        self.loss_drift_proc.restore_state(state["loss_drift"])
        self.source_drift_proc.restore_state(state["source_drift"])
        self.detector_drift_proc.restore_state(state["detector_drift"])
        self.temp_sensor.restore_state(state["sensor_temp"])
        self.humidity_sensor.restore_state(state["sensor_humidity"])
        self.vibration_sensor.restore_state(state["sensor_vibration"])
        self.bg_sensor.restore_state(state["sensor_bg"])
        self._current_state = self._build_state()
