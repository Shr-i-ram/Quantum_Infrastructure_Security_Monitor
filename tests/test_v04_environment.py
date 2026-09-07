"""Tests for V0.4 environmental drift processes, sensors, and state (V0.4)."""

import math
import numpy as np
import pytest

from qism.config import default_config, EnvironmentConfig
from qism.environment.processes import OUProcess, DiurnalCycle, EnvironmentalDrift
from qism.environment.sensors import PhysicalSensor
from qism.environment.dynamic import DynamicEnvironment


def test_ou_process_mean_reversion_and_bounds():
    """OU process should revert toward mean and respect hard physical bounds."""
    rng = np.random.default_rng(42)
    # Target mean = 20.0, start far away at 40.0
    ou = OUProcess(baseline=20.0, theta=2.0, sigma=0.5, min_val=15.0, max_val=25.0)
    ou.x = 40.0  # Force out of bound to test clamping on step

    dt = 0.1
    for _ in range(100):
        ou.step(dt, rng)

    # After 10 seconds of simulation with theta=2.0, should be close to 20.0
    assert 18.0 <= ou.current_value() <= 22.0
    # Must strictly satisfy bounds
    assert 15.0 <= ou.current_value() <= 25.0


def test_ou_process_exact_variance():
    """Stationary variance of OU process: Var = sigma^2 / (2 * theta)."""
    theta = 1.5
    sigma = 0.8
    expected_var = (sigma ** 2) / (2.0 * theta)

    rng = np.random.default_rng(12345)
    ou = OUProcess(baseline=0.0, theta=theta, sigma=sigma)

    # Sample a long trajectory with small dt
    dt = 0.05
    n_steps = 20000
    samples = np.empty(n_steps)
    for i in range(n_steps):
        samples[i] = ou.step(dt, rng)

    sample_var = float(np.var(samples[5000:]))  # Discard burn-in
    assert math.isclose(sample_var, expected_var, rel_tol=0.15)


def test_diurnal_cycle():
    """Diurnal cycle should compute A * sin(2*pi*t/P + phi)."""
    dc = DiurnalCycle(amplitude=5.0, period_s=86400.0, phase_rad=0.0)
    assert math.isclose(dc.value_at(0.0), 0.0, abs_tol=1e-9)
    assert math.isclose(dc.value_at(21600.0), 5.0, abs_tol=1e-9)  # 1/4 period: peak
    assert math.isclose(dc.value_at(43200.0), 0.0, abs_tol=1e-9)  # 1/2 period: zero
    assert math.isclose(dc.value_at(64800.0), -5.0, abs_tol=1e-9) # 3/4 period: trough


def test_environmental_drift_serialization():
    """EnvironmentalDrift process must roundtrip state exactly."""
    rng = np.random.default_rng(999)
    drift = EnvironmentalDrift(
        baseline=25.0,
        theta=0.5,
        sigma=0.2,
        diurnal_amplitude=2.0,
        diurnal_period_s=3600.0,
        min_val=10.0,
        max_val=40.0,
    )

    for _ in range(50):
        drift.step(1.0, rng)

    val_before = drift.current_value()
    saved = drift.get_state()

    # Create new instance and restore
    restored = EnvironmentalDrift(baseline=25.0, theta=0.5, sigma=0.2)
    restored.restore_state(saved)

    assert math.isclose(restored.current_value(), val_before, abs_tol=1e-12)
    assert math.isclose(restored.t, drift.t, abs_tol=1e-12)

    # Subsequent steps with same RNG seed must produce identical outputs
    rng1 = np.random.default_rng(111)
    rng2 = np.random.default_rng(111)
    next1 = drift.step(1.0, rng1)
    next2 = restored.step(1.0, rng2)
    assert math.isclose(next1, next2, abs_tol=1e-12)


def test_physical_sensor_noise_quantization_missingness():
    """Sensor should apply noise, ADC quantization, and Bernoulli missingness."""
    rng = np.random.default_rng(42)
    sensor = PhysicalSensor(
        label="temp_sensor",
        unit="C",
        noise_std=0.2,
        quantization_step=0.5,  # Nearest 0.5 C
        missing_probability=0.2,
        sampling_interval_s=1.0,
    )

    n_samples = 1000
    readings = []
    missing_count = 0

    for i in range(n_samples):
        t = float(i)
        r = sensor.measure(25.0, t, rng)
        if math.isnan(r):
            missing_count += 1
        else:
            readings.append(r)
            # Must be quantized to multiples of 0.5
            rem = (r / 0.5) - round(r / 0.5)
            assert math.isclose(rem, 0.0, abs_tol=1e-9)

    # Missing probability ~ 0.20
    missing_frac = missing_count / n_samples
    assert 0.15 <= missing_frac <= 0.25


def test_dynamic_environment_state_step():
    """DynamicEnvironment coordinates all physical drifts and sensor readings."""
    cfg = default_config().environment
    cfg.type = "dynamic"
    cfg.temperature.baseline = 22.0
    cfg.temperature.theta = 1.0
    cfg.temperature.sigma = 0.5

    env = DynamicEnvironment(cfg)
    state0 = env.get_state(0.0)
    assert state0.temperature_c == 22.0

    env_rng = np.random.default_rng(100)
    sensor_rng = np.random.default_rng(200)

    # Step for 100 seconds
    for _ in range(100):
        state = env.step_dynamics(1.0, env_rng, sensor_rng)

    assert state.label == "dynamic"
    assert not math.isnan(state.temperature_c)
    assert not math.isnan(state.channel_loss_drift_db)
    assert state.clock_offset_s is not None

    # Checkpoint roundtrip
    saved_state = env.get_runtime_state()
    env2 = DynamicEnvironment(cfg)
    env2.restore_runtime_state(saved_state)

    state2 = env2.get_state(env2.t)
    assert math.isclose(state.temperature_c, state2.temperature_c, abs_tol=1e-12)
    assert math.isclose(state.clock_offset_s, state2.clock_offset_s, abs_tol=1e-12)
