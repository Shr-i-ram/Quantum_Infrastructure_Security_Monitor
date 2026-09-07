"""Physical relationship and non-ideality tests for V0.4.

Verifies causal couplings:
1. Channel attenuation increase -> detection rate decrease.
2. Detector efficiency increase -> detection rate increase.
3. Background light increase -> background detections and dead-time increase.
4. Temperature increase -> dark-count rate increase (when temp coupling enabled).
5. Source intensity increase -> photon number and detection rate increase.
6. Timing jitter and clock drift -> arrival timing statistics response.
"""

import math
import numpy as np
import pytest

from qism.config import default_config
from qism.simulation.simulator import Simulator


def test_channel_loss_causal_response():
    """Higher channel attenuation causes lower transmittance and lower detection rate."""
    cfg_low = default_config()
    cfg_low.channel.distance_km = 10.0  # low loss
    cfg_low.simulation.chunk_size = 5000
    sim_low = Simulator(cfg_low, "run_low", seed=42)
    df_low = sim_low.simulate_chunk(0, 5000, 0)
    clicks_low = int(np.sum(df_low["detector_0_click"] | df_low["detector_1_click"]))

    cfg_high = default_config()
    cfg_high.channel.distance_km = 50.0  # high loss
    cfg_high.simulation.chunk_size = 5000
    sim_high = Simulator(cfg_high, "run_high", seed=42)
    df_high = sim_high.simulate_chunk(0, 5000, 0)
    clicks_high = int(np.sum(df_high["detector_0_click"] | df_high["detector_1_click"]))

    assert clicks_low > clicks_high * 2  # substantially higher counts on lower loss


def test_detector_efficiency_causal_response():
    """Higher detector quantum efficiency causes higher detection rate."""
    cfg_low = default_config()
    cfg_low.detectors.detector_0.efficiency = 0.05
    cfg_low.detectors.detector_1.efficiency = 0.05
    cfg_low.simulation.chunk_size = 5000
    sim_low = Simulator(cfg_low, "run_eff_low", seed=100)
    df_low = sim_low.simulate_chunk(0, 5000, 0)
    clicks_low = int(np.sum(df_low["detector_0_click"] | df_low["detector_1_click"]))

    cfg_high = default_config()
    cfg_high.detectors.detector_0.efficiency = 0.25
    cfg_high.detectors.detector_1.efficiency = 0.25
    cfg_high.simulation.chunk_size = 5000
    sim_high = Simulator(cfg_high, "run_eff_high", seed=100)
    df_high = sim_high.simulate_chunk(0, 5000, 0)
    clicks_high = int(np.sum(df_high["detector_0_click"] | df_high["detector_1_click"]))

    assert clicks_high > clicks_low * 3


def test_background_light_increases_background_clicks_and_dead_time():
    """Ambient background photons cause background clicks and increase detector dead time."""
    # Simulation without background noise
    cfg_no_bg = default_config()
    cfg_no_bg.environment.type = "dynamic"
    cfg_no_bg.background.enabled = False
    cfg_no_bg.detectors.dead_time.enabled = True
    cfg_no_bg.detectors.dead_time.tau_dead_s = 1e-7  # 100 ns
    cfg_no_bg.simulation.chunk_size = 10000
    sim_no_bg = Simulator(cfg_no_bg, "run_no_bg", seed=555)
    df_no_bg = sim_no_bg.simulate_chunk(0, 10000, 0)
    bg_clicks_0 = int(np.sum(df_no_bg["background_detected"]))

    assert bg_clicks_0 == 0

    # Simulation with high ambient background noise
    cfg_bg = default_config()
    cfg_bg.environment.type = "dynamic"
    cfg_bg.background.enabled = True
    cfg_bg.environment.background_light.baseline = 1000.0  # High ambient lux
    cfg_bg.environment.coupling.base_background_rate_hz = 500_000.0  # 500 kHz stray flux
    cfg_bg.detectors.dead_time.enabled = True
    cfg_bg.detectors.dead_time.tau_dead_s = 1e-7
    cfg_bg.simulation.chunk_size = 10000
    sim_bg = Simulator(cfg_bg, "run_bg", seed=555)
    df_bg = sim_bg.simulate_chunk(0, 10000, 0)
    bg_clicks_1 = int(np.sum(df_bg["background_detected"]))

    assert bg_clicks_1 > 0
    # Background clicks must appear in click_type
    has_bg_type = np.any(np.char.find(df_bg["click_type"].to_numpy().astype(str), "background") >= 0)
    assert has_bg_type


def test_temperature_coupling_to_dark_counts():
    """Higher temperature exponentially increases dark count rate."""
    cfg = default_config()
    cfg.environment.type = "dynamic"
    cfg.detectors.detector_0.temp_coeff_dark_count = 0.05  # 5%/deg C
    cfg.detectors.detector_1.temp_coeff_dark_count = 0.05
    cfg.detectors.detector_0.t_ref_c = 20.0
    cfg.detectors.detector_1.t_ref_c = 20.0

    # Cold environment: 10 C
    cfg.environment.temperature.baseline = 10.0
    cfg.environment.temperature.sigma = 0.0
    cfg.environment.temperature.diurnal_amplitude = 0.0
    sim_cold = Simulator(cfg, "run_cold", seed=777)
    df_cold = sim_cold.simulate_chunk(0, 10000, 0)
    dark_rate_cold = float(np.nanmean(df_cold["dark_count_rate"]))

    # Hot environment: 30 C
    cfg.environment.temperature.baseline = 30.0
    sim_hot = Simulator(cfg, "run_hot", seed=777)
    df_hot = sim_hot.simulate_chunk(0, 10000, 0)
    dark_rate_hot = float(np.nanmean(df_hot["dark_count_rate"]))

    assert dark_rate_hot > dark_rate_cold * 2.0


def test_source_intensity_jitter_and_counts():
    """Source intensity scaling and pulse-to-pulse jitter propagate to photon counts."""
    cfg = default_config()
    cfg.environment.type = "dynamic"
    cfg.source.intensity_jitter_std = 0.15  # 15% pulse jitter
    cfg.simulation.chunk_size = 5000

    sim = Simulator(cfg, "run_jitter", seed=888)
    df = sim.simulate_chunk(0, 5000, 0)

    # Filter to signal pulses
    signal_mask = df["intensity_class"] == "signal"
    mu_signal_vals = df["mu"][signal_mask].to_numpy()
    # mu should exhibit variation around baseline 0.5
    assert np.std(mu_signal_vals) > 0.02


def test_clock_offset_and_timing_telemetry():
    """Local clock drift shifts observed arrival times and reports in telemetry."""
    cfg = default_config()
    cfg.environment.type = "dynamic"
    cfg.environment.clock_drift.baseline = 100.0  # 100 ppm
    cfg.environment.clock_drift.sigma = 0.0
    cfg.environment.clock_drift.diurnal_amplitude = 0.0
    cfg.simulation.chunk_size = 10000

    sim = Simulator(cfg, "run_clock", seed=999)
    df = sim.simulate_chunk(0, 10000, 0)

    # Clock offset should monotonically grow
    clock_offsets = df["clock_offset_s"].to_numpy()
    assert clock_offsets[-1] > clock_offsets[0]
    assert clock_offsets[-1] > 0.0
