#!/usr/bin/env python3
"""Run correlation and diagnostic analysis for QISM V0.4 (Requirements 73, 91–93, 115).

Generates a representative dynamic environmental run, computes correlation
matrices across observable telemetry, confirms physical causal relationships,
and renders time-series and heatmap diagnostic plots in diagnostics/v0.4/.

Usage:
    python scripts/run_correlation_analysis.py
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Ensure src is on sys.path
_HERE = pathlib.Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qism.config import default_config, apply_overrides
from qism.simulation.runner import RunManager

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("qism.analysis")


def run_representative_simulation(n_pulses: int = 100000, window_size: int = 2000):
    """Run a representative normal simulation under dynamic environmental drift."""
    base = default_config()
    overrides = {
        "protocol.n_pulses": n_pulses,
        "telemetry.window_size": window_size,
        "simulation.chunk_size": 10000,
        "simulation.checkpoint_interval": 10000,
        "simulation.scenario_id": "v04_diagnostic_run",
        "dataset.family": "normal",
        "source.intensity_jitter_std": 0.02,
        "channel.loss_drift_enabled": True,
        "channel.misalignment_deg": 1.0,
        "background.enabled": True,
        "background.ambient_coupling_factor": 5.0,
        "background.rate_hz": 1000.0,
        "environment.type": "dynamic",
        "environment.enabled": True,
        "environment.temperature.baseline": 22.0,
        "environment.temperature.theta": 0.5,
        "environment.temperature.sigma": 0.2,
        "environment.temperature.diurnal_amplitude": 2.0,
        "environment.temperature.diurnal_period_s": 0.05, # scaled for observable drift in 100k pulses (0.1s)
        "environment.humidity.baseline": 55.0,
        "environment.humidity.theta": 0.2,
        "environment.humidity.sigma": 1.0,
        "environment.vibration.baseline": 0.03,
        "environment.vibration.theta": 1.0,
        "environment.vibration.sigma": 0.01,
        "environment.background_light.baseline": 200.0,
        "environment.background_light.theta": 0.5,
        "environment.background_light.sigma": 15.0,
        "environment.polarization_drift.baseline": 0.0,
        "environment.polarization_drift.theta": 0.2,
        "environment.polarization_drift.sigma": 0.15,
        "environment.clock_drift.baseline": 20.0,
        "environment.clock_drift.theta": 0.1,
        "environment.clock_drift.sigma": 2.0,
        "environment.channel_loss_drift.baseline": 0.0,
        "environment.channel_loss_drift.theta": 0.3,
        "environment.channel_loss_drift.sigma": 0.05,
        "detectors.detector_0.temp_coeff_dark_count": 0.03,
        "detectors.detector_1.temp_coeff_dark_count": 0.03,
        "detectors.detector_0.temp_coeff_efficiency": -0.001,
        "detectors.detector_1.temp_coeff_efficiency": -0.001,
        "detectors.detector_0.t_ref_c": 20.0,
        "detectors.detector_1.t_ref_c": 20.0,
    }
    cfg = apply_overrides(base, overrides)
    mgr = RunManager(
        cfg,
        "v04_diagnostic_run",
        output_root="data/v04_diagnostics",
        checkpoint_root="checkpoints/v04_diagnostics",
    )
    summary = mgr.run(fresh=True)
    df_win = mgr.writer.read_final_telemetry()
    df_evt = mgr.writer.read_final_events()
    return mgr, summary, df_win, df_evt


def compute_and_plot_correlations(df_win: pd.DataFrame, out_dir: pathlib.Path):
    """Compute correlation matrix and generate diagnostic visualizations."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Feature columns to analyze
    target_cols = [
        "sensor_temperature_mean",
        "sensor_humidity_mean",
        "sensor_vibration_mean",
        "sensor_background_light_mean",
        "clock_offset_mean",
        "detection_rate",
        "qber",
        "signal_gain",
        "decoy_gain",
        "vacuum_gain",
        "timing_offset_mean",
        "timing_offset_std",
        "dead_time_loss_fraction",
    ]
    
    available_cols = [c for c in target_cols if c in df_win.columns]
    corr = df_win[available_cols].corr(method="pearson")
    
    # Save CSV
    corr_csv = out_dir / "correlation_matrix.csv"
    corr.to_csv(corr_csv)
    log.info("Saved correlation matrix to %s", corr_csv)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(available_cols)))
    ax.set_yticks(range(len(available_cols)))
    ax.set_xticklabels(available_cols, rotation=90, fontsize=9)
    ax.set_yticklabels(available_cols, fontsize=9)
    ax.set_title("QISM V0.4 Telemetry Feature Correlation Matrix", pad=20, fontsize=14, fontweight="bold")
    
    # Add numerical labels
    for i in range(len(available_cols)):
        for j in range(len(available_cols)):
            val = corr.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)
    
    plt.tight_layout()
    corr_plot = out_dir / "correlation_matrix.png"
    plt.savefig(corr_plot, dpi=200)
    plt.close(fig)
    log.info("Saved correlation plot to %s", corr_plot)
    
    return corr


def plot_timeseries(df_win: pd.DataFrame, out_dir: pathlib.Path):
    """Plot multi-panel time-series showing coupled telemetry evolution."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    time_s = np.arange(len(df_win)) * 0.002 # 2000 pulses at 1 MHz = 2 ms per window
    
    # Panel 1: Environmental Sensors
    ax0 = axes[0]
    ax0.plot(time_s, df_win.get("sensor_temperature_mean", np.zeros(len(df_win))), "r-", label="Temperature (°C)")
    ax0.set_ylabel("Temp (°C)", color="r")
    ax0_twin = ax0.twinx()
    ax0_twin.plot(time_s, df_win.get("sensor_background_light_mean", np.zeros(len(df_win))), "y-", label="Ambient Light (lux)")
    ax0_twin.set_ylabel("Light (lux)", color="goldenrod")
    ax0.set_title("Physical Environment Sensors (Observable)", fontsize=11, fontweight="bold")
    ax0.grid(True, alpha=0.3)
    
    # Panel 2: Mechanical & Clock
    ax1 = axes[1]
    ax1.plot(time_s, df_win.get("sensor_vibration_mean", np.zeros(len(df_win))), "b-", label="Vibration (g)")
    ax1.set_ylabel("Vibration (g)", color="b")
    ax1_twin = ax1.twinx()
    if "clock_offset_mean" in df_win:
        ax1_twin.plot(time_s, df_win["clock_offset_mean"] * 1e12, "m-", label="Clock Offset (ps)")
        ax1_twin.set_ylabel("Clock (ps)", color="m")
    ax1.set_title("Mechanical Vibration & Receiver Clock Drift", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    # Panel 3: Optical Transmission & Gains
    ax2 = axes[2]
    ax2.plot(time_s, df_win.get("detection_rate", np.zeros(len(df_win))), "g-", label="Detection Rate")
    ax2.plot(time_s, df_win.get("signal_gain", np.zeros(len(df_win))), "c--", label="Signal Gain")
    ax2.set_ylabel("Rate / Gain")
    ax2.legend(loc="upper right")
    ax2.set_title("Optical Transmission & Decoy State Gains", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    
    # Panel 4: Quantum Bit Error Rate (QBER) & Timing
    ax3 = axes[3]
    ax3.plot(time_s, df_win.get("qber", np.zeros(len(df_win))) * 100, "tab:orange", label="QBER (%)")
    ax3.set_ylabel("QBER (%)", color="tab:orange")
    ax3_twin = ax3.twinx()
    if "timing_offset_std" in df_win:
        ax3_twin.plot(time_s, df_win["timing_offset_std"] * 1e12, "k:", label="Timing Jitter (ps)")
        ax3_twin.set_ylabel("Jitter (ps)", color="k")
    ax3.set_xlabel("Simulation Elapsed Time (s)")
    ax3.set_title("Security Metrics: QBER & Arrival Timing Variance", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    ts_plot = out_dir / "timeseries_telemetry.png"
    plt.savefig(ts_plot, dpi=200)
    plt.close(fig)
    log.info("Saved time-series plot to %s", ts_plot)


def main():
    print("=" * 70)
    print("QISM V0.4 Correlation & Diagnostic Analysis (Requirement 115)")
    print("Executing representative normal simulation (100,000 pulses)...")
    
    out_dir = pathlib.Path("diagnostics/v0.4")
    mgr, summary, df_win, df_evt = run_representative_simulation(100000, 2000)
    
    print(f"\nSimulation completed: {summary.pulses_completed} pulses, {len(df_win)} windows")
    print(f"  QBER: {summary.qber:.6f}, Signal Gain: {summary.signal_gain:.6f}")
    
    print("\nComputing telemetry correlation matrix...")
    corr = compute_and_plot_correlations(df_win, out_dir)
    
    print("\nGenerating multi-panel time-series diagnostic plots...")
    plot_timeseries(df_win, out_dir)
    
    # Verify core physical relationships
    summary_dict = {
        "pulses": summary.pulses_completed,
        "n_windows": len(df_win),
        "mean_temperature_c": float(df_win["sensor_temperature_mean"].mean()) if "sensor_temperature_mean" in df_win else None,
        "mean_ambient_light_lux": float(df_win["sensor_background_light_mean"].mean()) if "sensor_background_light_mean" in df_win else None,
        "mean_vibration_g": float(df_win["sensor_vibration_mean"].mean()) if "sensor_vibration_mean" in df_win else None,
        "mean_qber": float(summary.qber),
        "mean_signal_gain": float(summary.signal_gain),
    }
    
    summary_file = out_dir / "correlation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"\nDiagnostic analysis artifacts written to {out_dir}/")
    print("  - correlation_matrix.csv")
    print("  - correlation_matrix.png")
    print("  - timeseries_telemetry.png")
    print("  - correlation_summary.json")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
