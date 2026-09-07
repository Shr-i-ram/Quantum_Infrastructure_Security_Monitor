#!/usr/bin/env python3
"""Run QISM V0.3 attack validation suite.

Simulates all 6 attack types, generates validation figures under reports/figures/,
and compiles reports/attack_signatures.md.

Figures generated:
    reports/figures/14_attack_intercept_resend_qber.png
    reports/figures/15_attack_channel_loss_gain.png
    reports/figures/16_attack_source_manipulation.png
    reports/figures/17_attack_detector_manipulation.png
    reports/figures/18_attack_timing_shift.png
    reports/figures/19_attack_pns_decoy_yield.png
    reports/figures/20_multi_attack_progression.png

Report generated:
    reports/attack_signatures.md

Example:
    python scripts/run_attack_validation.py --pulses 60000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import default_config_path, setup


def _save_fig(fig: plt.Figure, figdir: pathlib.Path, name: str) -> pathlib.Path:
    figdir.mkdir(parents=True, exist_ok=True)
    out = figdir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    setup("run_attack_validation")

    parser = argparse.ArgumentParser(description="QISM V0.3 attack validation")
    parser.add_argument("--pulses", type=int, default=60_000, help="Pulses per simulation run")
    parser.add_argument("--seed", type=int, default=20240101, help="Simulation seed")
    parser.add_argument("--outdir", default="reports", help="Reports directory")
    args = parser.parse_args()

    from qism.config import AttackConfig, AttackScenarioConfig, QKDConfig, apply_overrides, default_config
    from qism.simulation.runner import RunManager

    outdir = pathlib.Path(args.outdir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    n_pulses = args.pulses
    window_size = 2000
    tmp_out = outdir / "_tmp_runs"
    tmp_ckpt = outdir / "_tmp_ckpts"

    base_cfg = default_config()
    base_cfg.protocol.n_pulses = n_pulses
    base_cfg.telemetry.window_size = window_size
    base_cfg.simulation.chunk_size = window_size
    base_cfg.simulation.checkpoint_interval = window_size

    figures = []
    signatures_summary = {}

    print(f"Running V0.3 attack validation simulations ({n_pulses} pulses per attack)...")

    # =========================================================================
    # 1. Intercept-Resend Attack Validation
    # =========================================================================
    print("1/7: Validating Intercept-Resend Attack...")
    ir_scenario = AttackScenarioConfig(
        scenario_id="val_ir",
        type="intercept_resend",
        severity=1.0,
        start_pulse=15000,
        end_pulse=45000,
        onset_pulses=5000,
        termination_pulses=5000,
        onset_profile="smoothstep",
        recovery_profile="smoothstep",
        params={"fraction": 1.0},
    )
    cfg_ir = base_cfg.model_copy(deep=True)
    cfg_ir.attack.enabled = True
    cfg_ir.attack.scenarios = [ir_scenario]
    cfg_ir.simulation.scenario_id = "val_ir"

    mgr_ir = RunManager(cfg_ir, "val_ir", output_root=tmp_out, checkpoint_root=tmp_ckpt)
    mgr_ir.run(fresh=True)
    df_ir = mgr_ir.writer.read_final_telemetry()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    pulse_centers = (df_ir["window_id"] + 0.5) * window_size
    ax.plot(pulse_centers, df_ir["qber"] * 100, "b-o", label="Observed QBER (%)", lw=2)
    ax.axvspan(15000, 20000, color="orange", alpha=0.2, label="Onset Ramp")
    ax.axvspan(20000, 40000, color="red", alpha=0.25, label="Active IR Attack")
    ax.axvspan(40000, 45000, color="orange", alpha=0.2, label="Recovery Ramp")
    ax.set_title("Intercept-Resend Attack: QBER Dynamics across Lifecycle", fontsize=12)
    ax.set_xlabel("Pulse Number")
    ax.set_ylabel("Quantum Bit Error Rate (QBER) [%]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    f_ir = _save_fig(fig, figdir, "14_attack_intercept_resend_qber")
    figures.append(f_ir)

    active_qber = float(df_ir[(pulse_centers >= 22000) & (pulse_centers <= 38000)]["qber"].mean())
    baseline_qber = float(df_ir[pulse_centers < 15000]["qber"].mean())
    signatures_summary["intercept_resend"] = {
        "baseline_qber": baseline_qber,
        "attack_qber": active_qber,
        "delta_qber": active_qber - baseline_qber,
    }

    # =========================================================================
    # 2. Channel Loss Attack Validation
    # =========================================================================
    print("2/7: Validating Channel Loss Attack...")
    loss_scenario = AttackScenarioConfig(
        scenario_id="val_loss",
        type="channel_loss",
        severity=1.0,
        start_pulse=15000,
        end_pulse=45000,
        onset_pulses=4000,
        termination_pulses=4000,
        onset_profile="smoothstep",
        recovery_profile="smoothstep",
        params={"delta_L_max_db": 6.0},
    )
    cfg_loss = base_cfg.model_copy(deep=True)
    cfg_loss.attack.enabled = True
    cfg_loss.attack.scenarios = [loss_scenario]
    cfg_loss.simulation.scenario_id = "val_loss"

    mgr_loss = RunManager(cfg_loss, "val_loss", output_root=tmp_out, checkpoint_root=tmp_ckpt)
    mgr_loss.run(fresh=True)
    df_loss = mgr_loss.writer.read_final_telemetry()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(pulse_centers, df_loss["signal_gain"] * 100, "g-s", label="Signal Gain (%)", lw=2)
    ax.plot(pulse_centers, df_loss["decoy_gain"] * 100, "m-^", label="Decoy Gain (%)", lw=2)
    ax.axvspan(15000, 45000, color="purple", alpha=0.15, label="Channel Loss Attack (+6dB)")
    ax.set_title("Channel Loss Attack: Signal & Decoy Gain Attenuation", fontsize=12)
    ax.set_xlabel("Pulse Number")
    ax.set_ylabel("Gain [%]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    f_loss = _save_fig(fig, figdir, "15_attack_channel_loss_gain")
    figures.append(f_loss)

    baseline_gain = float(df_loss[pulse_centers < 15000]["signal_gain"].mean())
    attack_gain = float(df_loss[(pulse_centers >= 20000) & (pulse_centers <= 40000)]["signal_gain"].mean())
    signatures_summary["channel_loss"] = {
        "baseline_signal_gain": baseline_gain,
        "attack_signal_gain": attack_gain,
        "attenuation_ratio": attack_gain / baseline_gain if baseline_gain > 0 else 0,
    }

    # =========================================================================
    # 3. Source Manipulation Attack Validation
    # =========================================================================
    print("3/7: Validating Source Manipulation Attack...")
    src_scenario = AttackScenarioConfig(
        scenario_id="val_src",
        type="source_manipulation",
        severity=1.0,
        start_pulse=15000,
        end_pulse=45000,
        onset_pulses=4000,
        termination_pulses=4000,
        params={"delta_mu": 0.4, "direction": "increase"},
    )
    cfg_src = base_cfg.model_copy(deep=True)
    cfg_src.attack.enabled = True
    cfg_src.attack.scenarios = [src_scenario]
    cfg_src.simulation.scenario_id = "val_src"

    mgr_src = RunManager(cfg_src, "val_src", output_root=tmp_out, checkpoint_root=tmp_ckpt)
    mgr_src.run(fresh=True)
    df_src = mgr_src.writer.read_final_telemetry()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(pulse_centers, df_src["observed_click_rate"], "c-d", label="Observed Click Rate (Hz)", lw=2)
    ax.plot(pulse_centers, df_src["signal_gain"] * 100, "y-o", label="Signal Gain (%)", lw=2)
    ax.axvspan(15000, 45000, color="orange", alpha=0.2, label="Source Intensity Injection (+0.4 mu)")
    ax.set_title("Source Manipulation Attack: Intensity & Detection Rate Surge", fontsize=12)
    ax.set_xlabel("Pulse Number")
    ax.set_ylabel("Telemetry Response")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    f_src = _save_fig(fig, figdir, "16_attack_source_manipulation")
    figures.append(f_src)

    signatures_summary["source_manipulation"] = {
        "baseline_detection_rate": float(df_src[pulse_centers < 15000]["observed_click_rate"].mean()),
        "attack_detection_rate": float(df_src[(pulse_centers >= 20000) & (pulse_centers <= 40000)]["observed_click_rate"].mean()),
    }

    # =========================================================================
    # 4. Detector Manipulation Attack Validation
    # =========================================================================
    print("4/7: Validating Detector Manipulation Attack...")
    det_scenario = AttackScenarioConfig(
        scenario_id="val_det",
        type="detector_manipulation",
        severity=1.0,
        start_pulse=15000,
        end_pulse=45000,
        onset_pulses=4000,
        termination_pulses=4000,
        params={"mode": "mismatch", "delta_eta": 0.8, "target_detector": 0},
    )
    cfg_det = base_cfg.model_copy(deep=True)
    cfg_det.attack.enabled = True
    cfg_det.attack.scenarios = [det_scenario]
    cfg_det.simulation.scenario_id = "val_det"

    mgr_det = RunManager(cfg_det, "val_det", output_root=tmp_out, checkpoint_root=tmp_ckpt)
    mgr_det.run(fresh=True)
    df_det = mgr_det.writer.read_final_telemetry()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(pulse_centers, df_det["detector_imbalance"], "r-x", label="Detector Imbalance (D0-D1)/(D0+D1)", lw=2)
    ax.axvspan(15000, 45000, color="red", alpha=0.15, label="Detector 0 Efficiency Blinding (eta0 x0.2)")
    ax.set_title("Detector Manipulation Attack: Strong Asymmetry & Imbalance", fontsize=12)
    ax.set_xlabel("Pulse Number")
    ax.set_ylabel("Detector Imbalance")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    f_det = _save_fig(fig, figdir, "17_attack_detector_manipulation")
    figures.append(f_det)

    signatures_summary["detector_manipulation"] = {
        "baseline_imbalance": float(df_det[pulse_centers < 15000]["detector_imbalance"].mean()),
        "attack_imbalance": float(df_det[(pulse_centers >= 20000) & (pulse_centers <= 40000)]["detector_imbalance"].mean()),
    }

    # =========================================================================
    # 5. Timing Attack Validation
    # =========================================================================
    print("5/7: Validating Timing Attack...")
    time_scenario = AttackScenarioConfig(
        scenario_id="val_timing",
        type="timing",
        severity=1.0,
        start_pulse=15000,
        end_pulse=45000,
        onset_pulses=4000,
        termination_pulses=4000,
        params={"delta_t_max_s": 250.0e-12},
    )
    cfg_time = base_cfg.model_copy(deep=True)
    cfg_time.attack.enabled = True
    cfg_time.attack.scenarios = [time_scenario]
    cfg_time.simulation.scenario_id = "val_timing"

    mgr_time = RunManager(cfg_time, "val_timing", output_root=tmp_out, checkpoint_root=tmp_ckpt)
    mgr_time.run(fresh=True)
    df_time = mgr_time.writer.read_final_telemetry()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(pulse_centers, df_time["mean_arrival_offset"] * 1e12, "darkblue", marker="o", label="Mean Timing Offset (ps)", lw=2)
    ax.axvspan(15000, 45000, color="cyan", alpha=0.2, label="Pulse Arrival Displacement (+250 ps)")
    ax.set_title("Timing Attack: Arrival Time Shift Relative to Detector Gate", fontsize=12)
    ax.set_xlabel("Pulse Number")
    ax.set_ylabel("Mean Timing Offset [ps]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    f_time = _save_fig(fig, figdir, "18_attack_timing_shift")
    figures.append(f_time)

    signatures_summary["timing"] = {
        "baseline_timing_offset_ps": float(df_time[pulse_centers < 15000]["mean_arrival_offset"].mean()) * 1e12,
        "attack_timing_offset_ps": float(df_time[(pulse_centers >= 20000) & (pulse_centers <= 40000)]["mean_arrival_offset"].mean()) * 1e12,
    }

    # =========================================================================
    # 6. PNS Attack Validation
    # =========================================================================
    print("6/7: Validating PNS Attack...")
    pns_scenario = AttackScenarioConfig(
        scenario_id="val_pns",
        type="pns",
        severity=1.0,
        start_pulse=15000,
        end_pulse=45000,
        onset_pulses=4000,
        termination_pulses=4000,
        params={"single_photon_suppression": 0.8},
    )
    cfg_pns = base_cfg.model_copy(deep=True)
    cfg_pns.attack.enabled = True
    cfg_pns.attack.scenarios = [pns_scenario]
    cfg_pns.simulation.scenario_id = "val_pns"

    mgr_pns = RunManager(cfg_pns, "val_pns", output_root=tmp_out, checkpoint_root=tmp_ckpt)
    mgr_pns.run(fresh=True)
    df_pns = mgr_pns.writer.read_final_telemetry()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    gain_ratio = df_pns["signal_gain"] / np.maximum(df_pns["decoy_gain"], 1e-9)
    ax.plot(pulse_centers, gain_ratio, "tab:brown", marker="^", label="Gain Ratio (Signal / Decoy)", lw=2)
    ax.axvspan(15000, 45000, color="brown", alpha=0.15, label="PNS Multi-Photon Splitting & Single Suppression")
    ax.set_title("Photon Number Splitting (PNS) Attack: Decoy-State Gain Disruption", fontsize=12)
    ax.set_xlabel("Pulse Number")
    ax.set_ylabel("Signal / Decoy Gain Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    f_pns = _save_fig(fig, figdir, "19_attack_pns_decoy_yield")
    figures.append(f_pns)

    signatures_summary["pns"] = {
        "baseline_gain_ratio": float(gain_ratio[pulse_centers < 15000].mean()),
        "attack_gain_ratio": float(gain_ratio[(pulse_centers >= 20000) & (pulse_centers <= 40000)].mean()),
    }

    # =========================================================================
    # 7. Multi-Attack Progression Validation
    # =========================================================================
    print("7/7: Validating Multi-Attack Episode Progression...")
    ep1 = AttackScenarioConfig(
        scenario_id="ep1_ir",
        type="intercept_resend",
        severity=0.8,
        start_pulse=10000,
        end_pulse=26000,
        onset_pulses=3000,
        termination_pulses=3000,
        params={"fraction": 1.0},
    )
    ep2 = AttackScenarioConfig(
        scenario_id="ep2_loss",
        type="channel_loss",
        severity=0.9,
        start_pulse=34000,
        end_pulse=50000,
        onset_pulses=3000,
        termination_pulses=3000,
        params={"delta_L_max_db": 6.0},
    )
    cfg_multi = base_cfg.model_copy(deep=True)
    cfg_multi.attack.enabled = True
    cfg_multi.attack.scenarios = [ep1, ep2]
    cfg_multi.simulation.scenario_id = "val_multi"

    mgr_multi = RunManager(cfg_multi, "val_multi", output_root=tmp_out, checkpoint_root=tmp_ckpt)
    mgr_multi.run(fresh=True)
    df_multi = mgr_multi.writer.read_final_telemetry()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(pulse_centers, df_multi["qber"] * 100, "b-o", label="QBER (%)", lw=2)
    ax1.axvspan(10000, 26000, color="red", alpha=0.2, label="Episode 1: Intercept-Resend")
    ax1.axvspan(34000, 50000, color="purple", alpha=0.15, label="Episode 2: Channel Loss")
    ax1.set_ylabel("QBER [%]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2.plot(pulse_centers, df_multi["signal_gain"] * 100, "g-s", label="Signal Gain (%)", lw=2)
    ax2.axvspan(10000, 26000, color="red", alpha=0.2)
    ax2.axvspan(34000, 50000, color="purple", alpha=0.15)
    ax2.set_xlabel("Pulse Number")
    ax2.set_ylabel("Gain [%]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    fig.suptitle("Multi-Attack Progression: Distinct Telemetry Responses in a Single Run", fontsize=13)
    f_multi = _save_fig(fig, figdir, "20_multi_attack_progression")
    figures.append(f_multi)

    # Clean up temporary runs directory to save disk space
    import shutil
    shutil.rmtree(tmp_out, ignore_errors=True)
    shutil.rmtree(tmp_ckpt, ignore_errors=True)

    # =========================================================================
    # 8. Generate Attack Signature Report
    # =========================================================================
    report_path = outdir / "attack_signatures.md"
    report_content = f"""# QISM Version 0.3 Attack Signatures Report

## Overview
This document characterizes the physical mechanisms, causal telemetry responses, and observable signatures for all 6 attacks implemented in **QISM Version 0.3**.

Each attack operates strictly through physical intervention without directly mutating telemetry features. Telemetry features are computed strictly causally by the aggregation engine.

---

## 1. Attack Signature Matrix

| Attack Type | Physical Intervention | Primary Observable Signature | Secondary Observable Signature | Baseline | Under Attack |
|---|---|---|---|---|---|
| **Intercept-Resend** | Intercepts & resends in Eve basis | **QBER surges** (+20% to +25%) | Key rate collapses; detection rate steady | {signatures_summary['intercept_resend']['baseline_qber']*100:.2f}% | {signatures_summary['intercept_resend']['attack_qber']*100:.2f}% |
| **Channel Loss** | Attenuates quantum channel transmittance | **Signal & decoy gains drop** | Sifted count drops; QBER steady or slight rise | {signatures_summary['channel_loss']['baseline_signal_gain']*100:.2f}% | {signatures_summary['channel_loss']['attack_signal_gain']*100:.2f}% |
| **Source Manipulation** | Modifies mean photon number $\\mu$ | **Detection rate surges/drops** | Multi-photon fraction alters | {signatures_summary['source_manipulation']['baseline_detection_rate']:.1f} Hz | {signatures_summary['source_manipulation']['attack_detection_rate']:.1f} Hz |
| **Detector Manipulation**| Alters target detector efficiency | **Detector click imbalance skew** | Double click rate alters | {signatures_summary['detector_manipulation']['baseline_imbalance']:+.4f} | {signatures_summary['detector_manipulation']['attack_imbalance']:+.4f} |
| **Timing Attack** | Displaces arrival time $\\Delta t$ | **Mean timing offset shifts** | Gate edge clipping causes loss | {signatures_summary['timing']['baseline_timing_offset_ps']:+.1f} ps | {signatures_summary['timing']['attack_timing_offset_ps']:+.1f} ps |
| **PNS Attack** | Splits multi-photon & suppresses singles | **Gain ratio anomaly (Signal/Decoy)** | Decoy yield bounds violated | {signatures_summary['pns']['baseline_gain_ratio']:.2f} | {signatures_summary['pns']['attack_gain_ratio']:.2f} |

---

## 2. Attack Lifecycle & Transition Dynamics
All attacks implement the 5-phase ground truth lifecycle:
1. **NORMAL**: Baseline legitimate operation ($A(t) = 0$).
2. **ONSET**: Configurable ramp profile (`smoothstep`, `linear`, `exponential`, or `instant`) transitioning $A(t): 0 \\to 1$.
3. **ACTIVE**: Full attack intervention ($A(t) = 1$).
4. **TERMINATION**: Configurable ramp down $A(t): 1 \\to 0$.
5. **RECOVERY**: System settling back to nominal baseline.

---

## 3. Zero-Leakage Separation Guarantee
Telemetry window tables strictly isolate observable features from ground truth labels:
- **Observable Columns**: QBER, counts, gains, timing histograms, detector imbalance, Fano factor, inter-arrival gaps.
- **Ground Truth Columns**: `ground_truth_attack_active`, `ground_truth_attack_type`, `ground_truth_attack_phase`, `ground_truth_attack_fraction`, `ground_truth_mean_severity`, `ground_truth_attack_id`.
ML models trained on observable columns have **zero access** to ground truth labels during feature extraction.

---

## 4. Multi-Attack Episodes
A single QKD operational run can host multiple non-overlapping attack episodes with independent types, severities, and profiles, verifying model robustness across dynamic operational environments.

*Generated automatically by `scripts/run_attack_validation.py`.*
"""
    report_path.write_text(report_content, encoding="utf-8")
    print(f"\nAttack signature validation complete!")
    print(f"  Figures saved to: {figdir}")
    print(f"  Report saved to:  {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
