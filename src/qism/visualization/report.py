"""Comprehensive technical presentation report and figure metadata compiler."""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any


def compile_figure_metadata(
    figures_info: list[dict[str, Any]],
    out_dir: pathlib.Path,
    dataset_manifest: dict[str, Any],
) -> pathlib.Path:
    """Save figure provenance and parameter tracking to JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "figure_metadata.json"

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "simulator_version": "0.5.0",
        "dataset_id": dataset_manifest.get("dataset_id", "v05_small_dataset"),
        "master_seed": dataset_manifest.get("master_seed", 20260502),
        "total_presentation_figures": len(figures_info),
        "figures": figures_info,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return meta_path


def generate_visualization_report(
    figures_info: list[dict[str, Any]],
    out_dir: pathlib.Path,
    dataset_manifest: dict[str, Any],
) -> pathlib.Path:
    """Compile comprehensive technical Markdown report for presentation slide preparation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "visualization_report.md"

    doc = []
    doc.append("# QISM Simulator & Dataset Generation Methodology — Technical Presentation Report")
    doc.append("")
    doc.append(f"**Baseline Simulator Version:** `0.5.0` (V0.1–V0.5 Architecture)")  
    doc.append(f"**Dataset Source:** `{dataset_manifest.get('dataset_id', 'v05_small_dataset')}`")
    doc.append(f"**Master RNG Seed:** `{dataset_manifest.get('master_seed', 20260502)}`")
    doc.append(f"**Generated:** `{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}`")
    doc.append("")
    doc.append("---")
    doc.append("")
    doc.append("## 1. Simulator Overview & Architectural Philosophy")
    doc.append("")
    doc.append("The Quantum Infrastructure Security Monitor (QISM) simulator models physical, stochastic, and temporal behavior of fiber-based decoy-state BB84 quantum key distribution systems. Unlike conventional analytical models or black-box statistical simulators, QISM enforces a **strict causal propagation invariant**:")
    doc.append("")
    doc.append("$$\\text{Physical Parameter Intervention} \\longrightarrow \\text{Quantum State & Channel Propagation} \\longrightarrow \\text{SPAD Detector Kinetics} \\longrightarrow \\text{Microscopic Events} \\longrightarrow \\text{Windowed Telemetry}$$")
    doc.append("")
    doc.append("No telemetry feature or QBER value is ever altered synthetically. Every anomaly, metric deviation, or environmental drift emerges bottom-up from microscopic photon interactions, fiber attenuation, SPAD dead-time collisions, and afterpulse carrier dynamics.")
    doc.append("")
    doc.append("---")
    doc.append("")
    doc.append("## 2. Microscopic Event & Photon Generation Mechanics")
    doc.append("")
    doc.append("Alice operates a weak coherent pulse (WCP) laser at repetition rate $f_{\\mathrm{rep}} = 1\\,\\text{MHz}$ with randomized basis selection ($Z$ or $X$), bit preparation ($0$ or $1$), and three discrete intensity classes:")
    doc.append("- **Signal State**: $\\mu \\approx 0.50$ (high single-photon probability)")
    doc.append("- **Decoy State**: $\\nu \\approx 0.10$ (attenuated probe for photon-number-splitting detection)")
    doc.append("- **Vacuum State**: $\\omega = 0.0$ (dark count calibration)")
    doc.append("")
    doc.append("Photon number per pulse follows the Poisson distribution:")
    doc.append("$$P(N = n) = \\frac{\\mu^n e^{-\\mu}}{n!}$$")
    doc.append("")
    doc.append("---")
    doc.append("")
    doc.append("## 3. Quantum Channel Loss Modeling & Dark Count Noise Floor")
    doc.append("")
    doc.append("Photons propagate through a single-mode telecommunication fiber channel governed by Beer-Lambert exponential attenuation:")
    doc.append("$$L_{\\mathrm{total}} = \\alpha L + L_{\\mathrm{sys}} \\quad [\\mathrm{dB}], \\qquad T(L) = 10^{-L_{\\mathrm{total}} / 10}$$")
    doc.append("")
    doc.append("where $\\alpha \\approx 0.18 - 0.22\\,\\text{dB/km}$ and $L_{\\mathrm{sys}} \\approx 1.0\\,\\text{dB}$ represents lumped splice and connector insertion losses. At long distances, signal detection decays exponentially until it collides with the constant thermal dark-count noise floor $2 R_{\\mathrm{dark}}$, causing an inevitable rise in baseline QBER.")
    doc.append("")
    doc.append("---")
    doc.append("")
    doc.append("## 4. Bob SPAD Detector Dynamics & State Machine")
    doc.append("")
    doc.append("Bob implements two Single-Photon Avalanche Diodes (D0 for bit 0, D1 for bit 1) operating in gated Geiger mode with temporal state dynamics:")
    doc.append("1. **Nonparalyzable Dead Time**: Following an avalanche, the detector enters a dead interval of duration $\\tau_{\\mathrm{dead}}$ during which subsequent incident photons are discarded without extending dead time:")
    doc.append("$$r_{\\mathrm{obs}} = \\frac{\\lambda}{1 + \\lambda \\tau_{\\mathrm{dead}}}$$")
    doc.append("2. **Rate-Dependent Dead Time**: Heavy photon bombardment induces thermal and resistive heating, increasing effective dead time:")
    doc.append("$$\\tau_{\\mathrm{eff}}(\\lambda) = \\tau_0 + \\Delta \\tau \\frac{\\lambda}{\\lambda + \\lambda_c}$$")
    doc.append("3. **Exponential Afterpulsing**: Trapped charge carriers release delayed secondary avalanches:")
    doc.append("$$P(\\Delta t) = \\frac{1}{\\tau_{\\mathrm{ap}}} \\exp\\left(-\\frac{\\Delta t}{\\tau_{\\mathrm{ap}}}\\right)$$")
    doc.append("")
    doc.append("---")
    doc.append("")
    doc.append("## 5. Decoy-State BB84 Sifting & QBER Calculation")
    doc.append("")
    doc.append("Classical post-processing sifts matching basis measurements ($Z_{\\mathrm{Alice}} = Z_{\\mathrm{Bob}}$ or $X_{\\mathrm{Alice}} = X_{\\mathrm{Bob}}$), discarding mismatched bases (~50% reduction). Quantum Bit Error Rate (QBER) is evaluated directly from sifted key bits:")
    doc.append("$$\\mathrm{QBER} = \\frac{N_{\\mathrm{error}}}{N_{\\mathrm{sifted}}}$$")
    doc.append("")
    doc.append("---")
    doc.append("")
    doc.append("## 6. Physical Attack Injection & Telemetry Signatures")
    doc.append("")
    doc.append("The simulator implements 6 distinct causal attack mechanisms with continuous transition profiles (Smoothstep Onset $\\to$ Active $\\to$ Recovery):")
    doc.append("1. **Intercept-Resend**: Eve measures in random basis and resends; induces theoretical $\\approx 25\\%$ QBER increase in attacked pulses.")
    doc.append("2. **Channel Loss**: Attenuation increase (+6 dB); suppresses signal and decoy gains while elevating relative dark-count share.")
    doc.append("3. **Source Manipulation**: Pulse intensity perturbation $\\Delta\\mu$; distorts signal-to-decoy gain ratios.")
    doc.append("4. **Detector Manipulation**: SPAD blinding / efficiency collapse; forces extreme count rate drops and detector imbalance.")
    doc.append("5. **Timing Shift**: Arrival offset outside 1 ns gate; induces asymmetric count loss and timing jitter kurtosis.")
    doc.append("6. **Photon Number Splitting (PNS)**: Selective multi-photon extraction; violates decoy state gain bounds.")
    doc.append("")
    doc.append("---")
    doc.append("")
    doc.append("## 7. Presentation Figures Index & Slide Provenance")
    doc.append("")
    doc.append("| Figure ID | File Name | What It Shows | Key Analytical Formula | Slide Purpose |")
    doc.append("| :--- | :--- | :--- | :--- | :--- |")

    for f in figures_info:
        doc.append(f"| **{f.get('id', '')}** | `{f.get('filename', '')}` | {f.get('what_it_shows', '')} | {f.get('formula', 'N/A')} | {f.get('why_it_matters', '')} |")

    doc.append("")
    doc.append("---")
    doc.append("")
    doc.append("## 8. Presentation Delivery Notes")
    doc.append("")
    doc.append("When presenting these figures to technical audiences:")
    doc.append("1. **Emphasize Causality**: Point out that anomalies are not artificially drawn; they propagate from physical parameters through photon events to telemetry.")
    doc.append("2. **Highlight Multi-Dimensionality**: Show that while single-variable QBER is a scalar, the 130-feature telemetry vector exposes subtle attacks that QBER alone cannot reveal.")
    doc.append("3. **Distinguish Baseline vs ML**: Remind listeners that this presentation covers the physical simulator and dataset generation methodology (V0.1–V0.5), providing the foundational ground truth before any ML anomaly detector is applied.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(doc))

    return report_path
