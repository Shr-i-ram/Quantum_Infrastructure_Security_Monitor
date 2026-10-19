"""Event-level pulse raster timeline, photon statistics, and timing jitter plots."""

from __future__ import annotations

import pathlib
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qism.visualization.style import COLORS, FIGURE_SIZES, save_figure


def plot_event_timeline(
    events_df: pd.DataFrame,
    out_dir: pathlib.Path,
    filename_stem: str = "02_event_timeline",
    n_pulses: int = 50,
    start_idx: int = 0,
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
) -> dict[str, pathlib.Path]:
    """Figure 02: Pulse-by-pulse event raster timeline showing state prep, channel, detection, and sifting."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["wide_timeline"])

    slice_df = events_df.iloc[start_idx : start_idx + n_pulses].copy()
    pulse_indices = np.arange(len(slice_df))

    # Y-axis rows
    rows = [
        ("Alice Bit", 6),
        ("Alice Basis", 5),
        ("Intensity Class", 4),
        ("Photon Number", 3),
        ("Bob Basis", 2),
        ("Detector Click", 1),
        ("Sifting Outcome", 0),
    ]

    ax.set_ylim(-0.8, 7.0)
    ax.set_xlim(-1, n_pulses)
    ax.set_yticks([r[1] for r in rows])
    ax.set_yticklabels([r[0] for r in rows], fontweight="bold", fontsize=11)
    ax.set_xlabel(f"Simulated Pulse Index (Pulses {start_idx} to {start_idx + n_pulses - 1})", fontsize=11, fontweight="bold")
    ax.set_title(
        "Event-Level Microscopic Simulation: Pulse-by-Pulse State Preparation, Detection & Sifting",
        fontsize=13, fontweight="bold", pad=12
    )

    # Alternate background banding for pulses
    for p in pulse_indices:
        if p % 2 == 1:
            ax.axvspan(p - 0.45, p + 0.45, color="#F8FAFC", zorder=0)

    # 1. Alice Bit: 0 (circle) vs 1 (square)
    if "alice_bit" in slice_df.columns:
        bits = slice_df["alice_bit"].to_numpy()
        for p, b in zip(pulse_indices, bits):
            marker = "o" if b == 0 else "s"
            color = COLORS["alice"] if b == 0 else "#1E3A8A"
            ax.scatter(p, 6, marker=marker, s=80, color=color, zorder=3)
            ax.text(p, 6, str(int(b)), color="white", fontsize=7.5, ha="center", va="center", fontweight="bold", zorder=4)

    # 2. Alice Basis: Z vs X
    if "alice_basis" in slice_df.columns:
        bases = slice_df["alice_basis"].astype(str).str.upper().to_numpy()
        for p, b in zip(pulse_indices, bases):
            color = COLORS["alice"] if b == "Z" else "#0891B2"
            ax.text(p, 5, b, color=color, fontsize=10, ha="center", va="center", fontweight="bold", zorder=3)

    # 3. Intensity Class: Signal, Decoy, Vacuum
    if "intensity_class" in slice_df.columns:
        int_classes = slice_df["intensity_class"].astype(str).str.lower().to_numpy()
        for p, ic in zip(pulse_indices, int_classes):
            short = "S" if "sig" in ic else ("D" if "dec" in ic else "V")
            color = COLORS["signal"] if short == "S" else (COLORS["decoy"] if short == "D" else COLORS["vacuum"])
            ax.text(p, 4, short, color=color, fontsize=9.5, ha="center", va="center", fontweight="bold", zorder=3)

    # 4. Photon Number
    if "photon_number" in slice_df.columns:
        photons = slice_df["photon_number"].to_numpy()
        for p, n in zip(pulse_indices, photons):
            color = COLORS["channel"] if n == 0 else (COLORS["normal"] if n == 1 else COLORS["eve"])
            ax.scatter(p, 3, marker="D", s=70, color=color, zorder=3)
            ax.text(p, 3, str(int(n)), color="white" if n > 0 else "black", fontsize=7.5, ha="center", va="center", fontweight="bold", zorder=4)

    # 5. Bob Basis: Z vs X
    if "bob_basis" in slice_df.columns:
        bob_bases = slice_df["bob_basis"].astype(str).str.upper().to_numpy()
        for p, b in zip(pulse_indices, bob_bases):
            color = COLORS["bob"] if b == "Z" else "#0D9488"
            ax.text(p, 2, b, color=color, fontsize=10, ha="center", va="center", fontweight="bold", zorder=3)

    # 6. Detector Clicks: D0, D1, Double, Lost
    d0_clicks = slice_df.get("detector_0_click", pd.Series(0, index=slice_df.index)).to_numpy()
    d1_clicks = slice_df.get("detector_1_click", pd.Series(0, index=slice_df.index)).to_numpy()
    click_types = slice_df.get("click_type", pd.Series("none", index=slice_df.index)).astype(str).to_numpy()

    for p, d0, d1, ct in zip(pulse_indices, d0_clicks, d1_clicks, click_types):
        if d0 and d1:
            ax.scatter(p, 1, marker="X", s=90, color=COLORS["eve"], zorder=3)
            ax.text(p, 0.75, "2x", color=COLORS["eve"], fontsize=7.5, ha="center")
        elif d0:
            color = COLORS["dark_count"] if "dark" in ct else COLORS["detector_0"]
            ax.scatter(p, 1, marker="^", s=80, color=color, zorder=3)
            ax.text(p, 1, "D0", color="white", fontsize=6.5, ha="center", va="center", fontweight="bold")
        elif d1:
            color = COLORS["dark_count"] if "dark" in ct else COLORS["detector_1"]
            ax.scatter(p, 1, marker="v", s=80, color=color, zorder=3)
            ax.text(p, 1, "D1", color="white", fontsize=6.5, ha="center", va="center", fontweight="bold")
        else:
            ax.scatter(p, 1, marker="_", s=40, color="#CBD5E1", zorder=2)

    # 7. Sifting Outcome
    sifted = slice_df.get("sifted", pd.Series(False, index=slice_df.index)).astype(bool).to_numpy()
    errors = slice_df.get("bit_error", pd.Series(False, index=slice_df.index)).astype(bool).to_numpy()

    for p, s, err in zip(pulse_indices, sifted, errors):
        if s and err:
            ax.scatter(p, 0, marker="P", s=110, color=COLORS["eve"], zorder=3)
            ax.text(p, -0.35, "ERR", color=COLORS["eve"], fontsize=7.5, fontweight="bold", ha="center")
        elif s:
            ax.scatter(p, 0, marker="o", s=100, color=COLORS["bob"], zorder=3)
            ax.text(p, -0.35, "KEY", color=COLORS["bob"], fontsize=7.5, fontweight="bold", ha="center")
        else:
            ax.scatter(p, 0, marker="x", s=40, color="#CBD5E1", zorder=2)

    # Gridlines and legend elements
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.grid(axis="y", linestyle="-", alpha=0.3)

    return save_figure(fig, out_dir, filename_stem, formats=formats, dpi=dpi)


def plot_timing_jitter_and_arrival(
    events_df: pd.DataFrame,
    out_dir: pathlib.Path,
    filename_stem: str = "timing_jitter_gate_acceptance",
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
) -> dict[str, pathlib.Path]:
    """Exploratory Plot: SPAD timing jitter, gate width window, and inter-arrival times."""
    fig, (ax_jitter, ax_inter) = plt.subplots(1, 2, figsize=FIGURE_SIZES["double_panel"])

    # 1. Timing Jitter & 1 ns Gate Window
    if "timing_offset_s" in events_df.columns:
        offsets_ps = events_df["timing_offset_s"].dropna().to_numpy() * 1e12  # picoseconds
        if len(offsets_ps) > 0:
            ax_jitter.hist(offsets_ps, bins=50, density=True, color=COLORS["detector_0"], alpha=0.7, edgecolor=COLORS["border"])
            ax_jitter.axvline(-500, color=COLORS["eve"], linestyle="--", lw=1.8, label="Gate Window (-500 ps)")
            ax_jitter.axvline(500, color=COLORS["eve"], linestyle="--", lw=1.8, label="Gate Window (+500 ps)")
            ax_jitter.set_xlabel("Timing Offset from Pulse Center [ps]", fontsize=11, fontweight="bold")
            ax_jitter.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
            ax_jitter.set_title("A: Detection Arrival Timing Jitter (Gate Width = 1 ns)", fontsize=12, fontweight="bold")
            ax_jitter.legend(loc="upper right")
    else:
        ax_jitter.text(0.5, 0.5, "timing_offset_s not available in events", ha="center", va="center")

    # 2. Inter-arrival time distribution
    if "detector_0_time_since_last_click_s" in events_df.columns:
        inter_s = events_df["detector_0_time_since_last_click_s"].dropna().to_numpy()
        inter_us = inter_s[inter_s > 0] * 1e6  # microseconds
        if len(inter_us) > 0:
            ax_inter.hist(inter_us, bins=50, density=True, color=COLORS["detector_1"], alpha=0.7, edgecolor=COLORS["border"])
            ax_inter.set_xlabel("Time Since Last Click [µs]", fontsize=11, fontweight="bold")
            ax_inter.set_ylabel("Density (log scale)", fontsize=11, fontweight="bold")
            ax_inter.set_yscale("log")
            ax_inter.set_title("B: SPAD Inter-Arrival Time Distribution", fontsize=12, fontweight="bold")
    else:
        ax_inter.text(0.5, 0.5, "inter-arrival field not available in events", ha="center", va="center")

    fig.suptitle("SPAD Temporal Dynamics: Sub-Nanosecond Jitter and Inter-Arrival Statistics", fontsize=14, fontweight="bold", y=1.02)
    return save_figure(fig, out_dir, filename_stem, formats=formats, dpi=dpi)
