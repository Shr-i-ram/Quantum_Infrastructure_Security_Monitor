"""QKD Protocol, Architecture Pipeline, Basis Selection, and Sifting Plots."""

from __future__ import annotations

import pathlib
from typing import Any, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from qism.visualization.style import COLORS, FIGURE_SIZES, save_figure


def plot_simulation_pipeline(
    out_dir: pathlib.Path,
    filename_stem: str = "01_simulation_pipeline",
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
) -> dict[str, pathlib.Path]:
    """Figure 01: Presentation slide showing the end-to-end V0.1–V0.5 QKD simulation pipeline."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_slide"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Title and Subtitle
    ax.text(
        8.0, 8.4,
        "QISM Simulation Pipeline: Physics-Informed Stochastic QKD Telemetry",
        fontsize=16, fontweight="bold", ha="center", va="center", color=COLORS["text_dark"]
    )
    ax.text(
        8.0, 7.9,
        "Decoy-State BB84 Protocol with Fiber Channel, SPAD Dynamics, Physical Attack Injection, and Sharded Telemetry",
        fontsize=11, ha="center", va="center", color=COLORS["text_muted"]
    )

    # Box styling helper
    def draw_stage_box(x, y, w, h, title, subtitle, items, color, bg_color):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15,rounding_size=0.2",
            linewidth=1.8, edgecolor=color, facecolor=bg_color,
            zorder=2
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.35, title, fontsize=11, fontweight="bold", ha="center", color=color, zorder=3)
        ax.text(x + w / 2, y + h - 0.75, subtitle, fontsize=8.5, fontstyle="italic", ha="center", color=COLORS["text_muted"], zorder=3)
        
        y_item = y + h - 1.2
        for item in items:
            ax.text(x + 0.25, y_item, f"• {item}", fontsize=8.5, ha="left", color=COLORS["text_dark"], zorder=3)
            y_item -= 0.38

    # Stages definition
    # 1. Alice
    draw_stage_box(
        0.6, 2.2, 2.7, 4.8,
        "Alice (Transmitter)",
        "State & Intensity Prep",
        [
            "Weak Coherent Pulse",
            "Poisson statistics: N ~ Pois(μ)",
            "Decoy states: μ_sig, μ_dec, μ_vac",
            "Random bits: {0, 1}",
            "Random bases: {Z, X}",
            "Laser intensity jitter std",
            "Repetition rate: f_rep = 1 MHz",
        ],
        COLORS["alice"], "#EFF6FF"
    )

    # 2. Quantum Channel & Eve
    draw_stage_box(
        4.0, 2.2, 3.2, 4.8,
        "Quantum Channel & Eve",
        "Fiber Loss & Interventions",
        [
            "Beer-Lambert Attenuation:",
            "  T(L) = 10^(-(αL + L_sys)/10)",
            "Fiber distance L ∈ [10, 50] km",
            "Loss coefficient α ≈ 0.2 dB/km",
            "Polarization drift & rotation θ",
            "Background ambient light coupling",
            "Causal Attack Injection (V0.3+):",
            "  IR, Loss, PNS, Timing, Blinding",
        ],
        COLORS["eve"], "#FEF2F2"
    )

    # 3. Bob Receiver & SPAD
    draw_stage_box(
        7.9, 2.2, 3.3, 4.8,
        "Bob (Receiver & SPAD)",
        "Detection & Dynamics",
        [
            "Passive/Active Basis: {Z, X}",
            "Two SPADs (D0 and D1)",
            "Efficiency: η_det ∈ [10%, 25%]",
            "Dark counts: Pois(λ_dark · Δt)",
            "Timing jitter & 1 ns gate window",
            "Nonparalyzable Dead Time τ_dead",
            "Rate-dependent τ_eff(λ)",
            "Exponential Afterpulsing queue",
        ],
        COLORS["bob"], "#ECFDF5"
    )

    # 4. Telemetry Engine & Sifting
    draw_stage_box(
        11.9, 2.2, 3.5, 4.8,
        "Post-Processing & Telemetry",
        "Windowing & Dataset Shards",
        [
            "Classical Sifting (Z/X match)",
            "Raw & Sifted Key Rates",
            "QBER Calculation (E = N_err/N_sift)",
            "Decoy Yields & Gain Estimation",
            "Temporal Windows (e.g. 2k pulses)",
            "130-feature Telemetry Schema",
            "Output: events.parquet, tel.parquet",
            "Splits: Calib, Normal, Atk, Gen",
        ],
        COLORS["calibration"], "#F5F3FF"
    )

    # Connecting Arrows
    arrow_props = dict(arrowstyle="->", lw=2.2, color=COLORS["text_muted"], shrinkA=5, shrinkB=5)
    ax.annotate("", xy=(4.0, 4.6), xytext=(3.3, 4.6), arrowprops=arrow_props, zorder=4)
    ax.annotate("", xy=(7.9, 4.6), xytext=(7.2, 4.6), arrowprops=arrow_props, zorder=4)
    ax.annotate("", xy=(11.9, 4.6), xytext=(11.2, 4.6), arrowprops=arrow_props, zorder=4)

    # Bottom notes / metadata bar
    bar_rect = patches.FancyBboxPatch(
        (0.6, 0.6), 14.8, 1.1,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        linewidth=1.0, edgecolor=COLORS["border"], facecolor=COLORS["card_bg"],
        zorder=1
    )
    ax.add_patch(bar_rect)
    ax.text(
        1.0, 1.25,
        "Scientific Invariant: Causal Propagation",
        fontsize=9.5, fontweight="bold", color=COLORS["text_dark"], zorder=2
    )
    ax.text(
        1.0, 0.85,
        "Attack and environmental drift causally perturb physical simulation variables (photons, transmittances, efficiencies) -> events -> windowed telemetry. Telemetry features are never modified directly.",
        fontsize=8.5, color=COLORS["text_muted"], zorder=2
    )

    return save_figure(fig, out_dir, filename_stem, formats=formats, dpi=dpi)


def plot_basis_sifting_funnel(
    events_df: pd.DataFrame,
    out_dir: pathlib.Path,
    filename_stem: str = "05_basis_selection_and_sifting_funnel",
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
) -> dict[str, pathlib.Path]:
    """Figure 05: Basis selection matrix and sifting funnel reduction."""
    fig, (ax_mat, ax_funnel) = plt.subplots(1, 2, figsize=FIGURE_SIZES["double_panel"])

    # 1. Basis Selection Matching Matrix
    if "alice_basis" in events_df.columns and "bob_basis" in events_df.columns:
        contingency = pd.crosstab(
            events_df["alice_basis"].astype(str),
            events_df["bob_basis"].astype(str),
            normalize=True
        ) * 100

        im = ax_mat.imshow(contingency.values, cmap="Blues", vmin=0, vmax=50)
        ax_mat.set_xticks(range(len(contingency.columns)))
        ax_mat.set_yticks(range(len(contingency.index)))
        ax_mat.set_xticklabels([f"Bob {c.upper()}" for c in contingency.columns], fontweight="bold")
        ax_mat.set_yticklabels([f"Alice {r.upper()}" for r in contingency.index], fontweight="bold")

        for i in range(len(contingency.index)):
            for j in range(len(contingency.columns)):
                val = contingency.values[i, j]
                is_match = (contingency.index[i] == contingency.columns[j])
                match_label = "MATCH (Sifted)" if is_match else "DISCARD"
                color = "white" if val > 20 else "black"
                ax_mat.text(
                    j, i, f"{val:.1f}%\n({match_label})",
                    ha="center", va="center", color=color, fontsize=10.5, fontweight="bold"
                )

        ax_mat.set_title("A: BB84 Basis Matching Matrix", fontsize=12, fontweight="bold", pad=10)
        plt.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04, label="Fraction of Pulses [%]")
    else:
        ax_mat.text(0.5, 0.5, "Basis columns not found in events", ha="center", va="center")

    # 2. Sifting Funnel Bar Chart
    total_pulses = len(events_df)
    n_clicks = int(events_df["detector_0_click"].sum() + events_df["detector_1_click"].sum()) if "detector_0_click" in events_df.columns else 0
    n_basis_match = int((events_df["alice_basis"] == events_df["bob_basis"]).sum()) if "alice_basis" in events_df.columns else 0
    
    n_sifted = int(events_df["sifted"].sum()) if "sifted" in events_df.columns else 0
    n_errors = int(events_df["bit_error"].sum()) if "bit_error" in events_df.columns else 0

    stages = [
        "1. Total Pulses\n(N_total)",
        "2. Basis Matched\n(Z=Z or X=X)",
        "3. Detected Clicks\n(D0 or D1)",
        "4. Sifted Key Bits\n(Match & Click)",
        "5. Key Errors\n(N_error)",
    ]
    counts = [total_pulses, n_basis_match, n_clicks, n_sifted, n_errors]
    percentages = [100.0 * c / max(1, total_pulses) for c in counts]
    colors = [COLORS["alice"], COLORS["channel"], COLORS["detector_0"], COLORS["bob"], COLORS["eve"]]

    bars = ax_funnel.bar(stages, counts, color=colors, edgecolor=COLORS["border"], width=0.55, zorder=3)
    ax_funnel.set_yscale("log")
    ax_funnel.set_ylabel("Count (log scale)", fontsize=11, fontweight="bold")
    ax_funnel.set_title("B: Sifting Funnel Reduction (Actual Simulator Counts)", fontsize=12, fontweight="bold", pad=10)

    for bar, count, pct in zip(bars, counts, percentages):
        y_pos = bar.get_height()
        ax_funnel.text(
            bar.get_x() + bar.get_width() / 2, y_pos * 1.25,
            f"{count:,}\n({pct:.2f}%)",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=COLORS["text_dark"]
        )

    ax_funnel.set_ylim(bottom=max(1, min(counts) * 0.2), top=total_pulses * 10)
    fig.suptitle(
        r"BB84 Protocol Physics: Basis Sifting Funnel $N_{\mathrm{total}} \to N_{\mathrm{sifted}}$ & Error Rate",
        fontsize=14, fontweight="bold", y=1.02
    )

    return save_figure(fig, out_dir, filename_stem, formats=formats, dpi=dpi)
