"""Presentation-quality styling and theme system for QISM visualizations.

Standardized for publication-grade figures, slide presentations (16:9),
clear typographic hierarchy, and explicit physical units.
"""

from __future__ import annotations

import pathlib
from typing import Sequence
import matplotlib
import matplotlib.pyplot as plt

# Presentation color palette
COLORS = {
    # Actors
    "alice": "#1D4ED8",        # Deep royal blue
    "bob": "#059669",          # Emerald green
    "eve": "#DC2626",          # Crimson red
    "channel": "#6B7280",      # Neutral slate gray
    
    # States & Regimes
    "normal": "#2563EB",       # Vibrant blue
    "attack": "#DC2626",       # Crimson red
    "onset": "#D97706",        # Amber
    "termination": "#F59E0B",  # Light amber
    "recovery": "#10B981",    # Mint green
    "calibration": "#7C3AED",   # Purple
    "generalization": "#DB2777", # Rose pink
    
    # Decoy intensities
    "signal": "#2563EB",       # Blue
    "decoy": "#059669",        # Green
    "vacuum": "#D97706",       # Amber
    
    # Detectors
    "detector_0": "#0284C7",   # Sky blue
    "detector_1": "#EA580C",   # Orange
    "dark_count": "#64748B",   # Slate
    "afterpulse": "#8B5CF6",   # Violet
    
    # UI Elements
    "grid": "#E2E8F0",
    "text_dark": "#0F172A",
    "text_muted": "#475569",
    "background": "#FFFFFF",
    "card_bg": "#F8FAFC",
    "border": "#CBD5E1",
}

# Standardized slide dimensions (inches) for 16:9 presentations
FIGURE_SIZES = {
    "single_slide": (12.0, 6.75),   # 16:9 full slide
    "double_panel": (13.5, 6.0),    # 2 horizontal subplots
    "triple_panel": (15.0, 5.0),    # 3 horizontal subplots
    "grid_2x2": (13.0, 8.5),        # 2x2 grid
    "grid_2x3": (15.0, 8.5),        # 2x3 grid
    "wide_timeline": (14.0, 6.5),   # Long temporal traces
    "compact": (8.0, 5.0),          # Compact single plot
}


def apply_presentation_style() -> None:
    """Configure matplotlib with presentation-grade defaults."""
    matplotlib.use("Agg")  # Non-interactive headless backend
    
    plt.rcParams.update({
        # Figure layout
        "figure.facecolor": COLORS["background"],
        "figure.edgecolor": COLORS["background"],
        "figure.autolayout": False,
        
        # Typography
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Arial", "Liberation Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.5,
        "axes.labelweight": "medium",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10.5,
        
        # Axes & Grids
        "axes.facecolor": COLORS["background"],
        "axes.edgecolor": COLORS["border"],
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.8,
        
        # Lines & Markers
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "patch.edgecolor": COLORS["border"],
        
        # Legend
        "legend.facecolor": COLORS["background"],
        "legend.edgecolor": COLORS["border"],
        "legend.framealpha": 0.95,
        "legend.frameon": True,
        
        # Output
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
        "savefig.facecolor": COLORS["background"],
    })


def save_figure(
    fig: plt.Figure,
    out_dir: pathlib.Path,
    filename_stem: str,
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
) -> dict[str, pathlib.Path]:
    """Save figure in specified formats with explicit metadata.

    Returns dict mapping format extension -> absolute file path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pathlib.Path] = {}
    
    for fmt in formats:
        fmt_clean = fmt.lower().lstrip(".")
        target_path = out_dir / f"{filename_stem}.{fmt_clean}"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_path, dpi=dpi, format=fmt_clean, bbox_inches="tight")
        results[fmt_clean] = target_path
        
    plt.close(fig)
    return results


def format_axis_scientific(ax: plt.Axes, axis: str = "y", scilimits: tuple[int, int] = (-3, 4)) -> None:
    """Apply clean scientific notation to an axis."""
    axis_obj = ax.yaxis if axis == "y" else ax.xaxis
    axis_obj.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis=axis, style="sci", scilimits=scilimits)
