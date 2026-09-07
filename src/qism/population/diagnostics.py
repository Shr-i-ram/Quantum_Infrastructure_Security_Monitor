"""Statistical validation, coverage reporting, and correlation diagnostics for QISM populations.

In accordance with Requirements 63-68, 145-154:
- Computes empirical parameter distribution statistics (min, max, mean, std, quantiles).
- Runs Kolmogorov-Smirnov (KS) goodness-of-fit tests against theoretical distributions.
- Computes parameter correlation matrices to identify physical couplings vs accidental artifacts.
- Generates coverage plots and reports.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import kstest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qism.config import Scenario
from qism.population.distributions import SampledParameter

log = logging.getLogger("qism.population.diagnostics")


def extract_scenario_dataframe(scenarios: list[Scenario]) -> pd.DataFrame:
    """Extract flat DataFrame of numeric and categorical parameters across scenarios."""
    rows: list[dict[str, Any]] = []
    for sc in scenarios:
        row: dict[str, Any] = {
            "scenario_id": sc.scenario_id,
            "infrastructure_id": sc.infrastructure_id,
            "regime": sc.regime,
            "is_edge_case": sc.is_edge_case,
            "dataset_split": sc.dataset_split,
            "config_hash": sc.config_hash,
        }
        # Flatten QKDConfig attributes
        cfg = sc.config
        row["channel.distance_km"] = cfg.channel.distance_km
        row["channel.attenuation_db_per_km"] = cfg.channel.attenuation_db_per_km
        row["channel.insertion_loss_db"] = cfg.channel.insertion_loss_db
        row["channel.misalignment_deg"] = cfg.channel.misalignment_deg
        row["source.mu_signal"] = cfg.source.mu_signal
        row["source.mu_decoy"] = cfg.source.mu_decoy
        row["source.intensity_jitter_std"] = cfg.source.intensity_jitter_std
        row["detectors.d0_efficiency"] = cfg.detectors.detector_0.efficiency
        row["detectors.d1_efficiency"] = cfg.detectors.detector_1.efficiency
        row["detectors.d0_dark_rate"] = cfg.detectors.detector_0.dark_count_rate_hz
        row["detectors.d1_dark_rate"] = cfg.detectors.detector_1.dark_count_rate_hz
        row["detectors.d0_jitter_ps"] = cfg.detectors.detector_0.timing_jitter_s * 1e12
        row["background.rate_hz"] = cfg.background.rate_hz
        row["environment.temp_baseline"] = cfg.environment.temperature.baseline
        row["environment.temp_sigma"] = cfg.environment.temperature.sigma
        row["environment.vibration_baseline"] = cfg.environment.vibration.baseline
        row["environment.clock_baseline"] = cfg.environment.clock_drift.baseline
        row["attack.enabled"] = cfg.attack.enabled
        row["attack.severity"] = cfg.attack.scenarios[0].severity if cfg.attack.scenarios else 0.0
        row["attack.type"] = cfg.attack.scenarios[0].type if cfg.attack.scenarios else "none"
        rows.append(row)
    return pd.DataFrame(rows)


def generate_coverage_report(df: pd.DataFrame, out_path: pathlib.Path | None = None) -> pd.DataFrame:
    """Compute summary statistics (min, max, mean, std, 25%, 50%, 75%) for continuous parameters."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[numeric_cols].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]).T
    stats = stats.rename(columns={"5%": "p05", "25%": "p25", "50%": "median", "75%": "p75", "95%": "p95"})

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(out_path)
        log.info("Saved coverage report to %s", out_path)
    return stats


def validate_population_distributions(
    samples_df: pd.DataFrame,
    parameter_specs: dict[str, SampledParameter],
) -> dict[str, dict[str, Any]]:
    """Run Kolmogorov-Smirnov (KS) tests comparing empirical samples against configured CDFs."""
    results: dict[str, dict[str, Any]] = {}
    for param_name, param_obj in parameter_specs.items():
        if param_name not in samples_df.columns:
            continue
        vals = samples_df[param_name].dropna().to_numpy()
        if len(vals) < 5:
            continue

        try:
            stat, pval = kstest(vals, param_obj.cdf)
            passed = pval > 0.01 or stat < 0.25 # practical deviation threshold
            results[param_name] = {
                "ks_stat": float(stat),
                "p_value": float(pval),
                "n_samples": len(vals),
                "passed": bool(passed),
            }
        except Exception as exc:
            results[param_name] = {"error": str(exc), "passed": False}
    return results


def plot_population_coverage(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    """Generate representative 2D scatter and correlation plots showing space coverage."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parameter correlation matrix
    numeric_cols = [
        c for c in [
            "channel.distance_km",
            "channel.insertion_loss_db",
            "source.mu_signal",
            "detectors.d0_efficiency",
            "detectors.d0_dark_rate",
            "background.rate_hz",
            "environment.temp_baseline",
            "environment.vibration_baseline",
            "attack.severity",
        ] if c in df.columns
    ]
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        fig.colorbar(cax)
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha="left", fontsize=8)
        ax.set_yticklabels(numeric_cols, fontsize=8)
        ax.set_title("Population Parameter Correlation Matrix", pad=20, fontsize=12, fontweight="bold")
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                val = corr.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.6 else "black", fontsize=7)
        plt.tight_layout()
        plt.savefig(out_dir / "parameter_correlation_matrix.png", dpi=150)
        plt.close(fig)

    # 2. Representative 2D space-filling scatter plots
    pairs = [
        ("channel.distance_km", "detectors.d0_efficiency", "Distance (km) vs Detector Efficiency"),
        ("source.mu_signal", "background.rate_hz", "Source Intensity vs Background Rate (Hz)"),
        ("environment.temp_baseline", "detectors.d0_dark_rate", "Temperature (°C) vs Dark Count Rate (Hz)"),
        ("environment.vibration_baseline", "channel.misalignment_deg", "Vibration (g) vs Misalignment (deg)"),
    ]
    valid_pairs = [(x, y, t) for x, y, t in pairs if x in df.columns and y in df.columns]
    if valid_pairs:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for i, (x_col, y_col, title) in enumerate(valid_pairs[:4]):
            ax = axes[i]
            # Color by split or regime if available
            colors = ["tab:blue" if not r else "tab:red" for r in df.get("is_edge_case", [False]*len(df))]
            ax.scatter(df[x_col], df[y_col], c=colors, alpha=0.7, edgecolors="none", s=25)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "parameter_space_coverage.png", dpi=150)
        plt.close(fig)
