"""Parameter dependency resolution and regime classification for QISM population generation.

In accordance with Requirements 10, 18, 19, 60, 61, 62:
- Enforces hierarchical physical dependencies across parameters.
- Prevents unphysical combinations (e.g. independently uncorrelated detector pairs).
- Supports deliberate corner-case injection and nominal regime preservation.
- Assigns internal operational regime labels (nominal, low_signal, high_loss, high_noise, high_instability).
"""

from __future__ import annotations

import copy
from typing import Any
import numpy as np


def resolve_dependencies(
    params: dict[str, Any],
    rng: np.random.Generator,
    is_edge_case: bool = False,
    edge_case_type: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Resolve parameter dependencies, enforce physical relationships, and assign regime.

    Returns:
        (resolved_params, regime_label)
    """
    resolved = copy.deepcopy(params)

    # 1. Channel physical couplings
    # If distance is sampled, total channel loss relates directly to attenuation * distance + insertion loss
    dist_km = resolved.get("channel.distance_km", 25.0)
    att_db = resolved.get("channel.attenuation_db_per_km", 0.20)
    ins_loss = resolved.get("channel.insertion_loss_db", 1.0)
    # Ensure insertion loss is non-negative
    resolved["channel.insertion_loss_db"] = max(0.0, float(ins_loss))
    resolved["channel.attenuation_db_per_km"] = max(0.05, float(att_db))
    resolved["channel.distance_km"] = max(1.0, float(dist_km))

    # 2. Detector pair fabrication correlations
    # In real hardware, D0 and D1 are co-packaged or fabricated in similar batches.
    # If efficiency is sampled as "detectors.efficiency", link D0 and D1 with realistic mismatch.
    if "detectors.efficiency" in resolved:
        base_eta = float(resolved.pop("detectors.efficiency"))
        # fabrication mismatch std ~ 1% relative
        mismatch = float(rng.normal(0.0, 0.01 * base_eta))
        resolved["detectors.detector_0.efficiency"] = np.clip(base_eta + 0.5 * mismatch, 0.01, 0.95)
        resolved["detectors.detector_1.efficiency"] = np.clip(base_eta - 0.5 * mismatch, 0.01, 0.95)
    elif "detectors.detector_0.efficiency" in resolved and "detectors.detector_1.efficiency" not in resolved:
        resolved["detectors.detector_1.efficiency"] = resolved["detectors.detector_0.efficiency"]

    # Dark count rate linkage
    if "detectors.dark_count_rate_hz" in resolved:
        base_dark = float(resolved.pop("detectors.dark_count_rate_hz"))
        dark_mismatch = float(rng.normal(0.0, 0.05 * base_dark))
        resolved["detectors.detector_0.dark_count_rate_hz"] = max(10.0, base_dark + dark_mismatch)
        resolved["detectors.detector_1.dark_count_rate_hz"] = max(10.0, base_dark - dark_mismatch)
    elif "detectors.detector_0.dark_count_rate_hz" in resolved and "detectors.detector_1.dark_count_rate_hz" not in resolved:
        resolved["detectors.detector_1.dark_count_rate_hz"] = resolved["detectors.detector_0.dark_count_rate_hz"]

    # Timing jitter linkage
    if "detectors.timing_jitter_s" in resolved:
        base_jitter = float(resolved.pop("detectors.timing_jitter_s"))
        resolved["detectors.detector_0.timing_jitter_s"] = max(10.0e-12, base_jitter)
        resolved["detectors.detector_1.timing_jitter_s"] = max(10.0e-12, base_jitter)

    # 3. Source Decoy Hierarchy
    # Decoy BB84 requires mu_signal > mu_decoy > mu_vacuum == 0
    mu_sig = resolved.get("source.mu_signal", 0.50)
    mu_dec = resolved.get("source.mu_decoy", 0.10)
    if mu_dec >= mu_sig:
        resolved["source.mu_decoy"] = float(mu_sig * 0.25)
    resolved["source.mu_vacuum"] = 0.0

    # 4. Environmental Coupled Volatilities
    # If ambient temperature variance is elevated, drift volatility increases
    temp_sigma = resolved.get("environment.temperature.sigma", 0.1)
    if "environment.detector_efficiency_drift.sigma" not in resolved:
        resolved["environment.detector_efficiency_drift.sigma"] = float(np.clip(0.05 * temp_sigma, 0.001, 0.02))

    # 5. Corner Case Stress Regimes
    regime = "nominal"
    if is_edge_case:
        choices = ["low_signal", "high_loss", "high_noise", "high_instability"]
        edge_type = edge_case_type if edge_case_type in choices else str(rng.choice(choices))
        regime = edge_type

        if edge_type == "low_signal":
            resolved["source.mu_signal"] = float(rng.uniform(0.15, 0.28))
            resolved["source.mu_decoy"] = float(resolved["source.mu_signal"] * 0.2)
            if "detectors.detector_0.efficiency" in resolved:
                resolved["detectors.detector_0.efficiency"] *= 0.6
                resolved["detectors.detector_1.efficiency"] *= 0.6
        elif edge_type == "high_loss":
            resolved["channel.distance_km"] = float(rng.uniform(45.0, 65.0))
            resolved["channel.insertion_loss_db"] = float(rng.uniform(2.5, 4.5))
        elif edge_type == "high_noise":
            resolved["background.rate_hz"] = float(rng.uniform(5000.0, 20000.0))
            resolved["environment.background_light.baseline"] = float(rng.uniform(400.0, 900.0))
            if "detectors.detector_0.dark_count_rate_hz" in resolved:
                resolved["detectors.detector_0.dark_count_rate_hz"] *= 3.0
                resolved["detectors.detector_1.dark_count_rate_hz"] *= 3.0
        elif edge_type == "high_instability":
            resolved["environment.vibration.baseline"] = float(rng.uniform(0.05, 0.12))
            resolved["environment.clock_drift.baseline"] = float(rng.uniform(30.0, 70.0))
            resolved["environment.polarization_drift.sigma"] = float(rng.uniform(0.15, 0.35))
    else:
        # Determine regime from natural parameter values
        if resolved.get("channel.distance_km", 25.0) > 40.0 or resolved.get("channel.insertion_loss_db", 1.0) > 3.0:
            regime = "high_loss"
        elif resolved.get("source.mu_signal", 0.50) < 0.35:
            regime = "low_signal"
        elif resolved.get("background.rate_hz", 1000.0) > 4000.0:
            regime = "high_noise"
        elif resolved.get("environment.vibration.baseline", 0.02) > 0.04:
            regime = "high_instability"

    return resolved, regime
