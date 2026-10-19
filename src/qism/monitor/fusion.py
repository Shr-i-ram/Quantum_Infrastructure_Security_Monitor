"""Multivariate signal fusion engine for QISM V0.9.

Fuses disparate detector signals (temporal anomaly scores, statistical drift,
QBER deviations, optical gain deviations, detector timing deviations) into a unified,
interpretable security evidence score.

Supports:
- Method A: Rule-based synergy boosting
- Method B: Weighted evidence aggregation: F_t = sum(w_i * s_i)
- Method C: Persistence-aware evidence scaling
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from qism.monitor.types import DetectorSignal, FusionMethod


@dataclass
class SignalWeightConfig:
    """Configurable weights for heterogeneous security signals."""
    w_anomaly_model: float = 0.35
    w_qber_deviation: float = 0.25
    w_gain_deviation: float = 0.15
    w_timing_deviation: float = 0.15
    w_drift_score: float = 0.10

    def __post_init__(self):
        total = (
            self.w_anomaly_model
            + self.w_qber_deviation
            + self.w_gain_deviation
            + self.w_timing_deviation
            + self.w_drift_score
        )
        if not math.isclose(total, 1.0, rel_tol=1e-3):
            # Normalize to sum to 1.0
            self.w_anomaly_model /= total
            self.w_qber_deviation /= total
            self.w_gain_deviation /= total
            self.w_timing_deviation /= total
            self.w_drift_score /= total


@dataclass
class SignalFusionConfig:
    """Configuration for signal fusion engine."""
    fusion_method: FusionMethod = FusionMethod.PERSISTENCE_AWARE
    weights: SignalWeightConfig = field(default_factory=SignalWeightConfig)
    evidence_threshold: float = 0.50
    persistence_boost_factor: float = 0.10  # Max +50% boost for persisting signals
    max_persistence_steps: int = 5
    orthogonal_synergy_boost: float = 0.15  # Bonus when >= 3 subsystems deviate simultaneously
    # Normalization baselines
    nominal_qber: float = 0.015
    qber_tolerance: float = 0.010  # Std dev or margin
    nominal_gain: float = 0.045
    gain_tolerance: float = 0.015
    nominal_timing: float = 0.05
    timing_tolerance: float = 0.05


class SignalFusionEngine:
    """Fuses multi-subsystem telemetry and detector outputs into calibrated evidence."""

    def __init__(self, config: SignalFusionConfig | None = None):
        self.config = config or SignalFusionConfig()
        self._persistence_counter: int = 0
        self._prev_evidence: float = 0.0

    def normalize_score(self, raw_value: float, threshold: float) -> float:
        """Normalize an unbounded score against its operating threshold to [0.0, 1.0].

        Using calibrated piecewise linear-logistic transformation:
        - raw == 0 -> 0.0
        - raw == threshold -> 0.50
        - raw >= 2*threshold -> 1.00
        """
        if threshold <= 0.0:
            return 1.0 if raw_value > 0 else 0.0
        ratio = raw_value / threshold
        if ratio <= 1.0:
            return float(np.clip(0.50 * ratio, 0.0, 0.50))
        else:
            return float(np.clip(0.50 + 0.50 * (ratio - 1.0), 0.50, 1.0))

    def normalize_deviation(self, observed: float, nominal: float, tolerance: float) -> float:
        """Normalize a physical feature absolute deviation to [0.0, 1.0]."""
        if tolerance <= 0.0:
            return 0.0
        dev = abs(observed - nominal) / tolerance
        return float(np.clip(dev / 3.0, 0.0, 1.0))

    def evaluate(
        self,
        anomaly_score: float,
        anomaly_threshold: float,
        drift_score: float,
        drift_threshold: float,
        features: dict[str, float],
        timestamp_s: float,
        feature_contributions: dict[str, float] | None = None,
    ) -> tuple[float, list[DetectorSignal], dict[str, Any]]:
        """Compute the unified security evidence score and individual constituent signals.

        Returns:
            evidence_score: Scalar in [0.0, 1.0] representing security evidence
            signals: List of normalized DetectorSignal instances
            breakdown: Detailed dictionary explaining signal contributions
        """
        cfg = self.config
        w = cfg.weights

        # 1. Anomaly Model Signal
        norm_anom = self.normalize_score(anomaly_score, anomaly_threshold)
        is_anom = anomaly_score >= anomaly_threshold
        sig_anom = DetectorSignal(
            signal_name="temporal_anomaly_model",
            subsystem="ml_engine",
            raw_value=anomaly_score,
            normalized_value=norm_anom,
            threshold=anomaly_threshold,
            is_anomalous=is_anom,
            timestamp_s=timestamp_s,
            metadata={"ratio": anomaly_score / max(1e-6, anomaly_threshold)},
        )

        # 2. Drift Detector Signal
        norm_drift = self.normalize_score(drift_score, drift_threshold)
        is_drift = drift_score >= drift_threshold
        sig_drift = DetectorSignal(
            signal_name="statistical_drift",
            subsystem="drift_engine",
            raw_value=drift_score,
            normalized_value=norm_drift,
            threshold=drift_threshold,
            is_anomalous=is_drift,
            timestamp_s=timestamp_s,
            metadata={"ratio": drift_score / max(1e-6, drift_threshold)},
        )

        # 3. Physical QBER Signal (Quantum Channel Subsystem)
        qber_val = features.get("qber", cfg.nominal_qber)
        norm_qber = self.normalize_deviation(qber_val, cfg.nominal_qber, cfg.qber_tolerance)
        sig_qber = DetectorSignal(
            signal_name="qber_deviation",
            subsystem="quantum_channel",
            raw_value=qber_val,
            normalized_value=norm_qber,
            threshold=cfg.nominal_qber + 2.0 * cfg.qber_tolerance,
            is_anomalous=norm_qber >= 0.50,
            timestamp_s=timestamp_s,
        )

        # 4. Optical Gain / Sifted Rate Signal (Optical Receiver Subsystem)
        gain_val = features.get("signal_gain", cfg.nominal_gain)
        norm_gain = self.normalize_deviation(gain_val, cfg.nominal_gain, cfg.gain_tolerance)
        sig_gain = DetectorSignal(
            signal_name="optical_gain_deviation",
            subsystem="optical_receiver",
            raw_value=gain_val,
            normalized_value=norm_gain,
            threshold=cfg.nominal_gain + 2.0 * cfg.gain_tolerance,
            is_anomalous=norm_gain >= 0.50,
            timestamp_s=timestamp_s,
        )

        # 5. Timing / Autocorrelation Signal (Synchronization Subsystem)
        timing_val = features.get("clock_drift_rate", features.get("autocorr_lag1", cfg.nominal_timing))
        norm_timing = self.normalize_deviation(timing_val, cfg.nominal_timing, cfg.timing_tolerance)
        sig_timing = DetectorSignal(
            signal_name="timing_deviation",
            subsystem="synchronization",
            raw_value=timing_val,
            normalized_value=norm_timing,
            threshold=cfg.nominal_timing + 2.0 * cfg.timing_tolerance,
            is_anomalous=norm_timing >= 0.50,
            timestamp_s=timestamp_s,
        )

        signals = [sig_anom, sig_drift, sig_qber, sig_gain, sig_timing]

        # Base weighted evidence score (Method B)
        base_evidence = (
            w.w_anomaly_model * norm_anom
            + w.w_drift_score * norm_drift
            + w.w_qber_deviation * norm_qber
            + w.w_gain_deviation * norm_gain
            + w.w_timing_deviation * norm_timing
        )

        fused_evidence = base_evidence
        synergy_boost_applied = False
        persistence_boost_applied = False

        # Method A: Rule-based orthogonal synergy boost
        # If signals from >= 3 distinct subsystems are anomalous, apply synergy bonus
        anomalous_subsystems = {s.subsystem for s in signals if s.is_anomalous}
        if len(anomalous_subsystems) >= 3 and cfg.fusion_method in (FusionMethod.RULE_BASED, FusionMethod.PERSISTENCE_AWARE):
            fused_evidence += cfg.orthogonal_synergy_boost
            synergy_boost_applied = True

        # Method C: Persistence-aware evidence scaling
        if cfg.fusion_method == FusionMethod.PERSISTENCE_AWARE:
            if fused_evidence >= cfg.evidence_threshold or is_anom:
                self._persistence_counter += 1
            else:
                self._persistence_counter = max(0, self._persistence_counter - 1)

            if self._persistence_counter > 1:
                boost_steps = min(self._persistence_counter, cfg.max_persistence_steps)
                persist_multiplier = 1.0 + (boost_steps - 1) * cfg.persistence_boost_factor
                fused_evidence *= persist_multiplier
                persistence_boost_applied = True

        final_evidence = float(np.clip(fused_evidence, 0.0, 1.0))
        self._prev_evidence = final_evidence

        breakdown = {
            "base_evidence": float(base_evidence),
            "final_evidence": final_evidence,
            "fusion_method": cfg.fusion_method.value,
            "synergy_boost_applied": synergy_boost_applied,
            "persistence_boost_applied": persistence_boost_applied,
            "persistence_counter": self._persistence_counter,
            "anomalous_subsystems_count": len(anomalous_subsystems),
            "dominant_signals": sorted(
                [{"name": s.signal_name, "norm_val": s.normalized_value} for s in signals],
                key=lambda x: x["norm_val"],
                reverse=True,
            ),
            "feature_contributions": feature_contributions or {},
        }

        return final_evidence, signals, breakdown

    def reset(self) -> None:
        """Reset internal state."""
        self._persistence_counter = 0
        self._prev_evidence = 0.0
