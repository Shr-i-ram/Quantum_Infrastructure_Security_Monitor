"""Controlled physical concept drift and distribution shift scenarios (V0.8).

Models legitimate operational changes using actual physical parameters simulated in QISM:
1. Optical fiber attenuation / channel loss drift (dB)
2. Detector thermal drift and coupled dark count rate
3. Laser source pump intensity fluctuation (mu)
4. Environmental ambient coupling

Supports:
- STATIONARY: Baseline unperturbed normal operation
- ABRUPT_BENIGN: Abrupt shift A -> B at a specific time step
- GRADUAL_BENIGN: Linear/smooth continuous evolution A -> B over an interval
- RECURRING_BENIGN: Reversible shift A -> B -> A
- MULTI_CONDITION: Coupled simultaneous changes in attenuation and temperature
- DRIFT_THEN_ATTACK: Benign shift followed by physical attack in condition B
- ATTACK_DURING_DRIFT: Physical attack occurring during distribution transition
- CONTAMINATION_STRESS: Physical attack initiating precisely inside the candidate adaptation window
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

import numpy as np
import pandas as pd

from qism.random import derive_scenario_seed


class DriftScenarioType(str, Enum):
    """Supported drift scenario types."""
    STATIONARY = "stationary"
    ABRUPT_BENIGN = "abrupt_benign"
    GRADUAL_BENIGN = "gradual_benign"
    RECURRING_BENIGN = "recurring_benign"
    MULTI_CONDITION = "multi_condition"
    ATTACK_ONLY = "attack_only"
    DRIFT_THEN_ATTACK = "drift_then_attack"
    ATTACK_DURING_DRIFT = "attack_during_drift"
    CONTAMINATION_STRESS = "contamination_stress"
    OOD_GENERALIZATION = "ood_generalization"


@dataclass
class ScenarioSpec:
    """Specification parameters for a drift simulation stream."""
    scenario_type: DriftScenarioType
    total_steps: int = 150
    step_duration_s: float = 0.002 # 2.0 ms per aggregation window (2,000 pulses at 1 MHz)
    drift_start_step: int = 40
    drift_end_step: int = 70
    recovery_start_step: int = 100
    recovery_end_step: int = 120
    attack_start_step: int = 100
    attack_end_step: int = 130
    attack_type: str = "intercept_resend"
    attack_strength: float = 0.80
    channel_loss_delta_db: float = 2.0 # Legitimate channel loss change (+2 dB)
    temp_delta_c: float = 5.0          # Legitimate detector temperature change (+5 C)
    seed: int = 42


class DriftScenarioGenerator:
    """Generates synthetic and physics-coupled telemetry streams with controlled drift schedules."""

    def __init__(self, spec: ScenarioSpec, base_telemetry_df: pd.DataFrame | None = None):
        self.spec = spec
        self.base_df = base_telemetry_df

    def generate_stream(self) -> tuple[pd.DataFrame, list[str]]:
        """Generate a complete chronological telemetry stream with ground truth sidecars.

        Returns:
            df: DataFrame containing all observable features plus sidecar ground-truth columns.
            feature_cols: List of column names representing purely observable telemetry.
        """
        rng = np.random.RandomState(self.spec.seed)
        N = self.spec.total_steps
        t_step = self.spec.step_duration_s

        # Observable telemetry feature definitions matching QISM V0.3–V0.7
        feature_cols = [
            "qber",
            "qber_z",
            "qber_x",
            "sifted_rate",
            "signal_gain",
            "decoy_gain",
            "vacuum_gain",
            "detector_imbalance",
            "double_click_rate",
            "dark_count_rate",
            "sensor_temperature_mean",
            "sensor_temperature_std",
            "sensor_background_light_mean",
            "sensor_vibration_mean",
            "clock_drift_rate",
            "autocorr_lag1",
        ]

        # 1. Base baseline values for Condition A (Normal Healthy Baseline)
        base_values = {
            "qber": 0.015,
            "qber_z": 0.015,
            "qber_x": 0.015,
            "sifted_rate": 850.0,
            "signal_gain": 0.045,
            "decoy_gain": 0.012,
            "vacuum_gain": 0.0008,
            "detector_imbalance": 0.002,
            "double_click_rate": 0.0004,
            "dark_count_rate": 25.0,
            "sensor_temperature_mean": 21.0,
            "sensor_temperature_std": 0.15,
            "sensor_background_light_mean": 12.0,
            "sensor_vibration_mean": 0.012,
            "clock_drift_rate": 0.05,
            "autocorr_lag1": 0.02,
        }

        # 2. Compute drift schedule factor lambda_drift(t) in [0, 1]
        drift_factor = np.zeros(N, dtype=np.float64)
        is_drift = np.zeros(N, dtype=bool)
        op_condition = ["Condition_A"] * N

        st = self.spec.scenario_type

        for i in range(N):
            if st == DriftScenarioType.STATIONARY or st == DriftScenarioType.ATTACK_ONLY:
                drift_factor[i] = 0.0
                is_drift[i] = False
                op_condition[i] = "Condition_A"

            elif st in (DriftScenarioType.ABRUPT_BENIGN, DriftScenarioType.DRIFT_THEN_ATTACK):
                if i >= self.spec.drift_start_step:
                    drift_factor[i] = 1.0
                    is_drift[i] = True
                    op_condition[i] = "Condition_B"
                else:
                    drift_factor[i] = 0.0
                    is_drift[i] = False
                    op_condition[i] = "Condition_A"

            elif st in (DriftScenarioType.GRADUAL_BENIGN, DriftScenarioType.ATTACK_DURING_DRIFT, DriftScenarioType.CONTAMINATION_STRESS):
                if i < self.spec.drift_start_step:
                    drift_factor[i] = 0.0
                    is_drift[i] = False
                    op_condition[i] = "Condition_A"
                elif i <= self.spec.drift_end_step:
                    # Linear gradual ramp from 0 to 1
                    frac = (i - self.spec.drift_start_step) / max(1, (self.spec.drift_end_step - self.spec.drift_start_step))
                    drift_factor[i] = float(frac)
                    is_drift[i] = True
                    op_condition[i] = "Transition_A_to_B"
                else:
                    drift_factor[i] = 1.0
                    is_drift[i] = True
                    op_condition[i] = "Condition_B"

            elif st == DriftScenarioType.RECURRING_BENIGN:
                if i < self.spec.drift_start_step:
                    drift_factor[i] = 0.0
                    is_drift[i] = False
                    op_condition[i] = "Condition_A"
                elif i < self.spec.recovery_start_step:
                    drift_factor[i] = 1.0
                    is_drift[i] = True
                    op_condition[i] = "Condition_B"
                elif i <= self.spec.recovery_end_step:
                    # Return ramp from 1 to 0
                    frac = 1.0 - (i - self.spec.recovery_start_step) / max(1, (self.spec.recovery_end_step - self.spec.recovery_start_step))
                    drift_factor[i] = float(frac)
                    is_drift[i] = True
                    op_condition[i] = "Transition_B_to_A"
                else:
                    drift_factor[i] = 0.0
                    is_drift[i] = False
                    op_condition[i] = "Condition_A"

            elif st == DriftScenarioType.MULTI_CONDITION or st == DriftScenarioType.OOD_GENERALIZATION:
                if i >= self.spec.drift_start_step:
                    drift_factor[i] = 1.5 if st == DriftScenarioType.OOD_GENERALIZATION else 1.0
                    is_drift[i] = True
                    op_condition[i] = "Condition_B_Multi"
                else:
                    drift_factor[i] = 0.0
                    is_drift[i] = False
                    op_condition[i] = "Condition_A"

        # 3. Compute physical modifications from drift
        # Channel attenuation delta: +2 dB reduces transmittance by factor 10^(-2/10) = 0.631
        loss_ratio = 10.0 ** (-self.spec.channel_loss_delta_db / 10.0) # ~0.63
        # Temperature delta: +5 C increases dark count rate by factor 2^(5/7) = ~1.64
        dark_ratio = 2.0 ** (self.spec.temp_delta_c / 7.0) # ~1.64

        # 4. Compute attack schedule
        is_attack = np.zeros(N, dtype=bool)
        attack_strength = np.zeros(N, dtype=np.float64)
        attack_type = ["none"] * N

        has_attack = st in (
            DriftScenarioType.ATTACK_ONLY,
            DriftScenarioType.DRIFT_THEN_ATTACK,
            DriftScenarioType.ATTACK_DURING_DRIFT,
            DriftScenarioType.CONTAMINATION_STRESS,
        )

        if has_attack:
            a_start = self.spec.attack_start_step
            a_end = self.spec.attack_end_step
            if st == DriftScenarioType.ATTACK_DURING_DRIFT:
                # Attack overlaps directly with drift transition
                a_start = self.spec.drift_start_step + 5
                a_end = a_start + 25
            elif st == DriftScenarioType.CONTAMINATION_STRESS:
                # Attack initiates right when adaptation buffer is accumulating (4 steps after drift onset)
                a_start = self.spec.drift_start_step + 3
                a_end = a_start + 25

            for i in range(N):
                if a_start <= i <= a_end:
                    is_attack[i] = True
                    attack_strength[i] = self.spec.attack_strength
                    attack_type[i] = self.spec.attack_type

        # 5. Synthesize feature matrices with physical couplings and stochastic noise
        data: dict[str, Any] = {}

        for feat in feature_cols:
            base_val = base_values[feat]
            noise_scale = base_val * 0.05 if base_val > 0 else 0.001
            noise = rng.normal(loc=0.0, scale=noise_scale, size=N)

            val = np.full(N, base_val, dtype=np.float64)

            # Apply legitimate drift couplings
            if feat in ("signal_gain", "decoy_gain", "sifted_rate"):
                # Proportional to channel transmittance
                transmittance_factor = 1.0 + (loss_ratio - 1.0) * drift_factor
                val = val * transmittance_factor
            elif feat == "sensor_temperature_mean":
                val = val + self.spec.temp_delta_c * drift_factor
            elif feat in ("dark_count_rate", "vacuum_gain"):
                val = val * (1.0 + (dark_ratio - 1.0) * drift_factor)
            elif feat == "sensor_background_light_mean":
                val = val + 8.0 * drift_factor
            elif feat == "detector_imbalance":
                val = val + 0.015 * drift_factor

            # Apply attack perturbations
            for i in range(N):
                if is_attack[i]:
                    atk_str = attack_strength[i]
                    if self.spec.attack_type == "intercept_resend":
                        # Intercept-resend dramatically increases QBER (~0.25) and disrupts autocorrelation
                        if feat in ("qber", "qber_z", "qber_x"):
                            val[i] += 0.23 * atk_str
                        elif feat == "sifted_rate":
                            val[i] *= (1.0 - 0.35 * atk_str)
                        elif feat == "autocorr_lag1":
                            val[i] += 0.45 * atk_str
                    elif self.spec.attack_type == "channel_loss":
                        # Malicious extra attenuation
                        if feat in ("signal_gain", "decoy_gain", "sifted_rate"):
                            val[i] *= 0.20
                    elif self.spec.attack_type == "detector_blinding":
                        if feat in ("signal_gain", "vacuum_gain"):
                            val[i] *= 10.0
                        elif feat == "double_click_rate":
                            val[i] += 0.05

            # Combine with noise
            data[feat] = np.maximum(0.0, val + noise)

        # 6. Assemble DataFrame with metadata and sidecar ground truth
        df = pd.DataFrame(data)
        df["step_index"] = np.arange(N)
        df["timestamp"] = np.arange(N) * t_step
        df["scenario_id"] = f"scenario_{st.value}"
        df["run_id"] = f"run_{self.spec.seed:04d}"

        # Sidecar ground-truth columns (strictly excluded from model features)
        df["ground_truth_is_drift"] = is_drift
        df["ground_truth_drift_type"] = st.value
        df["ground_truth_operating_condition"] = op_condition
        df["ground_truth_is_attack"] = is_attack
        df["ground_truth_attack_type"] = attack_type
        df["ground_truth_attack_strength"] = attack_strength
        df["ground_truth_attack_active"] = is_attack.astype(int)

        return df, feature_cols
