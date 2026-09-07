"""Event-level telemetry schema.

Every pulse produces one structured event record. The schema is the contract
between the simulator and every downstream consumer (datasets, validation,
future ML). Columns are tagged with a ``FieldCategory`` so ground-truth-only
fields can never accidentally leak into an ML feature matrix.

Categories:
    OBSERVABLE_TELEMETRY    — what a deployed monitoring system could observe
    DERIVED_TELEMETRY       — computed from observables (sifting, click types)
    SIMULATOR_GROUND_TRUTH  — exact simulator internals (hidden photon number,
                              exact channel transmittance, exact parameters)
    FUTURE_ONLY             — reserved for V0.2+ (null/placeholder in V0.1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from qism.types import FieldCategory


@dataclass(frozen=True)
class FieldSpec:
    """Schema entry for one event-level column."""

    name: str
    dtype: str  # numpy dtype string used for storage
    category: FieldCategory
    description: str
    unit: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "category": self.category.value,
            "description": self.description,
            "unit": self.unit,
        }


# --------------------------------------------------------------------------- #
# Event schema (ordered; this ordering is preserved in Parquet output)
# --------------------------------------------------------------------------- #

EVENT_SCHEMA: dict[str, FieldSpec] = {
    spec.name: spec
    for spec in [
        # --- identifiers ---
        FieldSpec(
            "run_id",
            "string",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Identifier of the simulation run.",
        ),
        FieldSpec(
            "scenario_id",
            "string",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Scenario identifier (e.g. normal, calib_dist_25km).",
        ),
        FieldSpec(
            "window_id",
            "int64",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Telemetry window index (pulse_id // window_size).",
        ),
        FieldSpec(
            "pulse_id",
            "int64",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Pulse index within the run (0-based, contiguous).",
        ),
        FieldSpec(
            "timestamp",
            "float64",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Pulse emission timestamp (Alice clock) [s].",
            "s",
        ),
        # --- Alice / source ---
        FieldSpec("alice_bit", "int8", FieldCategory.OBSERVABLE_TELEMETRY, "Alice's bit (0 or 1)."),
        FieldSpec(
            "alice_basis", "int8", FieldCategory.OBSERVABLE_TELEMETRY, "Alice's basis (0=Z, 1=X)."
        ),
        FieldSpec(
            "intensity_class",
            "string",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Decoy-state class: signal | decoy | vacuum.",
        ),
        FieldSpec(
            "mu",
            "float64",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Mean photon number of this pulse's class.",
        ),
        FieldSpec(
            "photon_number",
            "int16",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "Sampled photon number (hidden ground truth).",
        ),
        # --- channel ---
        FieldSpec(
            "channel_transmittance",
            "float64",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "Exact channel transmittance eta_ch (hidden ground truth).",
        ),
        FieldSpec(
            "channel_delay_s",
            "float64",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "Exact propagation delay n_eff*L/c (hidden ground truth).",
            "s",
        ),
        FieldSpec(
            "polarization_misalignment",
            "float64",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "Exact p_mis = sin^2(theta) (hidden ground truth).",
        ),
        # --- Bob / measurement ---
        FieldSpec(
            "bob_basis", "int8", FieldCategory.OBSERVABLE_TELEMETRY, "Bob's basis (0=Z, 1=X)."
        ),
        FieldSpec(
            "detector_0_click",
            "bool",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Detector D0 clicked (signal and/or dark).",
        ),
        FieldSpec(
            "detector_1_click",
            "bool",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Detector D1 clicked (signal and/or dark).",
        ),
        FieldSpec(
            "double_click", "bool", FieldCategory.DERIVED_TELEMETRY, "Both detectors clicked."
        ),
        FieldSpec(
            "click_type",
            "string",
            FieldCategory.DERIVED_TELEMETRY,
            "Combined click outcome (see qism.types.ClickType).",
        ),
        FieldSpec(
            "detector_id",
            "float64",
            FieldCategory.DERIVED_TELEMETRY,
            "Clicked detector (0/1); -1 for double click; NaN for none.",
        ),
        # --- timing ---
        FieldSpec(
            "expected_arrival_time_s",
            "float64",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Expected arrival = pulse timestamp + channel delay [s].",
            "s",
        ),
        FieldSpec(
            "arrival_time_s",
            "float64",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Measured arrival of the clicked detector (NaN if no click) [s].",
            "s",
        ),
        FieldSpec(
            "timing_offset_s",
            "float64",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "arrival - expected (NaN if no click) [s].",
            "s",
        ),
        FieldSpec(
            "gate_width_s",
            "float64",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Detection gate width (config) [s].",
            "s",
        ),
        # --- detector parameters (per clicked detector) ---
        FieldSpec(
            "detector_efficiency",
            "float64",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "Efficiency of the clicked detector (NaN if none).",
        ),
        FieldSpec(
            "dark_count_rate",
            "float64",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "Dark-count rate of the clicked detector [Hz] (NaN if none).",
            "Hz",
        ),
        FieldSpec(
            "detector_state",
            "string",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Detector state at the pulse ('ready'; 'dead' from V0.2).",
        ),
        # --- sifting ---
        FieldSpec(
            "sifted", "bool", FieldCategory.DERIVED_TELEMETRY, "Alice and Bob used the same basis."
        ),
        FieldSpec(
            "bit_error",
            "float64",
            FieldCategory.DERIVED_TELEMETRY,
            "Bit error (1/0) for sifted single-click events; NaN otherwise.",
        ),
        # --- future-only fields (V0.2+) ---
        FieldSpec(
            "attack_active",
            "bool",
            FieldCategory.FUTURE_ONLY,
            "Whether an attack is active for this pulse (always False in V0.1).",
        ),
        FieldSpec(
            "attack_type",
            "string",
            FieldCategory.FUTURE_ONLY,
            "Attack identifier ('none' in V0.1).",
        ),
        FieldSpec(
            "attack_strength",
            "float64",
            FieldCategory.FUTURE_ONLY,
            "Attack strength parameter (0.0 in V0.1).",
        ),
        FieldSpec(
            "attack_id", "string", FieldCategory.FUTURE_ONLY, "Attack instance id (null in V0.1)."
        ),
        FieldSpec(
            "attack_phase",
            "string",
            FieldCategory.FUTURE_ONLY,
            "Attack lifecycle phase: normal | onset | active | terminating | recovery.",
        ),
        FieldSpec(
            "environment_state",
            "string",
            FieldCategory.FUTURE_ONLY,
            "Environment state label ('stable' in V0.1).",
        ),
        FieldSpec(
            "temperature_c",
            "float64",
            FieldCategory.FUTURE_ONLY,
            "Temperature [C] (null in V0.1).",
            "C",
        ),
        FieldSpec(
            "bias_voltage_v",
            "float64",
            FieldCategory.FUTURE_ONLY,
            "Detector bias voltage [V] (null in V0.1).",
            "V",
        ),
        FieldSpec(
            "optical_power_dbm",
            "float64",
            FieldCategory.FUTURE_ONLY,
            "Monitored optical power [dBm] (null in V0.1).",
            "dBm",
        ),
        FieldSpec(
            "recovery_time_s",
            "float64",
            FieldCategory.FUTURE_ONLY,
            "Detector recovery time [s] (null in V0.1).",
            "s",
        ),
        # --- V0.2 detector dynamics (filled by the stateful engine; the V0.1
        # vectorized path records neutral defaults so every frame matches the
        # schema) ---
        FieldSpec(
            "detector_0_state",
            "string",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Detector D0 state before this pulse's candidates ('ready'/'dead').",
        ),
        FieldSpec(
            "detector_1_state",
            "string",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Detector D1 state before this pulse's candidates ('ready'/'dead').",
        ),
        FieldSpec(
            "detector_0_dead_time_remaining_s",
            "float64",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Remaining dead time for D0 before recovery (0 when READY) [s].",
            "s",
        ),
        FieldSpec(
            "detector_1_dead_time_remaining_s",
            "float64",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Remaining dead time for D1 before recovery (0 when READY) [s].",
            "s",
        ),
        FieldSpec(
            "detector_0_time_since_last_click_s",
            "float64",
            FieldCategory.DERIVED_TELEMETRY,
            "Time since D0's last accepted avalanche at this pulse [s].",
            "s",
        ),
        FieldSpec(
            "detector_1_time_since_last_click_s",
            "float64",
            FieldCategory.DERIVED_TELEMETRY,
            "Time since D1's last accepted avalanche at this pulse [s].",
            "s",
        ),
        FieldSpec(
            "detector_0_dead_duration_s",
            "float64",
            FieldCategory.DERIVED_TELEMETRY,
            "Effective dead time applied after D0's avalanche (NaN if D0 did not click) [s].",
            "s",
        ),
        FieldSpec(
            "detector_1_dead_duration_s",
            "float64",
            FieldCategory.DERIVED_TELEMETRY,
            "Effective dead time applied after D1's avalanche (NaN if D1 did not click) [s].",
            "s",
        ),
        FieldSpec(
            "detector_0_arrival_time_s",
            "float64",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "D0 avalanche time inside this pulse's gate (NaN if D0 did not click) [s].",
            "s",
        ),
        FieldSpec(
            "detector_1_arrival_time_s",
            "float64",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "D1 avalanche time inside this pulse's gate (NaN if D1 did not click) [s].",
            "s",
        ),
        FieldSpec(
            "dead_time_loss",
            "bool",
            FieldCategory.DERIVED_TELEMETRY,
            "At least one candidate was rejected because a detector was DEAD.",
        ),
        FieldSpec(
            "detector_0_candidate_count",
            "int16",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "Candidates offered to D0 in this pulse (signal + dark + afterpulse).",
        ),
        FieldSpec(
            "detector_1_candidate_count",
            "int16",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "Candidates offered to D1 in this pulse (signal + dark + afterpulse).",
        ),
        FieldSpec(
            "detector_0_dead_rejected_count",
            "int16",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "D0 candidates rejected because D0 was DEAD (dead-time loss).",
        ),
        FieldSpec(
            "detector_1_dead_rejected_count",
            "int16",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "D1 candidates rejected because D1 was DEAD (dead-time loss).",
        ),
        FieldSpec(
            "afterpulse_candidate",
            "bool",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "An afterpulse candidate was attempted in this pulse (GT: real monitors "
            "cannot label exact afterpulses without detector-internal info).",
        ),
        FieldSpec(
            "afterpulse_detected",
            "bool",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "The afterpulse candidate produced a click (GT).",
        ),
        FieldSpec(
            "afterpulse_parent_pulse_id",
            "float64",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "Pulse of the parent avalanche (NaN if none) (GT).",
        ),
        FieldSpec(
            "afterpulse_delay_s",
            "float64",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "t_afterpulse - t_parent (NaN if none) (GT).",
            "s",
        ),
        # --- V0.4 Infrastructure & Environmental Telemetry ---
        FieldSpec(
            "sensor_temperature_c",
            "float64",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Monitoring sensor temperature reading [C].",
            "C",
        ),
        FieldSpec(
            "sensor_humidity_pct",
            "float64",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Monitoring sensor relative humidity reading [%].",
            "%",
        ),
        FieldSpec(
            "sensor_vibration_g",
            "float64",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Monitoring sensor vibration RMS reading [g].",
            "g",
        ),
        FieldSpec(
            "sensor_background_light_lux",
            "float64",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Monitoring sensor ambient light reading [lux].",
            "lux",
        ),
        FieldSpec(
            "clock_offset_s",
            "float64",
            FieldCategory.OBSERVABLE_TELEMETRY,
            "Local clock phase offset [s].",
            "s",
        ),
        FieldSpec(
            "true_temperature_c",
            "float64",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "Exact physical temperature (GT) [C].",
            "C",
        ),
        FieldSpec(
            "true_humidity_pct",
            "float64",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "Exact physical relative humidity (GT) [%].",
            "%",
        ),
        FieldSpec(
            "true_vibration_g",
            "float64",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "Exact physical vibration acceleration (GT) [g].",
            "g",
        ),
        FieldSpec(
            "true_background_photon_rate_hz",
            "float64",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "Exact stray background photon arrival rate (GT) [Hz].",
            "Hz",
        ),
        FieldSpec(
            "true_channel_loss_db",
            "float64",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "Exact dynamic channel attenuation (GT) [dB].",
            "dB",
        ),
        FieldSpec(
            "true_polarization_drift_deg",
            "float64",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "Exact polarization misalignment angle (GT) [deg].",
            "deg",
        ),
        FieldSpec(
            "background_detected",
            "bool",
            FieldCategory.SIMULATOR_GROUND_TRUTH,
            "A background stray photon triggered a detection click (GT).",
        ),
    ]
}

EVENT_COLUMNS: list[str] = list(EVENT_SCHEMA.keys())

# Columns that must never enter an ML feature matrix.
ML_EXCLUDED_CATEGORIES = {FieldCategory.SIMULATOR_GROUND_TRUTH, FieldCategory.FUTURE_ONLY}
IDENTIFIER_COLUMNS = {"run_id", "scenario_id", "window_id", "pulse_id"}


def ml_input_columns() -> list[str]:
    """Event-level columns acceptable as ML inputs (observable/derived only)."""
    return [
        name
        for name, spec in EVENT_SCHEMA.items()
        if spec.category not in ML_EXCLUDED_CATEGORIES and name not in IDENTIFIER_COLUMNS
    ]


# --------------------------------------------------------------------------- #
# Typed event record
# --------------------------------------------------------------------------- #


@dataclass
class QKDEvent:
    """Typed per-pulse event record (the canonical V0.1 event object)."""

    run_id: str = ""
    scenario_id: str = ""
    window_id: int = 0
    pulse_id: int = 0
    timestamp: float = 0.0
    alice_bit: int = 0
    alice_basis: int = 0
    intensity_class: str = "signal"
    mu: float = 0.0
    photon_number: int = 0
    bob_basis: int = 0
    channel_transmittance: float = 1.0
    channel_delay_s: float = 0.0
    polarization_misalignment: float = 0.0
    detector_0_click: bool = False
    detector_1_click: bool = False
    double_click: bool = False
    click_type: str = "none"
    detector_id: float | None = None
    expected_arrival_time_s: float = 0.0
    arrival_time_s: float | None = None
    timing_offset_s: float | None = None
    gate_width_s: float = 0.0
    detector_efficiency: float | None = None
    dark_count_rate: float | None = None
    detector_state: str = "ready"
    sifted: bool = False
    bit_error: float | None = None
    attack_active: bool = False
    attack_type: str = "none"
    attack_strength: float = 0.0
    attack_id: str | None = None
    attack_phase: str = "normal"
    environment_state: str = "stable"
    temperature_c: float | None = None
    bias_voltage_v: float | None = None
    optical_power_dbm: float | None = None
    recovery_time_s: float | None = None
    # V0.2 detector dynamics
    detector_0_state: str = "ready"
    detector_1_state: str = "ready"
    detector_0_dead_time_remaining_s: float = 0.0
    detector_1_dead_time_remaining_s: float = 0.0
    detector_0_time_since_last_click_s: float | None = None
    detector_1_time_since_last_click_s: float | None = None
    detector_0_dead_duration_s: float | None = None
    detector_1_dead_duration_s: float | None = None
    detector_0_arrival_time_s: float | None = None
    detector_1_arrival_time_s: float | None = None
    dead_time_loss: bool = False
    detector_0_candidate_count: int = 0
    detector_1_candidate_count: int = 0
    detector_0_dead_rejected_count: int = 0
    detector_1_dead_rejected_count: int = 0
    afterpulse_candidate: bool = False
    afterpulse_detected: bool = False
    afterpulse_parent_pulse_id: float | None = None
    afterpulse_delay_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: pd.Series) -> QKDEvent:
        values = {name: row.get(name) for name in EVENT_COLUMNS}
        # keep only known fields plus any extras
        return cls(**values)


# --------------------------------------------------------------------------- #
# Frame builder
# --------------------------------------------------------------------------- #


def events_frame(
    data: dict[str, Any],
    run_id: str,
    scenario_id: str,
    window_size: int,
) -> pd.DataFrame:
    """Assemble a typed event DataFrame from a dict of equal-length arrays.

    The dict must contain (at least) ``pulse_id`` plus every physics column;
    missing future-only columns are filled with V0.1 defaults. Column order
    and dtypes follow ``EVENT_SCHEMA``.
    """
    n = len(data["pulse_id"])
    pulse_ids = np.asarray(data["pulse_id"])

    filled: dict[str, Any] = {}
    for name, spec in EVENT_SCHEMA.items():
        if name in data:
            filled[name] = data[name]
        elif spec.category is FieldCategory.FUTURE_ONLY or name in _V02_NEUTRAL_DEFAULTS:
            filled[name] = _future_default(name, n)
        elif name == "run_id":
            filled[name] = run_id
        elif name == "scenario_id":
            filled[name] = scenario_id
        elif name == "window_id":
            filled[name] = pulse_ids // window_size
        else:
            raise KeyError(f"Missing required event field: {name!r}")

    df = pd.DataFrame(filled)
    # apply dtypes
    for name, spec in EVENT_SCHEMA.items():
        try:
            df[name] = df[name].astype(spec.dtype)
        except (ValueError, TypeError):
            # e.g. string column with None -> keep as object; pandas handles it
            pass
    return df[EVENT_COLUMNS]


def _future_default(name: str, n: int) -> Any:
    defaults: dict[str, Any] = {
        "attack_active": np.zeros(n, dtype=bool),
        "attack_type": np.full(n, "none", dtype=object),
        "attack_strength": np.zeros(n, dtype=np.float64),
        "attack_id": np.full(n, "none", dtype=object),
        "attack_phase": np.full(n, "normal", dtype=object),
        "environment_state": np.full(n, "stable", dtype=object),
        "temperature_c": np.full(n, np.nan, dtype=np.float64),
        "bias_voltage_v": np.full(n, np.nan, dtype=np.float64),
        "optical_power_dbm": np.full(n, np.nan, dtype=np.float64),
        "recovery_time_s": np.full(n, np.nan, dtype=np.float64),
        # neutral V0.2 values for the (stateless) V0.1 vectorized path
        "detector_0_state": np.full(n, "ready", dtype=object),
        "detector_1_state": np.full(n, "ready", dtype=object),
        "detector_0_dead_time_remaining_s": np.zeros(n, dtype=np.float64),
        "detector_1_dead_time_remaining_s": np.zeros(n, dtype=np.float64),
        "detector_0_time_since_last_click_s": np.full(n, np.nan, dtype=np.float64),
        "detector_1_time_since_last_click_s": np.full(n, np.nan, dtype=np.float64),
        "detector_0_dead_duration_s": np.full(n, np.nan, dtype=np.float64),
        "detector_1_dead_duration_s": np.full(n, np.nan, dtype=np.float64),
        "detector_0_arrival_time_s": np.full(n, np.nan, dtype=np.float64),
        "detector_1_arrival_time_s": np.full(n, np.nan, dtype=np.float64),
        "dead_time_loss": np.zeros(n, dtype=bool),
        "detector_0_candidate_count": np.zeros(n, dtype=np.int16),
        "detector_1_candidate_count": np.zeros(n, dtype=np.int16),
        "detector_0_dead_rejected_count": np.zeros(n, dtype=np.int16),
        "detector_1_dead_rejected_count": np.zeros(n, dtype=np.int16),
        "afterpulse_candidate": np.zeros(n, dtype=bool),
        "afterpulse_detected": np.zeros(n, dtype=bool),
        "afterpulse_parent_pulse_id": np.full(n, np.nan, dtype=np.float64),
        "afterpulse_delay_s": np.full(n, np.nan, dtype=np.float64),
        # V0.4 infrastructure & environmental telemetry defaults
        "sensor_temperature_c": np.full(n, np.nan, dtype=np.float64),
        "sensor_humidity_pct": np.full(n, np.nan, dtype=np.float64),
        "sensor_vibration_g": np.full(n, np.nan, dtype=np.float64),
        "sensor_background_light_lux": np.full(n, np.nan, dtype=np.float64),
        "clock_offset_s": np.zeros(n, dtype=np.float64),
        "true_temperature_c": np.full(n, np.nan, dtype=np.float64),
        "true_humidity_pct": np.full(n, np.nan, dtype=np.float64),
        "true_vibration_g": np.full(n, np.nan, dtype=np.float64),
        "true_background_photon_rate_hz": np.zeros(n, dtype=np.float64),
        "true_channel_loss_db": np.full(n, np.nan, dtype=np.float64),
        "true_polarization_drift_deg": np.full(n, np.nan, dtype=np.float64),
        "background_detected": np.zeros(n, dtype=bool),
    }
    if name not in defaults:
        raise KeyError(f"Missing required event field: {name!r}")
    return defaults[name]


#: V0.2 / V0.4 columns that receive neutral defaults when the (stateless) V0.1
#: vectorized engine builds a frame without providing them.
_V02_NEUTRAL_DEFAULTS = frozenset(
    name
    for name in (
        "detector_0_state",
        "detector_1_state",
        "detector_0_dead_time_remaining_s",
        "detector_1_dead_time_remaining_s",
        "detector_0_time_since_last_click_s",
        "detector_1_time_since_last_click_s",
        "detector_0_dead_duration_s",
        "detector_1_dead_duration_s",
        "detector_0_arrival_time_s",
        "detector_1_arrival_time_s",
        "dead_time_loss",
        "detector_0_candidate_count",
        "detector_1_candidate_count",
        "detector_0_dead_rejected_count",
        "detector_1_dead_rejected_count",
        "afterpulse_candidate",
        "afterpulse_detected",
        "afterpulse_parent_pulse_id",
        "afterpulse_delay_s",
        "sensor_temperature_c",
        "sensor_humidity_pct",
        "sensor_vibration_g",
        "sensor_background_light_lux",
        "clock_offset_s",
        "true_temperature_c",
        "true_humidity_pct",
        "true_vibration_g",
        "true_background_photon_rate_hz",
        "true_channel_loss_db",
        "true_polarization_drift_deg",
        "background_detected",
    )
)
