"""Strongly typed, validated configuration system for QISM.

Every physical parameter is accessible through configuration. There are no
scattered hard-coded constants in the simulation code:

    YAML  ->  Pydantic configuration object  ->  simulator components

Sections defined in V0.1 (``environment`` and ``attack`` exist with disabled
defaults so that V0.2–V0.6 can extend them without breaking the schema).
"""

from __future__ import annotations

import hashlib
import os
import pathlib
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from qism.types import DatasetSplit, DistributionType, SamplingStrategy
from qism.units import SPEED_OF_LIGHT_M_PER_S

# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #


class ProtocolConfig(BaseModel):
    """Top-level protocol parameters (decoy-state BB84)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field("decoy_bb84", description="Protocol identifier.")
    n_pulses: int = Field(100_000, gt=0, description="Total number of pulses to simulate.")
    repetition_rate_hz: float = Field(
        1_000_000.0, gt=0, description="Pulse repetition rate f_rep [Hz]."
    )

    @property
    def repetition_period_s(self) -> float:
        """T_rep = 1 / f_rep [s]."""
        return 1.0 / self.repetition_rate_hz


# --------------------------------------------------------------------------- #
# Source (Alice + decoy-state source)
# --------------------------------------------------------------------------- #


class SourceConfig(BaseModel):
    """Alice's bit/basis statistics and the decoy-state intensity settings."""

    model_config = ConfigDict(extra="forbid")

    bit_probability_zero: float = Field(0.5, ge=0.0, le=1.0, description="P(A=0).")
    basis_z_probability: float = Field(0.5, ge=0.0, le=1.0, description="P(B_A=Z).")
    signal_probability: float = Field(
        0.80, ge=0.0, le=1.0, description="P(signal intensity class)."
    )
    decoy_probability: float = Field(0.15, ge=0.0, le=1.0, description="P(decoy intensity class).")
    vacuum_probability: float = Field(
        0.05, ge=0.0, le=1.0, description="P(vacuum intensity class)."
    )
    mu_signal: float = Field(0.50, ge=0.0, description="Mean photon number of signal pulses.")
    mu_decoy: float = Field(0.10, ge=0.0, description="Mean photon number of decoy pulses.")
    mu_vacuum: float = Field(0.00, ge=0.0, description="Mean photon number of vacuum pulses.")
    max_photon_number: int = Field(
        10, ge=1, description="Reporting cutoff n_max for empirical yields Y_n."
    )
    intensity_jitter_std: float = Field(
        0.0, ge=0.0, description="Pulse-to-pulse Gaussian intensity jitter sigma_mu."
    )

    @model_validator(mode="after")
    def _check_intensity_probabilities(self) -> SourceConfig:
        total = self.signal_probability + self.decoy_probability + self.vacuum_probability
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "source: signal + decoy + vacuum probabilities must sum to 1, "
                f"got {self.signal_probability} + {self.decoy_probability} + "
                f"{self.vacuum_probability} = {total}"
            )
        return self


# --------------------------------------------------------------------------- #
# Channel
# --------------------------------------------------------------------------- #


class ChannelConfig(BaseModel):
    """Quantum channel (loss, polarization misalignment, propagation delay)."""

    model_config = ConfigDict(extra="forbid")

    distance_km: float = Field(25.0, ge=0.0, description="Fibre distance L [km].")
    attenuation_db_per_km: float = Field(
        0.20, ge=0.0, description="Fibre attenuation alpha [dB/km]."
    )
    insertion_loss_db: float = Field(1.0, ge=0.0, description="Insertion/system loss L_sys [dB].")
    misalignment_deg: float = Field(
        1.0, ge=0.0, le=90.0, description="Polarization misalignment theta [deg]."
    )
    n_eff: float = Field(1.468, ge=1.0, description="Effective refractive index of the fibre.")
    speed_of_light_m_per_s: float = Field(
        SPEED_OF_LIGHT_M_PER_S, gt=0.0, description="Speed of light in vacuum [m/s]."
    )
    loss_drift_enabled: bool = Field(
        False, description="Whether channel loss drifts dynamically."
    )

    @property
    def distance_m(self) -> float:
        return self.distance_km * 1e3

    @property
    def total_loss_db(self) -> float:
        """L_total = alpha * L + L_sys [dB]."""
        return self.attenuation_db_per_km * self.distance_km + self.insertion_loss_db


# --------------------------------------------------------------------------- #
# Background optical noise
# --------------------------------------------------------------------------- #


class BackgroundConfig(BaseModel):
    """In-band background optical noise coupled into Bob's detectors."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(True, description="Whether background optical noise is modelled.")
    rate_hz: float = Field(
        1000.0, ge=0.0, description="Background photon rate lambda_bg [photons/s]."
    )
    ambient_coupling_factor: float = Field(
        0.0, ge=0.0, description="Coupling coefficient from ambient lux to photon rate [Hz/lux]."
    )


# --------------------------------------------------------------------------- #
# Receiver (Bob)
# --------------------------------------------------------------------------- #


class DetectorConfig(BaseModel):
    """One single-photon detector of Bob's two-detector array."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field("D", description="Detector label (e.g. D0).")
    efficiency: float = Field(0.15, ge=0.0, le=1.0, description="Detection efficiency eta_d.")
    dark_count_rate_hz: float = Field(1000.0, ge=0.0, description="Dark count rate [Hz].")
    timing_jitter_s: float = Field(50e-12, ge=0.0, description="Timing jitter sigma_t [s].")
    temp_coeff_efficiency: float = Field(
        0.0, description="d(eta_d)/dT: temperature sensitivity of efficiency [1/C]."
    )
    temp_coeff_dark_count: float = Field(
        0.0, description="k_T in R_d(T) = R_d,0 * exp(k_T * (T - T0)) [1/C]."
    )
    t_ref_c: float = Field(20.0, description="Reference baseline temperature T_0 [C].")

    @property
    def dark_interval_s(self) -> float | None:
        return None  # placeholder; the effective detection interval is set by ReceiverConfig


class DeadTimeConfig(BaseModel):
    """Detector dead time and recovery (V0.2).

    Two models are supported:
      * ``fixed``:          tau_eff = tau_dead_s  (constant).
      * ``rate_dependent``: tau_eff(lambda) = tau_0_s + delta_tau_s *
                            lambda / (lambda + lambda_c_hz).

    ``lambda`` is the per-detector candidate-event rate estimated over the
    previous ``rate_estimation_window_pulses`` of the pulse train (or over the
    equivalent observation time in free-running validation experiments); it is
    never estimated from accepted clicks alone (that would bias tau_eff).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(True, description="Whether dead time is modelled (V0.2 default: on).")
    model: Literal["fixed", "rate_dependent"] = Field(
        "rate_dependent",
        description="Dead-time model: 'fixed' (tau_dead_s) or 'rate_dependent' "
        "(tau_eff = tau_0 + delta_tau * lambda/(lambda + lambda_c)).",
    )
    tau_dead_s: float = Field(
        50e-9, ge=0.0, description="Nominal dead time tau_dead [s] (used by the 'fixed' model)."
    )
    recovery_model: Literal["hard_dead_time"] = Field(
        "hard_dead_time",
        description="Recovery model. 'hard_dead_time': completely unavailable during the "
        "dead interval; the detector returns to READY when t >= dead_until. "
        "(Gradual/exponential recovery functions arrive in later versions.)",
    )
    tau_0_s: float = Field(
        50e-9, ge=0.0, description="Rate-dependent model: tau_eff at low rate [s]."
    )
    delta_tau_s: float = Field(
        50e-9, ge=0.0, description="Rate-dependent model: tau_eff increase at high rate [s]."
    )
    lambda_c_hz: float = Field(
        1e6, gt=0.0, description="Rate-dependent model: characteristic rate lambda_c [Hz]."
    )
    rate_estimation_window_pulses: int = Field(
        10_000, gt=0, description="Candidate-event rate estimation window (pulses)."
    )

    @property
    def rate_dependent(self) -> bool:
        return self.model == "rate_dependent"


class AfterpulseConfig(BaseModel):
    """Detector afterpulsing (V0.2).

    After every accepted primary avalanche (signal, dark or background origin),
    an afterpulse occurs with probability ``probability``; its delay is drawn
    from an exponential distribution with scale ``tau_afterpulse_s``. Only
    primary avalanches generate afterpulses (no second-generation chains in
    V0.2). An afterpulse candidate that arrives while the detector is DEAD is
    lost and does NOT extend the dead time.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(True, description="Whether afterpulsing is modelled (V0.2 default: on).")
    probability: float = Field(
        0.01, ge=0.0, le=1.0, description="P(afterpulse | primary avalanche) = p_afterpulse."
    )
    decay_model: Literal["exponential"] = Field(
        "exponential",
        description="Afterpulse delay model. 'exponential': delay ~ Exp(tau_afterpulse_s).",
    )
    tau_afterpulse_s: float = Field(
        100e-9, ge=0.0, description="Afterpulse time constant tau_AP [s]."
    )


class DetectorStateConfig(BaseModel):
    """Initial detector state machine configuration (V0.2)."""

    model_config = ConfigDict(extra="forbid")

    initial_state: Literal["ready"] = Field(
        "ready", description="State of every detector at t=0. (Future states: blinded/controlled.)"
    )


# Legacy V0.1 flat dead-time/afterpulse keys that DetectorArrayConfig accepted.
_LEGACY_DETECTOR_KEYS = (
    "dead_time_enabled",
    "dead_time_s",
    "afterpulse_enabled",
    "p_afterpulse",
    "tau_afterpulse_s",
)


class DetectorArrayConfig(BaseModel):
    """Bob's two-detector array plus dead-time/afterpulse settings.

    V0.2 uses nested ``dead_time`` / ``afterpulse`` / ``state`` blocks. V0.1
    configuration files that used the flat legacy keys
    (``detectors.dead_time_enabled`` etc.) keep loading unchanged: a
    ``model_validator(mode="before")`` migrates them into the nested form.
    """

    model_config = ConfigDict(extra="forbid")

    detector_0: DetectorConfig = Field(default_factory=lambda: DetectorConfig(label="D0"))
    detector_1: DetectorConfig = Field(default_factory=lambda: DetectorConfig(label="D1"))
    dead_time: DeadTimeConfig = Field(default_factory=DeadTimeConfig)
    afterpulse: AfterpulseConfig = Field(default_factory=AfterpulseConfig)
    state: DetectorStateConfig = Field(default_factory=DetectorStateConfig)

    @property
    def dead_time_enabled(self) -> bool:
        """Backwards-compatible property for V0.1 call sites."""
        return self.dead_time.enabled

    @property
    def afterpulse_enabled(self) -> bool:
        """Backwards-compatible property for V0.1 call sites."""
        return self.afterpulse.enabled

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_keys(cls, data: Any) -> Any:
        """Map the flat V0.1 keys onto the nested V0.2 structure."""
        if not isinstance(data, dict):
            return data
        legacy = {k: data.pop(k) for k in _LEGACY_DETECTOR_KEYS if k in data}
        if not legacy:
            return data
        if "dead_time" not in data and ("dead_time_enabled" in legacy or "dead_time_s" in legacy):
            dead = {"enabled": bool(legacy.get("dead_time_enabled", False))}
            if "dead_time_s" in legacy:
                # legacy field was a nominal dead time; keep 'fixed' semantics so
                # the same number is used, and only enable when requested.
                dead["tau_dead_s"] = float(legacy["dead_time_s"])
                dead.setdefault("model", "fixed")
            data["dead_time"] = dead
        if "afterpulse" not in data and (
            "afterpulse_enabled" in legacy or "p_afterpulse" in legacy
        ):
            ap = {"enabled": bool(legacy.get("afterpulse_enabled", False))}
            if "p_afterpulse" in legacy:
                ap["probability"] = float(legacy["p_afterpulse"])
            if "tau_afterpulse_s" in legacy:
                ap["tau_afterpulse_s"] = float(legacy["tau_afterpulse_s"])
            data["afterpulse"] = ap
        return data

    @property
    def dynamics_enabled(self) -> bool:
        """Whether any stateful detector dynamics (V0.2 engine) is active."""
        return self.dead_time.enabled or self.afterpulse.enabled


class ReceiverConfig(BaseModel):
    """Bob's receiver: basis selection, gate, dark-count integration interval."""

    model_config = ConfigDict(extra="forbid")

    basis_z_probability: float = Field(0.5, ge=0.0, le=1.0, description="P(B_B=Z).")
    gate_width_s: float = Field(1e-9, gt=0.0, description="Detection gate width W_gate [s].")
    detection_interval_s: float | None = Field(
        None,
        gt=0.0,
        description="Interval Delta_t over which dark/background counts are integrated. "
        "Defaults to the gate width.",
    )
    double_click_policy: Literal["exclude"] = Field(
        "exclude",
        description="V0.1 sifted-key handling of double clicks. 'exclude' removes "
        "double-click events from the sifted key / QBER while keeping their "
        "count in telemetry. (Other policies arrive in later versions.)",
    )

    @property
    def effective_detection_interval_s(self) -> float:
        """Delta_t used for dark-count/background integration (defaults to gate width)."""
        return (
            self.detection_interval_s
            if self.detection_interval_s is not None
            else self.gate_width_s
        )


# --------------------------------------------------------------------------- #
# Timing / telemetry misc
# --------------------------------------------------------------------------- #


class TimingConfig(BaseModel):
    """Timing-related telemetry settings."""

    model_config = ConfigDict(extra="forbid")

    histogram_bins: int = Field(20, ge=1, description="Number of bins for timing histograms.")


class ProcessDynamicsConfig(BaseModel):
    """Configuration for one physical continuous process (OU drift + diurnal cycle)."""

    model_config = ConfigDict(extra="forbid")

    baseline: float = Field(0.0, description="Baseline / equilibrium value mu.")
    theta: float = Field(0.01, ge=0.0, description="Mean-reversion rate theta [1/s].")
    sigma: float = Field(0.0, ge=0.0, description="Diffusion volatility sigma [unit/sqrt(s)].")
    diurnal_amplitude: float = Field(0.0, description="Diurnal/cyclic peak amplitude.")
    diurnal_period_s: float = Field(86400.0, gt=0.0, description="Diurnal period [s].")
    diurnal_phase_rad: float = Field(0.0, description="Diurnal initial phase [rad].")
    min_val: float | None = Field(None, description="Physical minimum value clamp.")
    max_val: float | None = Field(None, description="Physical maximum value clamp.")


class SensorConfig(BaseModel):
    """Configuration for one physical sensor in the monitoring suite."""

    model_config = ConfigDict(extra="forbid")

    noise_std: float = Field(0.0, ge=0.0, description="Measurement Gaussian noise standard deviation.")
    quantization_step: float | None = Field(None, gt=0.0, description="Sensor resolution / quantization Delta_q.")
    missing_probability: float = Field(0.0, ge=0.0, le=1.0, description="Probability of measurement dropout / NaN.")
    sampling_interval_s: float = Field(0.0, ge=0.0, description="Measurement update period (zero-order hold) [s].")
    min_reading: float | None = Field(None, description="Minimum sensor reading limit.")
    max_reading: float | None = Field(None, description="Maximum sensor reading limit.")


class EnvironmentSensorsConfig(BaseModel):
    """Sensor configurations for observable environmental telemetry."""

    model_config = ConfigDict(extra="forbid")

    temperature: SensorConfig = Field(default_factory=lambda: SensorConfig(quantization_step=0.1))
    humidity: SensorConfig = Field(default_factory=lambda: SensorConfig(quantization_step=0.5))
    vibration: SensorConfig = Field(default_factory=lambda: SensorConfig(quantization_step=0.01))
    background_light: SensorConfig = Field(default_factory=lambda: SensorConfig(quantization_step=1.0))


class EnvironmentalCouplingConfig(BaseModel):
    """Coupling factors from environmental parameters to hardware physics."""

    model_config = ConfigDict(extra="forbid")

    base_background_rate_hz: float = Field(1000.0, ge=0.0, description="Base background optical photon rate [Hz].")
    ambient_light_to_bg_rate: float = Field(0.5, ge=0.0, description="Coupling factor: photons/s per lux of ambient light.")


class EnvironmentConfig(BaseModel):
    """Environmental/operational variation model (V0.4)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["stable", "dynamic", "ou_drift", "diurnal", "composite"] = Field(
        "stable",
        description="Environment model: 'stable' (constant operating point) or 'dynamic' (V0.4).",
    )
    enabled: bool = Field(
        False, description="Whether environmental variation is active (V0.1/V0.3 default: False)."
    )
    temperature: ProcessDynamicsConfig = Field(
        default_factory=lambda: ProcessDynamicsConfig(baseline=20.0, theta=0.001, sigma=0.01, min_val=-20.0, max_val=60.0)
    )
    humidity: ProcessDynamicsConfig = Field(
        default_factory=lambda: ProcessDynamicsConfig(baseline=45.0, theta=0.001, sigma=0.05, min_val=0.0, max_val=100.0)
    )
    vibration: ProcessDynamicsConfig = Field(
        default_factory=lambda: ProcessDynamicsConfig(baseline=0.02, theta=0.05, sigma=0.005, min_val=0.0, max_val=2.0)
    )
    background_light: ProcessDynamicsConfig = Field(
        default_factory=lambda: ProcessDynamicsConfig(baseline=100.0, theta=0.001, sigma=1.0, min_val=0.0)
    )
    polarization_drift: ProcessDynamicsConfig = Field(
        default_factory=lambda: ProcessDynamicsConfig(baseline=1.0, theta=0.002, sigma=0.02, min_val=0.0, max_val=90.0)
    )
    clock_drift: ProcessDynamicsConfig = Field(
        default_factory=lambda: ProcessDynamicsConfig(baseline=0.0, theta=0.01, sigma=0.05, min_val=-50.0, max_val=50.0)
    )
    channel_loss_drift: ProcessDynamicsConfig = Field(
        default_factory=lambda: ProcessDynamicsConfig(baseline=0.0, theta=0.001, sigma=0.02, min_val=-5.0, max_val=20.0)
    )
    source_intensity_drift: ProcessDynamicsConfig = Field(
        default_factory=lambda: ProcessDynamicsConfig(baseline=0.0, theta=0.005, sigma=0.01, min_val=-0.2, max_val=0.5)
    )
    detector_efficiency_drift: ProcessDynamicsConfig = Field(
        default_factory=lambda: ProcessDynamicsConfig(baseline=0.0, theta=0.005, sigma=0.005, min_val=-0.1, max_val=0.1)
    )
    sensors: EnvironmentSensorsConfig = Field(default_factory=EnvironmentSensorsConfig)
    coupling: EnvironmentalCouplingConfig = Field(default_factory=EnvironmentalCouplingConfig)


class AttackScenarioConfig(BaseModel):
    """Configuration for a single attack scenario (V0.3).

    Each scenario defines one attack episode with configurable timing,
    severity, onset/recovery profiles, and attack-specific parameters.
    """

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field("attack_scenario", description="Unique scenario identifier.")
    type: str = Field(
        "none",
        description="Attack type: intercept_resend, channel_loss, source_manipulation, "
        "detector_manipulation, timing, pns, none.",
    )
    severity: float = Field(0.5, ge=0.0, le=1.0, description="Attack severity s in [0, 1].")
    start_pulse: int = Field(0, ge=0, description="First pulse affected by the attack.")
    end_pulse: int | None = Field(None, ge=0, description="Last pulse affected (None = to end).")
    duration_pulses: int | None = Field(None, ge=0, description="Duration in pulses at full severity.")
    onset_pulses: int = Field(1000, ge=0, description="Pulses to ramp up from 0 to full severity.")
    termination_pulses: int = Field(1000, ge=0, description="Pulses to ramp down from full severity to 0.")
    onset_profile: Literal["smoothstep", "linear", "instant", "exponential"] = Field(
        "smoothstep", description="Profile for onset ramp."
    )
    recovery_profile: Literal["smoothstep", "linear", "instant", "exponential"] = Field(
        "smoothstep", description="Profile for recovery ramp."
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Attack-specific parameters."
    )
    description: str = Field("", description="Human-readable description of this scenario.")

    @property
    def parameters(self) -> dict[str, Any]:
        return self.params


class AttackConfig(BaseModel):
    """Attack engine configuration (V0.1: disabled pass-through; V0.3: scenario list)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(False, description="Whether attacks are active (V0.1: no).")
    type: str = Field("none", description="Attack identifier ('none' in V0.1).")
    strength: float = Field(0.0, description="Attack strength parameter (unused in V0.1).")
    start_pulse: int = Field(
        0, ge=0, description="First pulse affected by the attack (unused in V0.1)."
    )
    end_pulse: int | None = Field(None, ge=0, description="Last pulse affected (None = to end).")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Attack-specific parameters."
    )
    # V0.3: scenario list for multiple attack episodes
    scenarios: list[AttackScenarioConfig] = Field(
        default_factory=list,
        description="V0.3 attack scenarios. Each defines one attack episode with timing, "
        "severity, and parameters. When non-empty and enabled=True, these are used "
        "instead of the legacy flat fields.",
    )


# --------------------------------------------------------------------------- #
# Telemetry / simulation / dataset
# --------------------------------------------------------------------------- #


class TelemetryConfig(BaseModel):
    """Windowed telemetry configuration."""

    model_config = ConfigDict(extra="forbid")

    window_size: int = Field(10_000, gt=0, description="Pulses per telemetry window.")
    timing_histogram_bins: int = Field(20, ge=1, description="Bins for timing histograms.")
    include_event_level: bool = Field(True, description="Whether to persist event-level records.")
    interarrival_histogram_bins: int = Field(
        30, ge=1, description="V0.2: bins for per-window inter-arrival histograms."
    )
    interarrival_histogram_scale: Literal["linear", "log"] = Field(
        "linear", description="V0.2: bin spacing of inter-arrival histograms."
    )
    fano_sub_bins: int = Field(
        64, ge=1, description="V0.2: sub-bins used for the window Fano-factor estimate."
    )


class SimulationConfig(BaseModel):
    """Run control: seeds, chunking, checkpointing, output locations."""

    model_config = ConfigDict(extra="forbid")

    master_seed: int = Field(20240101, ge=0, description="Master seed for the whole experiment.")
    chunk_size: int = Field(10_000, gt=0, description="Pulses simulated per chunk.")
    checkpoint_interval: int = Field(10_000, gt=0, description="Pulses between checkpoints.")
    output_root: str = Field("data", description="Root directory for generated datasets.")
    checkpoint_root: str = Field("checkpoints", description="Root directory for checkpoints.")
    scenario_id: str = Field("normal", description="Scenario identifier of this run.")
    log_level: str = Field("INFO", description="Logging level for run output.")


class DatasetConfig(BaseModel):
    """Dataset-level metadata."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["parquet"] = Field(
        "parquet", description="Primary on-disk format (V0.1: Parquet)."
    )
    family: Literal["calibration", "normal", "attacks", "generalization"] = Field(
        "normal", description="Dataset family: calibration | normal | attacks | generalization."
    )
    version: str = Field("0.4.0", description="Dataset schema version.")


# --------------------------------------------------------------------------- #
# Root configuration
# --------------------------------------------------------------------------- #


class QKDConfig(BaseModel):
    """Root configuration object for one simulation run."""

    model_config = ConfigDict(extra="forbid")

    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)
    source: SourceConfig = Field(default_factory=SourceConfig)
    channel: ChannelConfig = Field(default_factory=ChannelConfig)
    background: BackgroundConfig = Field(default_factory=BackgroundConfig)
    receiver: ReceiverConfig = Field(default_factory=ReceiverConfig)
    detectors: DetectorArrayConfig = Field(default_factory=DetectorArrayConfig)
    timing: TimingConfig = Field(default_factory=TimingConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    attack: AttackConfig = Field(default_factory=AttackConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)

    # ------------------------------------------------------------------ #
    def config_hash(self) -> str:
        """Deterministic SHA-256 over the canonical (sorted) config dump.

        ``protocol.n_pulses`` (run length) is deliberately excluded: it is a
        run-control parameter, not a physics parameter, and resuming/extending
        a run with a different pulse count must not invalidate the checkpoint.
        Every physics-affecting parameter is still covered by the hash.
        """
        data = self.model_dump(mode="json")
        protocol = data.get("protocol")
        if isinstance(protocol, dict):
            protocol.pop("n_pulses", None)
        canonical = yaml.safe_dump(data, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def default_config() -> QKDConfig:
    """The reference V0.1 configuration."""
    return QKDConfig()


def load_config(path: str | os.PathLike) -> QKDConfig:
    """Load a QKDConfig from a YAML file."""
    p = pathlib.Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {p} must contain a YAML mapping.")
    return QKDConfig.model_validate(data)


def save_config(config: QKDConfig, path: str | os.PathLike) -> None:
    """Atomically write a config file (tmp + rename)."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(config.to_yaml(), encoding="utf-8")
    os.replace(tmp, p)


def apply_overrides(config: QKDConfig, overrides: dict[str, Any]) -> QKDConfig:
    """Return a new config with dotted-path overrides applied, e.g.
    ``{"protocol.n_pulses": 10000, "simulation.master_seed": 7}``.

    The result is re-validated, so invalid overrides fail loudly.
    """
    data = config.model_dump()
    for key, value in overrides.items():
        parts = key.split(".")
        target: Any = data
        for part in parts[:-1]:
            if part not in target:
                raise KeyError(f"Unknown config path: {key!r}")
            target = target[part]
        if parts[-1] not in target:
            raise KeyError(f"Unknown config path: {key!r}")
        target[parts[-1]] = value
    return QKDConfig.model_validate(data)


# --------------------------------------------------------------------------- #
# Calibration / normal dataset configuration
# --------------------------------------------------------------------------- #


class CalibrationSweeps(BaseModel):
    """Parameter sweeps for the calibration dataset (one parameter varied per run)."""

    model_config = ConfigDict(extra="forbid")

    distance_km: list[float] = Field(default_factory=lambda: [0.0, 5.0, 10.0, 25.0, 50.0])
    attenuation_db_per_km: list[float] = Field(
        default_factory=lambda: [0.10, 0.15, 0.20, 0.25, 0.30]
    )
    mu_signal: list[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5, 0.7])
    mu_decoy: list[float] = Field(default_factory=lambda: [0.05, 0.10, 0.15])
    detector_efficiency: list[float] = Field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20, 0.30])
    dark_count_rate_hz: list[float] = Field(default_factory=lambda: [100.0, 500.0, 1000.0, 5000.0])
    misalignment_deg: list[float] = Field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0, 3.0])
    timing_jitter_s: list[float] = Field(
        default_factory=lambda: [10e-12, 25e-12, 50e-12, 100e-12, 200e-12]
    )

    def items(self) -> list[tuple[str, list[str], list[float]]]:
        """[(label, config_paths, values), ...] in the canonical sweep order.

        Detector-level sweeps vary both detectors symmetrically.
        """
        return [
            ("distance_km", ["channel.distance_km"], self.distance_km),
            (
                "attenuation_db_per_km",
                ["channel.attenuation_db_per_km"],
                self.attenuation_db_per_km,
            ),
            ("mu_signal", ["source.mu_signal"], self.mu_signal),
            ("mu_decoy", ["source.mu_decoy"], self.mu_decoy),
            (
                "detector_efficiency",
                ["detectors.detector_0.efficiency", "detectors.detector_1.efficiency"],
                self.detector_efficiency,
            ),
            (
                "dark_count_rate_hz",
                [
                    "detectors.detector_0.dark_count_rate_hz",
                    "detectors.detector_1.dark_count_rate_hz",
                ],
                self.dark_count_rate_hz,
            ),
            ("misalignment_deg", ["channel.misalignment_deg"], self.misalignment_deg),
            (
                "timing_jitter_s",
                ["detectors.detector_0.timing_jitter_s", "detectors.detector_1.timing_jitter_s"],
                self.timing_jitter_s,
            ),
        ]


class CalibrationConfig(BaseModel):
    """Top-level calibration dataset configuration (configs/calibration.yaml)."""

    model_config = ConfigDict(extra="forbid")

    base: QKDConfig = Field(default_factory=QKDConfig)
    pulses_per_run: int = Field(50_000, gt=0, description="Pulses per calibration run.")
    sweeps: CalibrationSweeps = Field(default_factory=CalibrationSweeps)


class NormalConfig(BaseModel):
    """Top-level normal dataset configuration (configs/normal.yaml)."""

    model_config = ConfigDict(extra="forbid")

    base: QKDConfig = Field(default_factory=QKDConfig)
    runs: int = Field(100, gt=0, description="Number of independent normal runs.")
    pulses_per_run: int | None = Field(
        None, gt=0, description="Optional override of base.protocol.n_pulses."
    )


class AttackDatasetConfig(BaseModel):
    """Top-level attack dataset configuration (configs/v0.3_attack.yaml)."""

    model_config = ConfigDict(extra="forbid")

    base: QKDConfig = Field(default_factory=QKDConfig)
    runs: int = Field(1, gt=0, description="Number of runs per scenario.")
    pulses_per_run: int | None = Field(
        None, gt=0, description="Optional override of base.protocol.n_pulses."
    )
    scenarios: list[AttackScenarioConfig] = Field(
        default_factory=list, description="Attack scenarios to simulate."
    )


class GeneralizationDatasetConfig(BaseModel):
    """Top-level generalization dataset configuration (configs/v0.3_generalization.yaml)."""

    model_config = ConfigDict(extra="forbid")

    base: QKDConfig = Field(default_factory=QKDConfig)
    runs: int = Field(1, gt=0, description="Number of runs per scenario.")
    pulses_per_run: int | None = Field(
        None, gt=0, description="Optional override of base.protocol.n_pulses."
    )
    held_out_types: list[str] = Field(
        default_factory=lambda: ["pns", "timing"],
        description="Attack types reserved for generalization testing.",
    )
    held_out_severities: list[float] = Field(
        default_factory=lambda: [0.15, 0.85],
        description="Severity levels reserved for generalization testing.",
    )
    scenarios: list[AttackScenarioConfig] = Field(
        default_factory=list, description="Explicit generalization scenarios."
    )


def load_calibration_config(path: str | os.PathLike) -> CalibrationConfig:
    p = pathlib.Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {p} must contain a YAML mapping.")
    calib = data.pop("calibration", {})
    base = QKDConfig.model_validate(data)
    return CalibrationConfig.model_validate({"base": base, **calib})


def load_normal_config(path: str | os.PathLike) -> NormalConfig:
    p = pathlib.Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {p} must contain a YAML mapping.")
    norm = data.pop("normal", {})
    base = QKDConfig.model_validate(data)
    return NormalConfig.model_validate({"base": base, **norm})


def load_attack_config(path: str | os.PathLike) -> AttackDatasetConfig:
    p = pathlib.Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {p} must contain a YAML mapping.")
    atk_section = data.pop("attack_dataset", data.pop("attacks", {}))
    base = QKDConfig.model_validate(data)
    if not atk_section.get("scenarios") and base.attack.scenarios:
        atk_section["scenarios"] = base.attack.scenarios
    return AttackDatasetConfig.model_validate({"base": base, **atk_section})


def load_generalization_config(path: str | os.PathLike) -> GeneralizationDatasetConfig:
    p = pathlib.Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {p} must contain a YAML mapping.")
    gen_section = data.pop("generalization_dataset", data.pop("generalization", {}))
    base = QKDConfig.model_validate(data)
    if not gen_section.get("scenarios") and base.attack.scenarios:
        gen_section["scenarios"] = base.attack.scenarios
    return GeneralizationDatasetConfig.model_validate({"base": base, **gen_section})


# --------------------------------------------------------------------------- #
# V0.5 Population and Scenario Models
# --------------------------------------------------------------------------- #


class ParameterDistributionConfig(BaseModel):
    """Specification of a randomized parameter distribution in a population."""

    model_config = ConfigDict(extra="ignore")

    distribution: str = Field("fixed", description="uniform, normal, truncated_normal, log_normal, beta, fixed, categorical")
    mean: float | None = Field(None, description="Mean for Gaussian/truncated/lognormal distributions.")
    std: float | None = Field(None, description="Standard deviation for Gaussian/truncated/lognormal distributions.")
    min_val: float | None = Field(None, description="Minimum clamping bound or uniform lower bound.")
    max_val: float | None = Field(None, description="Maximum clamping bound or uniform upper bound.")
    a: float | None = Field(None, description="Lower bound for uniform.")
    b: float | None = Field(None, description="Upper bound for uniform.")
    alpha: float | None = Field(None, description="Alpha parameter for beta distribution.")
    beta: float | None = Field(None, description="Beta parameter for beta distribution.")
    value: Any = Field(None, description="Fixed scalar value when distribution is 'fixed'.")
    choices: list[Any] | None = Field(None, description="Choices for categorical distribution.")
    weights: list[float] | None = Field(None, description="Weights for categorical distribution.")
    unit: str = Field("dimensionless", description="Physical unit of parameter.")
    description: str = Field("", description="Scientific reasoning and description.")


class PopulationConfig(BaseModel):
    """Parameter distribution definitions organized by component taxonomy."""

    model_config = ConfigDict(extra="ignore")

    source: dict[str, ParameterDistributionConfig] = Field(default_factory=dict)
    channel: dict[str, ParameterDistributionConfig] = Field(default_factory=dict)
    detectors: dict[str, ParameterDistributionConfig] = Field(default_factory=dict)
    timing: dict[str, ParameterDistributionConfig] = Field(default_factory=dict)
    environment: dict[str, ParameterDistributionConfig] = Field(default_factory=dict)
    attack: dict[str, ParameterDistributionConfig] = Field(default_factory=dict)


class SplitConfig(BaseModel):
    """Configuration for a dataset split partition."""

    model_config = ConfigDict(extra="ignore")

    n_scenarios: int = Field(10, ge=1, description="Number of independent scenarios in this split.")
    runs_per_scenario: int = Field(1, ge=1, description="Number of stochastic runs per scenario.")
    pulses_per_run: int = Field(20_000, ge=1, description="Pulses per simulation run.")
    edge_case_fraction: float = Field(0.1, ge=0.0, le=1.0, description="Fraction of scenarios assigned to corner cases.")
    attack_fraction: float = Field(0.5, ge=0.0, le=1.0, description="Fraction of runs/scenarios containing attacks.")
    attack_types: list[str] = Field(
        default_factory=lambda: [
            "intercept_resend",
            "channel_loss",
            "source_manipulation",
            "detector_manipulation",
            "timing",
            "pns",
        ]
    )
    held_out_dimensions: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Held-out parameter bounds for generalization testing."
    )


class Scenario(BaseModel):
    """A complete, frozen experimental condition."""

    model_config = ConfigDict(extra="ignore")

    scenario_id: str
    infrastructure_id: str
    seed: int
    regime: str = "nominal"
    is_edge_case: bool = False
    dataset_split: str = "normal"
    config: QKDConfig
    sampled_parameters: dict[str, Any] = Field(default_factory=dict)
    config_hash: str = ""


class V05DatasetConfig(BaseModel):
    """Top-level configuration for V0.5 population generation and parallel execution."""

    model_config = ConfigDict(extra="ignore")

    dataset_id: str = Field("v05_dataset", description="Identifier for dataset.")
    version: str = Field("0.5.0", description="Dataset schema version.")
    master_seed: int = Field(20260501, ge=0, description="Master root seed.")
    sampling_strategy: str = Field("latin_hypercube", description="'latin_hypercube' or 'random'.")
    worker_count: int = Field(4, ge=1, description="Number of parallel worker processes.")
    batch_size: int = Field(10, ge=1, description="Job batch size for scheduling.")
    shard_size: int = Field(10, ge=1, description="Runs per shard.")
    max_retries: int = Field(2, ge=0, description="Maximum worker retry attempts.")
    output_root: str = Field("data/v0.5", description="Base directory for output datasets.")
    checkpoint_root: str = Field("checkpoints/v0.5", description="Base directory for checkpoints.")
    splits: dict[str, SplitConfig] = Field(default_factory=dict)
    population: PopulationConfig = Field(default_factory=PopulationConfig)
    base: QKDConfig | None = Field(None, description="Base QKD configuration template.")


def load_v05_dataset_config(path: str | os.PathLike) -> V05DatasetConfig:
    p = pathlib.Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {p} must contain a YAML mapping.")

    base_data = data.pop("base", None)
    base_cfg: QKDConfig | None = None
    if base_data is not None:
        if isinstance(base_data, str):
            base_cfg = load_config(p.parent / base_data if not os.path.isabs(base_data) else base_data)
        elif isinstance(base_data, dict):
            base_cfg = QKDConfig.model_validate(base_data)
    else:
        qkd_keys = {"protocol", "source", "channel", "receiver", "detectors"}
        if qkd_keys.intersection(data.keys()):
            base_slice = {k: data.pop(k) for k in qkd_keys if k in data}
            for opt in ("background", "timing", "environment", "attack", "telemetry", "simulation", "dataset"):
                if opt in data:
                    base_slice[opt] = data.pop(opt)
            base_cfg = QKDConfig.model_validate(base_slice)

    data["base"] = base_cfg
    return V05DatasetConfig.model_validate(data)


__all__ = [
    "AfterpulseConfig",
    "AttackConfig",
    "AttackDatasetConfig",
    "AttackScenarioConfig",
    "BackgroundConfig",
    "CalibrationConfig",
    "CalibrationSweeps",
    "ChannelConfig",
    "DatasetConfig",
    "DeadTimeConfig",
    "DetectorArrayConfig",
    "DetectorConfig",
    "DetectorStateConfig",
    "EnvironmentConfig",
    "EnvironmentalCouplingConfig",
    "EnvironmentSensorsConfig",
    "GeneralizationDatasetConfig",
    "NormalConfig",
    "ParameterDistributionConfig",
    "PopulationConfig",
    "ProcessDynamicsConfig",
    "ProtocolConfig",
    "QKDConfig",
    "ReceiverConfig",
    "Scenario",
    "SensorConfig",
    "SimulationConfig",
    "SourceConfig",
    "SplitConfig",
    "TelemetryConfig",
    "TimingConfig",
    "V05DatasetConfig",
    "apply_overrides",
    "default_config",
    "load_attack_config",
    "load_calibration_config",
    "load_config",
    "load_generalization_config",
    "load_normal_config",
    "load_v05_dataset_config",
    "save_config",
]
