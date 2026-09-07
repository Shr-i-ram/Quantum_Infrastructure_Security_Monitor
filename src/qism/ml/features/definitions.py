"""Feature definitions, observability boundaries, and ablation feature sets (Requirements 26–31, 100–105, 128–130)."""

from __future__ import annotations

# Fields strictly classified as ground truth or unobservable internal states
GROUND_TRUTH_PREFIXES = ("ground_truth_", "yield_", "true_")
GROUND_TRUTH_FIELDS = frozenset(
    [
        "attack_active",
        "attack_type",
        "attack_strength",
        "attack_id",
        "attack_phase",
        "ground_truth_attack_active",
        "ground_truth_attack_type",
        "ground_truth_attack_phase",
        "ground_truth_attack_fraction",
        "ground_truth_mean_severity",
        "ground_truth_attack_id",
        "afterpulse_rate_ground_truth",
        "mean_afterpulse_delay",
        "mean_photon_number",
        "photon_variance",
        "vacuum_fraction",
        "single_photon_fraction",
        "multiphoton_fraction",
        "mean_channel_transmittance",
        "true_temperature_c",
        "true_humidity_pct",
        "true_vibration_g",
        "true_background_photon_rate_hz",
        "true_channel_loss_db",
        "true_polarization_drift_deg",
        "background_detected",
        "alice_bit",
        "alice_basis",
        "bob_basis",
        "click_type",
    ]
)

# Provenance and identifier fields (must not be used as numerical ML features)
METADATA_FIELDS = frozenset(
    [
        "run_id",
        "scenario_id",
        "infrastructure_id",
        "window_id",
        "window_start_pulse",
        "window_end_pulse",
        "pulse_id",
        "timestamp",
    ]
)

# Predefined feature sets for ablation and benchmarking
QKD_ONLY_FEATURES = (
    "qber",
    "qber_z",
    "qber_x",
    "sifted_rate",
    "signal_gain",
    "decoy_gain",
    "vacuum_gain",
    "signal_decoy_residual",
    "decoy_decoy_residual",
    "vacuum_residual",
)

DETECTOR_ONLY_FEATURES = (
    "detector_imbalance",
    "double_click_rate",
    "dark_count_rate",
    "dark_click_fraction",
    "dead_time_loss_rate",
    "candidate_event_rate",
    "accepted_event_rate",
    "effective_dead_time",
    "detector_0_busy_fraction",
    "detector_1_busy_fraction",
    "fano_factor",
    "autocorr_lag1",
    "autocorr_lag2",
)

ENVIRONMENTAL_ONLY_FEATURES = (
    "sensor_temperature_mean",
    "sensor_temperature_std",
    "sensor_temperature_drift",
    "sensor_humidity_mean",
    "sensor_humidity_std",
    "sensor_vibration_mean",
    "sensor_vibration_std",
    "sensor_background_light_mean",
    "sensor_background_light_std",
    "sensor_background_light_drift",
    "clock_offset_mean",
    "clock_offset_std",
    "clock_drift_rate",
)

TIMING_CHANNEL_FEATURES = (
    "mean_arrival_offset",
    "std_arrival_offset",
    "timing_skewness",
    "timing_kurtosis",
    "estimated_channel_loss_db",
    "channel_transmittance_variance",
    "channel_transmittance_drift",
    "source_intensity_mean",
    "source_intensity_variance",
    "source_stability",
    "detection_rate_hz",
    "sifted_key_rate_hz",
)

FULL_FEATURES = (
    QKD_ONLY_FEATURES
    + DETECTOR_ONLY_FEATURES
    + ENVIRONMENTAL_ONLY_FEATURES
    + TIMING_CHANNEL_FEATURES
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "qkd_only": QKD_ONLY_FEATURES,
    "detector_only": DETECTOR_ONLY_FEATURES,
    "environmental_only": ENVIRONMENTAL_ONLY_FEATURES,
    "infrastructure_only": DETECTOR_ONLY_FEATURES + ENVIRONMENTAL_ONLY_FEATURES + TIMING_CHANNEL_FEATURES,
    "minimal": QKD_ONLY_FEATURES,
    "full": FULL_FEATURES,
}


def is_ground_truth_field(field_name: str) -> bool:
    """Return True if field_name is classified as ground-truth or unobservable."""
    if field_name in GROUND_TRUTH_FIELDS:
        return True
    for prefix in GROUND_TRUTH_PREFIXES:
        if field_name.startswith(prefix):
            return True
    return False


def get_feature_set(name: str) -> tuple[str, ...]:
    """Retrieve list of feature names for a given feature set name."""
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{name}'. Available: {list(FEATURE_SETS.keys())}")
    return FEATURE_SETS[name]
