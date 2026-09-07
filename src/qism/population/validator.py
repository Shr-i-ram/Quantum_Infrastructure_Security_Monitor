"""Physical validation of generated QKD scenarios.

In accordance with Requirement 20:
Rejects any scenario with unphysical parameters before it can reach the simulation engine.
"""

from __future__ import annotations

import math
from qism.config import QKDConfig


def validate_scenario_config(config: QKDConfig) -> list[str]:
    """Validate a QKDConfig instance for physical consistency and sanity.

    Returns:
        List of validation error strings. If empty, the configuration is valid.
    """
    errors: list[str] = []

    # 1. Protocol sanity
    if config.protocol.n_pulses <= 0:
        errors.append(f"Invalid n_pulses: {config.protocol.n_pulses} (must be > 0)")
    if config.protocol.repetition_rate_hz <= 0:
        errors.append(f"Invalid repetition_rate_hz: {config.protocol.repetition_rate_hz} (must be > 0)")

    # 2. Source sanity
    p_sum = config.source.signal_probability + config.source.decoy_probability + config.source.vacuum_probability
    if abs(p_sum - 1.0) > 1e-4:
        errors.append(f"Source intensity probabilities do not sum to 1.0 (sum = {p_sum})")
    if config.source.mu_signal <= config.source.mu_decoy:
        errors.append(f"Source mu_signal ({config.source.mu_signal}) must be strictly greater than mu_decoy ({config.source.mu_decoy})")
    if config.source.mu_decoy < 0:
        errors.append(f"Source mu_decoy ({config.source.mu_decoy}) cannot be negative")
    if config.source.mu_vacuum != 0.0:
        errors.append(f"Source mu_vacuum must be exactly 0.0, got {config.source.mu_vacuum}")
    if config.source.intensity_jitter_std < 0:
        errors.append(f"Source intensity_jitter_std ({config.source.intensity_jitter_std}) cannot be negative")

    # 3. Channel sanity
    if config.channel.distance_km < 0:
        errors.append(f"Channel distance ({config.channel.distance_km}) cannot be negative")
    if config.channel.attenuation_db_per_km < 0:
        errors.append(f"Channel attenuation ({config.channel.attenuation_db_per_km}) cannot be negative")
    if config.channel.insertion_loss_db < 0:
        errors.append(f"Channel insertion loss ({config.channel.insertion_loss_db}) cannot be negative")
    if not (0.0 <= config.channel.misalignment_deg <= 90.0):
        errors.append(f"Channel misalignment ({config.channel.misalignment_deg}) must be in [0, 90] degrees")

    # 4. Detector sanity
    for d_name, d in [("detector_0", config.detectors.detector_0), ("detector_1", config.detectors.detector_1)]:
        if not (0.0 < d.efficiency <= 1.0):
            errors.append(f"Detector {d_name} efficiency ({d.efficiency}) must be in (0, 1]")
        if d.dark_count_rate_hz < 0:
            errors.append(f"Detector {d_name} dark count rate ({d.dark_count_rate_hz}) cannot be negative")
        if d.timing_jitter_s < 0:
            errors.append(f"Detector {d_name} timing jitter ({d.timing_jitter_s}) cannot be negative")

    if config.detectors.dead_time.enabled:
        if config.detectors.dead_time.tau_dead_s < 0:
            errors.append(f"Dead time tau_dead_s ({config.detectors.dead_time.tau_dead_s}) cannot be negative")

    if config.detectors.afterpulse.enabled:
        if not (0.0 <= config.detectors.afterpulse.probability <= 1.0):
            errors.append(f"Afterpulse probability ({config.detectors.afterpulse.probability}) must be in [0, 1]")
        if config.detectors.afterpulse.tau_afterpulse_s < 0:
            errors.append(f"Afterpulse decay time ({config.detectors.afterpulse.tau_afterpulse_s}) cannot be negative")

    # 5. Environment sanity
    if config.environment.type == "dynamic":
        if config.environment.humidity.min_val is not None and config.environment.humidity.min_val < 0:
            errors.append(f"Humidity min_val ({config.environment.humidity.min_val}) cannot be negative")
        if config.environment.humidity.max_val is not None and config.environment.humidity.max_val > 100:
            errors.append(f"Humidity max_val ({config.environment.humidity.max_val}) cannot exceed 100%")
        if config.environment.vibration.baseline < 0:
            errors.append(f"Vibration baseline ({config.environment.vibration.baseline}) cannot be negative")
        if config.environment.background_light.baseline < 0:
            errors.append(f"Background light baseline ({config.environment.background_light.baseline}) cannot be negative")

    # 6. Attack scenarios sanity
    if config.attack.enabled and config.attack.scenarios:
        for sc in config.attack.scenarios:
            if not (0.0 <= sc.severity <= 1.0):
                errors.append(f"Attack {sc.scenario_id} severity ({sc.severity}) must be in [0, 1]")
            if sc.start_pulse < 0:
                errors.append(f"Attack {sc.scenario_id} start_pulse ({sc.start_pulse}) cannot be negative")
            if sc.end_pulse is not None and sc.end_pulse < sc.start_pulse:
                errors.append(f"Attack {sc.scenario_id} end_pulse ({sc.end_pulse}) must be >= start_pulse ({sc.start_pulse})")
            if sc.onset_pulses < 0:
                errors.append(f"Attack {sc.scenario_id} onset_pulses ({sc.onset_pulses}) cannot be negative")
            if sc.termination_pulses < 0:
                errors.append(f"Attack {sc.scenario_id} termination_pulses ({sc.termination_pulses}) cannot be negative")

    return errors
