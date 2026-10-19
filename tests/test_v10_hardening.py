"""V1.0 Hardening, Malformed Input Validation, and Edge-Case Recovery Tests."""

import json
import math
import numpy as np
import pandas as pd
import pytest

from qism.monitor.ingestion import (
    DEFAULT_PHYSICAL_BOUNDS,
    FORBIDDEN_GROUND_TRUTH_KEYS,
    IngestionConfig,
    TelemetryIngestor,
    TelemetryValidationError,
)
from qism.monitor.orchestrator import MonitorConfig, QISMMonitor
from qism.monitor.types import SeverityLevel, TelemetryRecord


def test_nan_and_infinity_rejection():
    """Ensure that non-finite values (NaN and Inf) are immediately rejected with TelemetryValidationError."""
    ingestor = TelemetryIngestor()

    # Record with NaN
    nan_rec = TelemetryRecord(
        timestamp_s=1.0,
        sequence_id=1,
        subsystem="receiver",
        features={"qber": float("nan"), "sifted_rate": 800.0},
    )
    with pytest.raises(TelemetryValidationError, match="non-finite"):
        ingestor.validate_and_sanitize(nan_rec)

    # Record with +Infinity
    inf_rec = TelemetryRecord(
        timestamp_s=2.0,
        sequence_id=2,
        subsystem="receiver",
        features={"qber": 0.02, "sifted_rate": float("inf")},
    )
    with pytest.raises(TelemetryValidationError, match="non-finite"):
        ingestor.validate_and_sanitize(inf_rec)


def test_physical_bounds_violation_rejection():
    """Ensure that records violating documented physical limits are rejected."""
    ingestor = TelemetryIngestor()

    # QBER cannot exceed 1.0
    bad_qber = TelemetryRecord(
        timestamp_s=1.0,
        sequence_id=1,
        subsystem="receiver",
        features={"qber": 1.45, "sifted_rate": 800.0},
    )
    with pytest.raises(TelemetryValidationError, match="violates physical bounds"):
        ingestor.validate_and_sanitize(bad_qber)

    # Click rates cannot be negative
    bad_rate = TelemetryRecord(
        timestamp_s=2.0,
        sequence_id=2,
        subsystem="receiver",
        features={"qber": 0.02, "sifted_rate": -150.0},
    )
    with pytest.raises(TelemetryValidationError, match="violates physical bounds"):
        ingestor.validate_and_sanitize(bad_rate)


def test_non_monotonic_timestamp_rejection():
    """Ensure that backwards or out-of-order timestamps within a session are rejected."""
    ingestor = TelemetryIngestor()

    rec1 = TelemetryRecord(
        timestamp_s=10.0,
        sequence_id=1,
        subsystem="receiver",
        features={"qber": 0.02, "sifted_rate": 800.0},
        run_id="run_test",
    )
    ingestor.validate_and_sanitize(rec1)

    # Step back in time
    rec2 = TelemetryRecord(
        timestamp_s=9.5,
        sequence_id=2,
        subsystem="receiver",
        features={"qber": 0.02, "sifted_rate": 800.0},
        run_id="run_test",
    )
    with pytest.raises(TelemetryValidationError, match="Non-monotonic timestamp"):
        ingestor.validate_and_sanitize(rec2)


def test_duplicate_sequence_id_rejection():
    """Ensure that duplicate packet sequence IDs are rejected."""
    ingestor = TelemetryIngestor()

    rec1 = TelemetryRecord(
        timestamp_s=1.0,
        sequence_id=42,
        subsystem="receiver",
        features={"qber": 0.02, "sifted_rate": 800.0},
        run_id="run_test",
    )
    ingestor.validate_and_sanitize(rec1)

    rec2 = TelemetryRecord(
        timestamp_s=2.0,
        sequence_id=42,  # Duplicate sequence ID
        subsystem="receiver",
        features={"qber": 0.02, "sifted_rate": 800.0},
        run_id="run_test",
    )
    with pytest.raises(TelemetryValidationError, match="Duplicate sequence_id"):
        ingestor.validate_and_sanitize(rec2)


def test_ground_truth_stripping_guarantee():
    """Verify that forbidden ground truth fields never enter operational features."""
    ingestor = TelemetryIngestor()

    raw_dict = {
        "timestamp_s": 5.0,
        "sequence_id": 10,
        "subsystem": "composite",
        "qber": 0.018,
        "sifted_rate": 850.0,
        "attack_type": "intercept_resend",
        "attack_active": True,
        "is_attack": 1,
        "true_loss_delta_db": 3.5,
    }
    rec = ingestor.validate_and_sanitize(raw_dict)

    # Operational features must be stripped clean
    assert "qber" in rec.features
    assert "sifted_rate" in rec.features
    for forbidden in FORBIDDEN_GROUND_TRUTH_KEYS:
        assert forbidden not in rec.features, f"Ground-truth leak: {forbidden}"

    # Metadata preserves evaluation keys
    assert rec.metadata.get("attack_type") == "intercept_resend"
    assert rec.metadata.get("attack_active") is True


def test_corrupt_checkpoint_recovery_handling(tmp_path):
    """Verify that attempting to restore from an invalid or corrupted checkpoint raises a clean exception."""
    corrupt_file = tmp_path / "corrupt_ckpt.json"
    corrupt_file.write_text("{ incomplete json data ...")

    cfg = MonitorConfig()
    mon = QISMMonitor(cfg, feature_names=["qber", "sifted_rate"])

    with pytest.raises(json.JSONDecodeError):
        mon.restore(corrupt_file)


def test_monitor_resilience_to_isolated_rejection():
    """Ensure that an ingest error on bad input does not prevent subsequent valid records from processing."""
    cfg = MonitorConfig()
    mon = QISMMonitor(cfg, feature_names=["qber", "sifted_rate"])

    # Step 1: Valid
    rec1 = TelemetryRecord(timestamp_s=1.0, sequence_id=1, subsystem="rx", features={"qber": 0.02, "sifted_rate": 800.0})
    out1 = mon.update(rec1)
    assert out1.telemetry_valid

    # Step 2: Direct ingest validation raises TelemetryValidationError
    rec2 = TelemetryRecord(timestamp_s=2.0, sequence_id=2, subsystem="rx", features={"qber": float("nan"), "sifted_rate": 800.0})
    with pytest.raises(TelemetryValidationError):
        mon.ingestor.validate_and_sanitize(rec2)

    # Step 2b: Calling update() handles the error safely, logs audit event, and marks telemetry_valid=False
    out2 = mon.update(rec2)
    assert not out2.telemetry_valid

    # Step 3: Next valid record processes smoothly
    rec3 = TelemetryRecord(timestamp_s=3.0, sequence_id=3, subsystem="rx", features={"qber": 0.02, "sifted_rate": 800.0})
    out3 = mon.update(rec3)
    assert out3.telemetry_valid
    assert mon._step_counter == 2  # Step counter incremented for valid updates only
