"""Core enums and shared type definitions for QISM.

These enums are the vocabulary shared by every module. They are designed to be
extended (never broken) by later versions: future click sources, detector
states and field categories simply add members.
"""

from __future__ import annotations

import enum


class Basis(enum.IntEnum):
    """BB84 basis. Z=0, X=1 (stored as integers in event records)."""

    Z = 0
    X = 1


class IntensityClass(enum.Enum):
    """Decoy-state intensity classes."""

    SIGNAL = "signal"
    DECOY = "decoy"
    VACUUM = "vacuum"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class DetectorState(enum.Enum):
    """Detector state machine (V0.2: READY <-> DEAD). Future states:
    RECOVERING, BLINDED, CONTROLLED (V0.5) are added as members."""

    READY = "ready"
    DEAD = "dead"


class ClickSource(enum.Enum):
    """Per-detector origin of an accepted click. V0.2 adds AFTERPULSE.
    Future members: RECOVERY_INDUCED, ATTACK_INDUCED, CONTROL."""

    NONE = "none"
    SIGNAL = "signal"
    DARK = "dark"
    AFTERPULSE = "afterpulse"
    BACKGROUND = "background"
    SIGNAL_AND_DARK = "signal_and_dark"


class ClickType(enum.Enum):
    """Global click outcome for a pulse (both detectors combined).

    V0.2 adds the AFTERPULSE members. V0.4 adds the BACKGROUND members.
    Future members: RECOVERY_INDUCED_*, ATTACK_INDUCED_*, CONTROL_* —
    added as enumeration members, never removing existing ones.
    """

    NONE = "none"
    D0_SIGNAL = "d0_signal"
    D0_DARK = "d0_dark"
    D0_SIGNAL_DARK = "d0_signal_dark"
    D0_BACKGROUND = "d0_background"
    D1_SIGNAL = "d1_signal"
    D1_DARK = "d1_dark"
    D1_SIGNAL_DARK = "d1_signal_dark"
    D1_BACKGROUND = "d1_background"
    DOUBLE_DARK = "double_dark"
    DOUBLE_SIGNAL = "double_signal"
    DOUBLE_BACKGROUND = "double_background"
    DOUBLE_MIXED = "double_mixed"
    DOUBLE_MIXED_BACKGROUND = "double_mixed_background"
    D0_AFTERPULSE = "d0_afterpulse"
    D0_SIGNAL_AFTERPULSE = "d0_signal_afterpulse"
    D0_DARK_AFTERPULSE = "d0_dark_afterpulse"
    D0_SIGNAL_DARK_AFTERPULSE = "d0_signal_dark_afterpulse"
    D1_AFTERPULSE = "d1_afterpulse"
    D1_SIGNAL_AFTERPULSE = "d1_signal_afterpulse"
    D1_DARK_AFTERPULSE = "d1_dark_afterpulse"
    D1_SIGNAL_DARK_AFTERPULSE = "d1_signal_dark_afterpulse"
    DOUBLE_AFTERPULSE = "double_afterpulse"
    DOUBLE_SIGNAL_AFTERPULSE = "double_signal_afterpulse"
    DOUBLE_DARK_AFTERPULSE = "double_dark_afterpulse"
    DOUBLE_MIXED_AFTERPULSE = "double_mixed_afterpulse"


class FieldCategory(enum.Enum):
    """Ground-truth vs. observable classification of every telemetry field.

    ``SIMULATOR_GROUND_TRUTH`` and ``FUTURE_ONLY`` fields must NEVER enter the
    ML feature matrix. The telemetry schema (``qism.telemetry.events``) tags
    every column with one of these categories.
    """

    OBSERVABLE_TELEMETRY = "observable_telemetry"
    DERIVED_TELEMETRY = "derived_telemetry"
    SIMULATOR_GROUND_TRUTH = "simulator_ground_truth"
    FUTURE_ONLY = "future_only"


class DatasetStatus(str, enum.Enum):
    """Lifecycle state recorded in manifests."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    CHECKPOINTED = "CHECKPOINTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class RunStatus(str, enum.Enum):
    """Per-run lifecycle status (stored in the checkpoint)."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    CHECKPOINTED = "CHECKPOINTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class AttackPhase(str, enum.Enum):
    """Ground truth attack phase for temporal evaluation."""

    NORMAL = "normal"
    ONSET = "onset"
    ACTIVE = "active"
    TERMINATION = "termination"
    RECOVERY = "recovery"


class DatasetSplit(str, enum.Enum):
    """Dataset partition type."""

    CALIBRATION = "calibration"
    NORMAL = "normal"
    ATTACK = "attacks"
    GENERALIZATION = "generalization"


class JobStatus(str, enum.Enum):
    """Execution status for simulation jobs in parallel batch scheduling."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class SamplingStrategy(str, enum.Enum):
    """Parameter sampling strategy for scenario generation."""

    RANDOM = "random"
    LATIN_HYPERCUBE = "latin_hypercube"


class DistributionType(str, enum.Enum):
    """Statistical distribution family for population parameter randomization."""

    UNIFORM = "uniform"
    NORMAL = "normal"
    TRUNCATED_NORMAL = "truncated_normal"
    LOG_NORMAL = "log_normal"
    BETA = "beta"
    FIXED = "fixed"
    CATEGORICAL = "categorical"


# Sentinel used for "mathematically undefined" quantities that must stay NaN.
NOT_A_NUMBER = float("nan")
