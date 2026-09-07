"""Attack engine (V0.3: causal attack framework).

V0.3 concrete attacks:
    InterceptResendAttack       — Eve intercepts, measures, resends → QBER increase
    ChannelLossAttack           — Eve adds extra channel loss → gain decrease
    SourceManipulationAttack    — Eve modifies source intensity (mu) → photon distribution change
    DetectorManipulationAttack  — Eve modifies detector efficiency / mismatch / dark counts
    TimingAttack                — Eve shifts pulse arrival time → timing offset & gate losses
    PNSAttack                   — Photon-Number-Splitting attack on multiphoton pulses

All attacks follow the causal chain:
    attack → parameter intervention → physical response → events → telemetry
"""

from qism.attacks.base import (
    AttackEngine,
    AttackParameter,
    AttackScenario,
    AttackState,
    AttackType,
    BaseAttack,
    NullAttack,
    OnsetProfile,
    SystemIntervention,
    evaluate_profile,
)
from qism.attacks.intercept_resend import (
    ChannelLossAttack,
    DetectorManipulationAttack,
    InterceptResendAttack,
    PNSAttack,
    SourceManipulationAttack,
    TimingAttack,
)

__all__ = [
    "AttackEngine",
    "AttackParameter",
    "AttackScenario",
    "AttackState",
    "AttackType",
    "BaseAttack",
    "ChannelLossAttack",
    "DetectorManipulationAttack",
    "InterceptResendAttack",
    "NullAttack",
    "OnsetProfile",
    "PNSAttack",
    "SourceManipulationAttack",
    "SystemIntervention",
    "TimingAttack",
    "evaluate_profile",
]
