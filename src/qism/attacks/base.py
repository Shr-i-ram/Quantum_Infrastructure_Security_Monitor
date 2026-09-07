"""Attack abstraction and lifecycle (V0.3).

V0.3 introduces a causal attack framework where attacks modify the physical
simulation process (photon numbers, detector parameters, channel behavior)
rather than directly manipulating telemetry features.

Attack states:
    INACTIVE -> ARMED -> ACTIVE -> TERMINATING -> COMPLETED

Attack phases (for ground truth and detection latency evaluation):
    NORMAL, ONSET, ACTIVE, TERMINATION, RECOVERY

Attack effects propagate through the simulator:
    attack -> parameter intervention -> physical response -> events -> telemetry
"""

from __future__ import annotations

import enum
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from qism.types import AttackPhase


class AttackState(str, enum.Enum):
    """Lifecycle states for attacks."""
    INACTIVE = "inactive"
    ARMED = "armed"
    ACTIVE = "active"
    TERMINATING = "terminating"
    COMPLETED = "completed"


class AttackType(str, enum.Enum):
    """Attack type identifiers."""
    NONE = "none"
    INTERCEPT_RESEND = "intercept_resend"
    PNS = "pns"  # Photon Number Splitting
    DETECTOR_MANIPULATION = "detector_manipulation"
    CHANNEL_LOSS = "channel_loss"
    TIMING = "timing"
    SOURCE_MANIPULATION = "source_manipulation"


class OnsetProfile(str, enum.Enum):
    """Transition profile shapes for onset and recovery."""
    SMOOTHSTEP = "smoothstep"
    LINEAR = "linear"
    INSTANT = "instant"
    EXPONENTIAL = "exponential"


def evaluate_profile(u: float, profile: OnsetProfile | str) -> float:
    """Evaluate normalized profile factor g(u) for u in [0, 1].

    - instant: jumps immediately to 1 for u > 0
    - linear: g(u) = u
    - smoothstep: g(u) = 3u^2 - 2u^3
    - exponential: g(u) = (1 - e^(-3u)) / (1 - e^(-3))
    """
    u = min(1.0, max(0.0, float(u)))
    p = OnsetProfile(profile) if isinstance(profile, str) else profile
    if p == OnsetProfile.INSTANT:
        return 1.0 if u > 0.0 else 0.0
    elif p == OnsetProfile.LINEAR:
        return u
    elif p == OnsetProfile.SMOOTHSTEP:
        return 3.0 * (u**2) - 2.0 * (u**3)
    elif p == OnsetProfile.EXPONENTIAL:
        norm = 1.0 - math.exp(-3.0)
        return (1.0 - math.exp(-3.0 * u)) / norm
    return u


@dataclass
class SystemIntervention:
    """Represents a causal intervention on a simulator subsystem parameter."""
    target: str  # 'source', 'channel', 'receiver', 'detector', 'timing'
    parameter: str  # e.g. 'mu', 'channel_transmittance', 'loss_db', 'efficiency', 'time_offset'
    operation: str = "add"  # 'add', 'multiply', 'replace', 'clamp'
    value: Any = 0.0
    activation: float = 1.0


@dataclass
class AttackParameter:
    """A configurable attack parameter with distribution."""
    name: str
    value: float
    min_value: float = 0.0
    max_value: float = 1.0
    distribution: str = "uniform"  # uniform, normal, fixed

    def validate(self) -> bool:
        """Validate parameter is within bounds."""
        return self.min_value <= self.value <= self.max_value


@dataclass
class AttackScenario:
    """Complete attack scenario definition.

    Contains all information needed to run an attack episode.
    """
    scenario_id: str
    attack_type: AttackType
    severity: float  # 0.0 to 1.0
    start_pulse: int
    end_pulse: int | None = None
    duration_pulses: int | None = None

    # Attack parameters
    parameters: dict[str, Any] = field(default_factory=dict)

    # Onset/termination profiles
    onset_pulses: int = 1000  # pulses to ramp up
    termination_pulses: int = 1000  # pulses to ramp down
    onset_profile: OnsetProfile | str = OnsetProfile.SMOOTHSTEP
    recovery_profile: OnsetProfile | str = OnsetProfile.SMOOTHSTEP

    # Metadata
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.severity < 0.0 or self.severity > 1.0:
            raise ValueError(f"Severity must be in [0, 1], got {self.severity}")
        if self.start_pulse < 0:
            raise ValueError(f"start_pulse must be >= 0, got {self.start_pulse}")
        if self.duration_pulses is not None and self.duration_pulses < 0:
            raise ValueError(f"duration_pulses must be >= 0, got {self.duration_pulses}")
        if self.end_pulse is None and self.duration_pulses is not None:
            self.end_pulse = self.start_pulse + self.duration_pulses

    @property
    def effective_end_pulse(self) -> int:
        """Get effective end pulse (max int if None)."""
        if self.end_pulse is not None:
            return self.end_pulse
        if self.duration_pulses is not None:
            return self.start_pulse + self.duration_pulses
        return 2**31


class BaseAttack(ABC):
    """Base class for all causal attacks.

    Subclasses implement specific attack mechanisms that modify
    the physical simulation process.

    Attack effects must be causal:
        attack -> parameter change -> physical response -> events -> telemetry

    Never directly modify telemetry features.
    """

    def __init__(self, scenario: AttackScenario):
        self.scenario = scenario
        self.state = AttackState.INACTIVE
        self.phase = AttackPhase.NORMAL
        self.current_pulse = 0
        self.activation_level = 0.0  # 0.0 to 1.0, scaled by onset/recovery

    def update(self, pulse: int) -> None:
        """Deterministic lifecycle and phase update for current pulse."""
        self.current_pulse = pulse
        start = self.scenario.start_pulse
        end = self.scenario.effective_end_pulse
        onset = max(0, self.scenario.onset_pulses)
        recovery = max(0, self.scenario.termination_pulses)

        if pulse < start:
            self.state = AttackState.INACTIVE
            self.phase = AttackPhase.NORMAL
            self.activation_level = 0.0
        elif pulse < end:
            self.state = AttackState.ACTIVE
            elapsed = pulse - start
            if onset > 0 and elapsed < onset:
                self.phase = AttackPhase.ONSET
                u = elapsed / onset
                factor = evaluate_profile(u, self.scenario.onset_profile)
                self.activation_level = self.scenario.severity * factor
            else:
                self.phase = AttackPhase.ACTIVE
                self.activation_level = self.scenario.severity
        else:
            elapsed = pulse - end
            if recovery > 0 and elapsed < recovery:
                self.state = AttackState.TERMINATING
                self.phase = AttackPhase.TERMINATION
                u = elapsed / recovery
                factor = 1.0 - evaluate_profile(u, self.scenario.recovery_profile)
                self.activation_level = self.scenario.severity * factor
            elif elapsed < recovery + max(100, recovery // 2):
                self.state = AttackState.COMPLETED
                self.phase = AttackPhase.RECOVERY
                self.activation_level = 0.0
            else:
                self.state = AttackState.COMPLETED
                self.phase = AttackPhase.NORMAL
                self.activation_level = 0.0

    def activate(self, pulse: int) -> None:
        """Backwards compatible activate call (delegates to update)."""
        self.update(pulse)

    def terminate(self, pulse: int) -> None:
        """Backwards compatible terminate call (delegates to update)."""
        self.update(pulse)

    @abstractmethod
    def apply(self, event_state: dict[str, Any], rng: Any) -> dict[str, Any]:
        """Apply the attack to the physical event state causally.

        Args:
            event_state: Mutable dict of physical event parameters.
            rng: Dedicated random number generator for this attack.

        Returns:
            Modified event state.
        """
        pass

    def get_state(self) -> dict[str, Any]:
        """Serialize attack state for checkpointing."""
        return {
            "state": self.state.value,
            "phase": self.phase.value,
            "current_pulse": self.current_pulse,
            "activation_level": self.activation_level,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore attack state from checkpoint."""
        self.state = AttackState(state["state"])
        self.phase = AttackPhase(state["phase"])
        self.current_pulse = state["current_pulse"]
        self.activation_level = state["activation_level"]


class NullAttack(BaseAttack):
    """Disabled attack: passes state through unchanged."""

    def apply(self, event_state: dict[str, Any], rng: Any) -> dict[str, Any]:
        return event_state


class AttackEngine:
    """Manages multiple concurrent or sequential attacks."""

    def __init__(self, attack_config: Any = None):
        self.attacks: list[BaseAttack] = []
        self.active_attacks: list[BaseAttack] = []
        if attack_config is not None:
            self._build_from_config(attack_config)

    def _build_from_config(self, config: Any) -> None:
        """Build attack instances from an AttackConfig with scenarios."""
        from qism.attacks.intercept_resend import (
            ChannelLossAttack,
            DetectorManipulationAttack,
            InterceptResendAttack,
            PNSAttack,
            SourceManipulationAttack,
            TimingAttack,
        )

        _ATTACK_REGISTRY: dict[str, type[BaseAttack]] = {
            "intercept_resend": InterceptResendAttack,
            "channel_loss": ChannelLossAttack,
            "source_manipulation": SourceManipulationAttack,
            "detector_manipulation": DetectorManipulationAttack,
            "timing": TimingAttack,
            "pns": PNSAttack,
        }

        scenarios = getattr(config, "scenarios", [])
        if scenarios:
            for sc_cfg in scenarios:
                if isinstance(sc_cfg, BaseAttack):
                    self.attacks.append(sc_cfg)
                    continue
                if isinstance(sc_cfg, AttackScenario):
                    atk_type_val = sc_cfg.attack_type.value if hasattr(sc_cfg.attack_type, "value") else str(sc_cfg.attack_type)
                    atk_cls = _ATTACK_REGISTRY.get(atk_type_val)
                    if atk_cls:
                        self.attacks.append(atk_cls(sc_cfg))
                    continue

                sc_type = getattr(sc_cfg, "type", None) or getattr(sc_cfg, "attack_type", None)
                if sc_type == "none" or sc_type is None:
                    continue
                atk_cls = _ATTACK_REGISTRY.get(sc_type)
                if atk_cls is None:
                    raise ValueError(
                        f"Unknown attack type: {sc_type!r}; "
                        f"supported: {list(_ATTACK_REGISTRY.keys())}"
                    )
                onset_prof = getattr(sc_cfg, "onset_profile", "smoothstep")
                rec_prof = getattr(sc_cfg, "recovery_profile", "smoothstep")
                dur = getattr(sc_cfg, "duration_pulses", None)
                scenario = AttackScenario(
                    scenario_id=getattr(sc_cfg, "scenario_id", f"{sc_type}_{sc_cfg.start_pulse}"),
                    attack_type=AttackType(sc_type),
                    severity=sc_cfg.severity,
                    start_pulse=sc_cfg.start_pulse,
                    end_pulse=sc_cfg.end_pulse,
                    duration_pulses=dur,
                    onset_pulses=sc_cfg.onset_pulses,
                    termination_pulses=sc_cfg.termination_pulses,
                    onset_profile=onset_prof,
                    recovery_profile=rec_prof,
                    parameters=dict(getattr(sc_cfg, "params", {}) or getattr(sc_cfg, "parameters", {})),
                    description=getattr(sc_cfg, "description", ""),
                )
                self.attacks.append(atk_cls(scenario))
        else:
            atk_type = getattr(config, "type", "none")
            if atk_type != "none":
                atk_cls = _ATTACK_REGISTRY.get(atk_type)
                if atk_cls is not None:
                    scenario = AttackScenario(
                        scenario_id=f"{atk_type}_{config.start_pulse}",
                        attack_type=AttackType(atk_type),
                        severity=config.strength,
                        start_pulse=config.start_pulse,
                        end_pulse=config.end_pulse,
                        parameters=dict(config.params),
                    )
                    self.attacks.append(atk_cls(scenario))

    def add_attack(self, attack: BaseAttack) -> None:
        """Add an attack to the engine."""
        self.attacks.append(attack)

    def clear(self) -> None:
        """Clear all attacks."""
        self.attacks.clear()
        self.active_attacks.clear()

    def update(self, pulse: int) -> None:
        """Update all attack states and determine active attacks for current pulse."""
        self.active_attacks.clear()
        for attack in self.attacks:
            attack.update(pulse)
            if attack.state in (AttackState.ACTIVE, AttackState.TERMINATING) and attack.activation_level > 0.0:
                self.active_attacks.append(attack)

    def get_dominant_attack(self) -> BaseAttack | None:
        """Return the active attack with the highest activation level."""
        if not self.active_attacks:
            return None
        return max(self.active_attacks, key=lambda a: a.activation_level)

    def get_phase(self) -> AttackPhase:
        """Return the ground truth attack phase for the current pulse."""
        dominant = self.get_dominant_attack()
        if dominant is None:
            return AttackPhase.NORMAL
        return dominant.phase

    def apply(self, event_state: dict[str, Any], rng: Any) -> dict[str, Any]:
        """Apply all active attacks to the event state in deterministic order."""
        for attack in self.active_attacks:
            event_state = attack.apply(event_state, rng)
        return event_state

    def get_state(self) -> dict[str, Any]:
        """Get engine state for checkpointing."""
        return {
            "attacks": [attack.get_state() for attack in self.attacks],
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore engine state from checkpoint."""
        for i, attack_state in enumerate(state.get("attacks", [])):
            if i < len(self.attacks):
                self.attacks[i].restore_state(attack_state)