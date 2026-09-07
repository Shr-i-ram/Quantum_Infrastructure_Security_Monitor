"""V0.3 causal attack implementations.

All attacks follow the causal chain:
    attack activation -> parameter intervention -> physical response -> events -> telemetry

Never directly modify telemetry features.
"""

from __future__ import annotations

from typing import Any

from qism.attacks.base import AttackScenario, BaseAttack


class InterceptResendAttack(BaseAttack):
    """Intercept-resend attack: Eve intercepts, measures, and resends.

    Physical mechanism:
        1. Eve intercepts pulse with probability P_intercept = severity * fraction * A(t).
        2. Eve chooses measurement basis B_E (default 50% Z, 50% X).
        3. Eve measures the incoming state (gets Alice's bit if B_E == B_A, else random bit).
        4. Eve prepares and resends a photon with state (B_E, b_E) to Bob.
        5. When Bob measures in basis B_B:
           - If B_B == B_E: photon routes to detector b_E deterministically.
           - If B_B != B_E: photon routes to detector 0 or 1 with 50/50 probability.
        6. For sifted pulses (B_A == B_B):
           - When B_E == B_A (50%): Bob gets Alice's bit -> no error.
           - When B_E != B_A (50%): Bob gets 0 or 1 uniformly -> 50% error.
           -> Net error rate on intercepted sifted pulses = 25% (QBER increases causally).

    Parameters:
        fraction: fraction of pulses targeted for interception (0.0 to 1.0). Default: 1.0.
        eve_basis_prob_z: probability Eve measures in Z basis. Default: 0.5.
    """

    def __init__(self, scenario: AttackScenario):
        super().__init__(scenario)
        self.fraction: float = float(scenario.parameters.get("fraction", 1.0))
        self.eve_basis_prob_z: float = float(scenario.parameters.get("eve_basis_prob_z", 0.5))

    def apply(self, event_state: dict[str, Any], rng: Any) -> dict[str, Any]:
        """Apply intercept-resend attack to one pulse."""
        if self.activation_level <= 0.0:
            return event_state

        intercept_prob = self.activation_level * self.fraction
        if rng.random() >= intercept_prob:
            return event_state

        event_state["intercepted"] = True
        # Eve chooses measurement basis
        eve_basis = 0 if rng.random() < self.eve_basis_prob_z else 1
        alice_basis = int(event_state.get("alice_basis", 0))
        alice_bit = int(event_state.get("alice_bit", 0))

        # Eve measures Alice's photon
        if eve_basis == alice_basis:
            eve_bit = alice_bit
        else:
            eve_bit = 0 if rng.random() < 0.5 else 1

        event_state["eve_basis"] = eve_basis
        event_state["eve_bit"] = eve_bit
        event_state["attack_resend"] = True

        # Eve resends single photon
        n = int(event_state.get("photon_number", 1))
        if n > 0:
            event_state["photon_number"] = 1

        return event_state


class ChannelLossAttack(BaseAttack):
    """Channel manipulation attack: Eve adds extra channel loss.

    Models a bending tap or variable optical attenuator inserted by Eve.

    Physical effect:
        L_attack(t) = severity * delta_L_max * A(t) [dB]
        eta_attack = 10^(-L_attack / 10)
        eta_effective = eta_channel * eta_attack

    Causal chain:
        attack -> extra channel loss -> lower transmittance -> fewer detections
        -> lower gain and sifted key rate

    Parameters:
        delta_L_max_db: maximum extra loss in dB at full severity. Default: 6.0 dB.
    """

    def __init__(self, scenario: AttackScenario):
        super().__init__(scenario)
        self.delta_L_max_db: float = float(scenario.parameters.get("delta_L_max_db", 6.0))

    def apply(self, event_state: dict[str, Any], rng: Any) -> dict[str, Any]:
        """Reduce channel transmittance by the attack's extra loss."""
        if self.activation_level <= 0.0:
            return event_state

        extra_loss_db = self.activation_level * self.delta_L_max_db
        multiplier = float(10.0 ** (-extra_loss_db / 10.0))
        current_eta = float(event_state.get("channel_transmittance", 1.0))
        event_state["channel_transmittance"] = current_eta * multiplier
        event_state["attack_loss_db"] = extra_loss_db
        return event_state


class SourceManipulationAttack(BaseAttack):
    """Source manipulation attack: Eve modifies Alice's source intensity.

    Models side-channel optical injection, laser temperature alteration,
    or drive voltage manipulation altering mean photon number mu.

    Physical effect:
        mu_attack = max(0, mu_0 + severity * delta_mu * direction * A(t))

    Causal chain:
        attack -> modified mu -> Poisson sampling N ~ Poisson(mu)
        -> modified photon number -> modified detection rate and multiphoton fraction

    Parameters:
        delta_mu: intensity perturbation magnitude. Default: 0.3.
        direction: 'increase' or 'decrease'. Default: 'increase'.
    """

    def __init__(self, scenario: AttackScenario):
        super().__init__(scenario)
        self.delta_mu: float = float(scenario.parameters.get("delta_mu", 0.3))
        self.direction: str = str(scenario.parameters.get("direction", "increase"))

    def apply(self, event_state: dict[str, Any], rng: Any) -> dict[str, Any]:
        """Modify source intensity (mu) for this pulse."""
        if self.activation_level <= 0.0:
            return event_state

        current_mu = float(event_state.get("mu", 0.5))
        sign = 1.0 if self.direction == "increase" else -1.0
        new_mu = current_mu + self.activation_level * self.delta_mu * sign
        new_mu = max(0.0, new_mu)
        event_state["mu"] = new_mu
        event_state["source_manipulated"] = True
        return event_state


class DetectorManipulationAttack(BaseAttack):
    """Detector manipulation attack: modifies detector efficiency or dark count rate.

    Models detector efficiency mismatch attacks or partial continuous-wave blinding.

    Modes:
        'mismatch': alters efficiency of target_detector (0 or 1), inducing detector imbalance.
        'efficiency_reduction': symmetrically suppresses detector efficiency.
        'dark_count': increases dark count rate (e.g. thermal or optical leakage).

    Parameters:
        mode: 'mismatch' | 'efficiency_reduction' | 'dark_count'. Default: 'mismatch'.
        delta_eta: relative efficiency change at full severity. Default: 0.6.
        target_detector: 0, 1, or -1 (both). Default: 0.
        delta_dark_hz: dark count rate increase in Hz. Default: 5000.0.
    """

    def __init__(self, scenario: AttackScenario):
        super().__init__(scenario)
        self.mode: str = str(scenario.parameters.get("mode", "mismatch"))
        self.delta_eta: float = float(scenario.parameters.get("delta_eta", 0.6))
        self.target_detector: int = int(scenario.parameters.get("target_detector", 0))
        self.delta_dark_hz: float = float(scenario.parameters.get("delta_dark_hz", 5000.0))

    def apply(self, event_state: dict[str, Any], rng: Any) -> dict[str, Any]:
        """Apply detector efficiency or dark count manipulation."""
        if self.activation_level <= 0.0:
            return event_state

        eta0 = float(event_state.get("detector_0_efficiency", 0.15))
        eta1 = float(event_state.get("detector_1_efficiency", 0.15))
        dark0 = float(event_state.get("detector_0_dark_rate", 1000.0))
        dark1 = float(event_state.get("detector_1_dark_rate", 1000.0))

        if self.mode == "mismatch":
            factor = 1.0 - self.activation_level * self.delta_eta
            factor = max(0.01, min(1.0, factor))
            if self.target_detector == 0:
                event_state["detector_0_efficiency"] = eta0 * factor
            elif self.target_detector == 1:
                event_state["detector_1_efficiency"] = eta1 * factor
            else:
                event_state["detector_0_efficiency"] = eta0 * factor
                event_state["detector_1_efficiency"] = eta1 / factor
        elif self.mode == "efficiency_reduction":
            factor = max(0.01, 1.0 - self.activation_level * self.delta_eta)
            event_state["detector_0_efficiency"] = eta0 * factor
            event_state["detector_1_efficiency"] = eta1 * factor
        elif self.mode == "dark_count":
            add_dark = self.activation_level * self.delta_dark_hz
            if self.target_detector in (0, -1):
                event_state["detector_0_dark_rate"] = dark0 + add_dark
            if self.target_detector in (1, -1):
                event_state["detector_1_dark_rate"] = dark1 + add_dark

        event_state["detector_manipulated"] = True
        return event_state


class TimingAttack(BaseAttack):
    """Timing attack: shifts pulse arrival time relative to detector gate.

    Models a time-shift attack where Eve delays or advances pulses so they
    arrive near the gate edge, or shifts arrival time statistics.

    Physical effect:
        delta_t(t) = severity * delta_t_max * A(t)
        t_arrival = t_arrival + delta_t

    Causal chain:
        attack -> arrival time shift -> photon may fall outside gate
        -> detection rate drops, timing offset changes in telemetry

    Parameters:
        delta_t_max_s: maximum arrival time shift in seconds. Default: 300e-12 (300 ps).
    """

    def __init__(self, scenario: AttackScenario):
        super().__init__(scenario)
        self.delta_t_max_s: float = float(scenario.parameters.get("delta_t_max_s", 300e-12))

    def apply(self, event_state: dict[str, Any], rng: Any) -> dict[str, Any]:
        """Displace pulse arrival time."""
        if self.activation_level <= 0.0:
            return event_state

        shift = self.activation_level * self.delta_t_max_s
        current_offset = float(event_state.get("arrival_time_offset", 0.0))
        event_state["arrival_time_offset"] = current_offset + shift
        event_state["timing_attack_active"] = True
        return event_state


class PNSAttack(BaseAttack):
    """Photon-Number-Splitting (PNS) attack on multi-photon pulses.

    Physical mechanism:
        1. Eve measures photon number non-destructively.
        2. If N >= 2: Eve siphons 1 photon into quantum memory, forwarding N - 1
           photons to Bob through a lower-loss/lossless channel.
        3. If N <= 1: Eve suppresses or attenuates single-photon pulses.

    Causal chain:
        attack -> alters transmitted photon number distribution
        -> signal gain decreases or gain ratios shift -> decoy residual anomalies

    Parameters:
        single_photon_suppression: suppression fraction for N=1 pulses. Default: 0.5.
    """

    def __init__(self, scenario: AttackScenario):
        super().__init__(scenario)
        self.single_photon_suppression: float = float(
            scenario.parameters.get("single_photon_suppression", 0.5)
        )

    def apply(self, event_state: dict[str, Any], rng: Any) -> dict[str, Any]:
        """Apply PNS split/suppress intervention."""
        if self.activation_level <= 0.0:
            return event_state

        n = int(event_state.get("photon_number", 0))
        current_eta = float(event_state.get("channel_transmittance", 1.0))

        if n >= 2:
            # Eve splits off 1 photon
            event_state["photon_number"] = n - 1
            # Eve forwards remainder with enhanced transmission
            boost = 1.0 + self.activation_level * 0.5
            event_state["channel_transmittance"] = min(1.0, current_eta * boost)
            event_state["pns_split"] = True
        elif n == 1:
            # Eve suppresses single photons
            suppression = self.activation_level * self.single_photon_suppression
            event_state["channel_transmittance"] = current_eta * max(0.0, 1.0 - suppression)
            event_state["pns_suppressed"] = True

        return event_state
