"""Single-photon detector (SPAD) model with temporal state (V0.2).

V0.1 responsibilities (preserved exactly when dynamics are disabled):
    * detection efficiency:  P(signal click | n photons) = 1 - (1 - eta)^n
    * dark counts:           P_dark = 1 - exp(-lambda_dark * Delta_t)
    * timing jitter and gate acceptance (see qism.detector.timing)

V0.2 adds a continuous-time detector state machine:

    incident optical event -> detector availability -> detector response
    -> avalanche -> dead time -> possible afterpulse -> future events

The detector carries memory: an avalanche at time ``t`` renders the detector
DEAD until ``t + tau_eff`` (nonparalyzable: events arriving during the dead
interval are lost and never extend it). Afterpulses are scheduled as future
candidate events with an exponential delay distribution; a candidate that
arrives while the detector is still DEAD is lost.

The detector itself is gate-agnostic: *whoever* drives it (the pulse-train
receiver, a free-running validation probe, ...) decides which candidates exist
and offers them via :meth:`Detector.offer`. Gate acceptance of signal photons
is receiver-level logic (qism.detector.timing), so future time-shift / after-
gate attacks can change timing without rewriting this class.

Rate-dependent effective dead time:

    tau_eff(lambda) = tau_0 + delta_tau * lambda / (lambda + lambda_c)

``lambda`` is a per-detector candidate-event rate estimated over the previous
rate-estimation window by the driver (never from accepted clicks alone), set
via :meth:`Detector.set_candidate_rate`.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator

from qism.config import AfterpulseConfig, DeadTimeConfig, DetectorConfig
from qism.detector.dark_counts import dark_click_probability
from qism.types import ClickSource, DetectorState


@dataclass
class OfferResult:
    """Outcome of offering one candidate event to a detector.

    ``offered`` is always True. ``available`` is False when the candidate
    arrived while the detector was DEAD (``loss_reason == 'dead_time'``);
    such a candidate is recorded and rejected without touching the state.
    ``clicked`` is True when an avalanche occurred (the detector is then DEAD
    until ``t + dead_duration``).
    """

    t: float
    kind: str
    available: bool = True
    clicked: bool = False
    loss_reason: str = "none"  # 'none' | 'dead_time'
    origin: str = "none"  # 'signal' | 'dark' | 'afterpulse' | 'probe' | 'none'
    dead_duration: float = 0.0
    parent_pulse_id: float = float("nan")
    afterpulse_delay_s: float = float("nan")


@dataclass
class AfterpulseEntry:
    """A scheduled afterpulse candidate (parent = a primary avalanche)."""

    t: float
    parent_pulse_id: int
    delay_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": float(self.t),
            "parent_pulse_id": int(self.parent_pulse_id),
            "delay_s": float(self.delay_s),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AfterpulseEntry:
        return cls(
            t=float(d["t"]), parent_pulse_id=int(d["parent_pulse_id"]), delay_s=float(d["delay_s"])
        )


class Detector:
    """One detector of Bob's two-detector array (continuous-time state machine).

    Each detector owns its own state, dead-time clock, click counters and
    afterpulse queue — a click in D0 never puts D1 into dead time.
    """

    def __init__(
        self,
        config: DetectorConfig,
        index: int,
        *,
        gate_width_s: float = 1e-9,
        detection_interval_s: float = 1e-9,
        dead_time: DeadTimeConfig | None = None,
        afterpulse: AfterpulseConfig | None = None,
        # --- V0.1-compatible flat overrides (used by old call sites) ---
        dead_time_enabled: bool | None = None,
        dead_time_s: float | None = None,
        afterpulse_enabled: bool | None = None,
        p_afterpulse: float | None = None,
        tau_afterpulse_s: float | None = None,
    ):
        self.config = config
        self.index = index
        self.gate_width_s = float(gate_width_s)
        self.detection_interval_s = float(detection_interval_s)

        # --- dead-time model ---
        self.dead_time_cfg = dead_time or DeadTimeConfig(enabled=False)
        if dead_time_enabled is not None:
            self.dead_time_cfg.enabled = bool(dead_time_enabled)
        if dead_time_s is not None:
            # legacy flat override means fixed-model semantics
            self.dead_time_cfg.enabled = True if dead_time_enabled else self.dead_time_cfg.enabled
            self.dead_time_cfg.model = "fixed"
            self.dead_time_cfg.tau_dead_s = float(dead_time_s)

        # --- afterpulse model ---
        self.afterpulse_cfg = afterpulse or AfterpulseConfig(enabled=False)
        if afterpulse_enabled is not None:
            self.afterpulse_cfg.enabled = bool(afterpulse_enabled)
        if p_afterpulse is not None:
            self.afterpulse_cfg.probability = float(p_afterpulse)
        if tau_afterpulse_s is not None:
            self.afterpulse_cfg.tau_afterpulse_s = float(tau_afterpulse_s)

        # --- state machine ---
        self.state = DetectorState.READY
        self.last_click_time: float = -np.inf
        self.dead_until: float = -np.inf
        self._n_clicks = 0
        self._n_dead_rejected = 0
        self._n_afterpulse_scheduled = 0

        # pending afterpulse candidates, kept sorted by time (priority queue)
        self._pending_afterpulses: list[AfterpulseEntry] = []

        # busy intervals (start, end); nonparalyzable dead time makes them
        # disjoint, so the union time is simply the sum of the durations.
        self._busy_intervals: list[tuple[float, float]] = []
        # running tau_eff (set by the rate model / driver)
        self._tau_eff_s: float = self._initial_tau_eff()

    # ------------------------------------------------------------------ #
    # Parameters
    # ------------------------------------------------------------------ #
    @property
    def efficiency(self) -> float:
        return self.config.efficiency

    def efficiency_at_temperature(self, temp_c: float, delta_eff: float = 0.0) -> float:
        """Detector efficiency at temperature T [C] with optional perturbation:
        eta_d(T) = clamp(eta_d,0 + k_eta * (T - T0) + delta_eff, 0, 1)."""
        eta0 = self.config.efficiency
        k_eta = getattr(self.config, "temp_coeff_efficiency", 0.0)
        t_ref = getattr(self.config, "t_ref_c", 20.0)
        eff = eta0 + k_eta * (temp_c - t_ref) + delta_eff
        return max(0.0, min(1.0, float(eff)))

    @property
    def dark_count_rate_hz(self) -> float:
        return self.config.dark_count_rate_hz

    def dark_rate_at_temperature(self, temp_c: float, delta_rate: float = 0.0) -> float:
        """Dark count rate [Hz] at temperature T [C] with exponential scaling:
        R_d(T) = max(0, R_d,0 * exp(k_T * (T - T0)) + delta_rate)."""
        import math
        r0 = self.config.dark_count_rate_hz
        k_t = getattr(self.config, "temp_coeff_dark_count", 0.0)
        t_ref = getattr(self.config, "t_ref_c", 20.0)
        if k_t != 0.0:
            scale = math.exp(max(-20.0, min(20.0, k_t * (temp_c - t_ref))))
            rate = r0 * scale + delta_rate
        else:
            rate = r0 + delta_rate
        return max(0.0, float(rate))

    @property
    def temp_coeff_dark_count(self) -> float:
        return getattr(self.config, "temp_coeff_dark_count", 0.0)

    @property
    def temp_coeff_efficiency(self) -> float:
        return getattr(self.config, "temp_coeff_efficiency", 0.0)

    @property
    def t_ref_c(self) -> float:
        return getattr(self.config, "t_ref_c", 20.0)

    @property
    def timing_jitter_s(self) -> float:
        return self.config.timing_jitter_s

    @property
    def dead_time_enabled(self) -> bool:
        return self.dead_time_cfg.enabled

    @property
    def afterpulse_enabled(self) -> bool:
        return self.afterpulse_cfg.enabled

    @property
    def effective_dead_time_s(self) -> float:
        """Current effective dead time tau_eff [s] used for new avalanches."""
        return self._tau_eff_s

    # ------------------------------------------------------------------ #
    # Core physics
    # ------------------------------------------------------------------ #
    def signal_click_probability(self, n_photons: np.ndarray) -> np.ndarray:
        """P(signal click | n photons) = 1 - (1 - eta)^n, stable for n=0."""
        n = np.asarray(n_photons, dtype=np.float64)
        return -np.expm1(n * np.log1p(-self.efficiency))

    def dark_click_probability(self) -> float:
        """P_dark = 1 - exp(-lambda_dark * Delta_t) (per detection interval)."""
        return dark_click_probability(self.dark_count_rate_hz, self.detection_interval_s)

    def _initial_tau_eff(self) -> float:
        if not self.dead_time_cfg.enabled:
            return 0.0
        if self.dead_time_cfg.model == "rate_dependent":
            return float(self.dead_time_cfg.tau_0_s)
        return float(self.dead_time_cfg.tau_dead_s)

    def tau_eff_at_rate(self, lambda_hz: float) -> float:
        """tau_eff(lambda) for the configured model."""
        cfg = self.dead_time_cfg
        if cfg.model == "rate_dependent":
            lam = max(0.0, float(lambda_hz))
            return cfg.tau_0_s + cfg.delta_tau_s * lam / (lam + cfg.lambda_c_hz)
        return cfg.tau_dead_s

    def set_candidate_rate(self, lambda_hz: float) -> None:
        """Update tau_eff from a candidate-event rate (rate-dependent model).

        The driver estimates lambda over one rate-estimation window and calls
        this at the window boundary; all avalanches in the following window
        use the resulting tau_eff. (No-op for the fixed model.)
        """
        if not self.dead_time_cfg.enabled:
            return
        if self.dead_time_cfg.model == "rate_dependent":
            self._tau_eff_s = self.tau_eff_at_rate(lambda_hz)

    # ------------------------------------------------------------------ #
    # State machine
    # ------------------------------------------------------------------ #
    def update_state(self, t: float) -> DetectorState:
        """Fold time forward to ``t``: recover when t >= dead_until."""
        if self.state is DetectorState.DEAD and t >= self.dead_until:
            self.state = DetectorState.READY
        return self.state

    def is_available(self, t: float) -> bool:
        """Whether the detector can avalanche at time t (s)."""
        if not self.dead_time_enabled:
            return True
        self.update_state(t)
        return t >= self.dead_until

    def state_at(self, t: float) -> DetectorState:
        self.update_state(t)
        return self.state

    def dead_time_remaining(self, t: float) -> float:
        """max(0, dead_until - t); 0 for a READY detector."""
        if not self.dead_time_enabled or self.state is DetectorState.READY:
            return 0.0
        return max(0.0, self.dead_until - t)

    def time_since_last_click(self, t: float) -> float:
        if np.isneginf(self.last_click_time):
            return float("nan")
        return float(t - self.last_click_time)

    # ------------------------------------------------------------------ #
    # Candidate processing (the causal event-level primitive)
    # ------------------------------------------------------------------ #
    def offer(
        self,
        t: float,
        rng: Generator,
        *,
        kind: str,
        p_click: float,
        parent_pulse_id: int | None = None,
        afterpulse_delay_s: float | None = None,
    ) -> OfferResult:
        """Offer one candidate event at time ``t``.

        ``kind`` is one of ``'signal' | 'dark' | 'afterpulse' | 'probe'`` and
        ``p_click`` the probability that the candidate produces an avalanche
        when the detector is READY (1.0 for dark/afterpulse/probe candidates,
        whose arrival already encodes the detection statistics; for signal
        candidates ``1 - (1 - eta)^n``).

        Causal order (spec §10): availability is checked first; a DEAD
        detector rejects the candidate (``loss_reason='dead_time'``) and never
        extends its dead period. An accepted avalanche sets the detector DEAD
        until ``t + tau_eff`` and may schedule a primary-only afterpulse.
        """
        result = OfferResult(t=t, kind=kind)
        if not self.dead_time_enabled or self.is_available(t):
            result.available = True
            if kind == "afterpulse" or p_click >= 1.0:
                clicked = True
            else:
                clicked = bool(rng.random() < p_click)
            if clicked:
                tau = self.effective_dead_time_s
                self._register_avalanche(
                    t,
                    rng,
                    origin=kind,
                    tau=tau,
                    parent_pulse_id=parent_pulse_id,
                    ap_delay=afterpulse_delay_s,
                )
                result.clicked = True
                result.origin = kind
                result.dead_duration = tau
                if parent_pulse_id is not None:
                    result.parent_pulse_id = float(parent_pulse_id)
                if afterpulse_delay_s is not None:
                    result.afterpulse_delay_s = float(afterpulse_delay_s)
        else:
            result.available = False
            result.loss_reason = "dead_time"
            self._n_dead_rejected += 1
        return result

    def _register_avalanche(
        self,
        t: float,
        rng: Generator | None,
        *,
        origin: str,
        tau: float,
        parent_pulse_id: int | None = None,
        ap_delay: float | None = None,
    ) -> None:
        """Accept an avalanche at ``t``: dead interval + counters + afterpulse.

        With dead time disabled the detector remains READY (zero-length dead
        interval): the state machine is off and behaviour matches V0.1.
        """
        self.last_click_time = float(t)
        self._n_clicks += 1
        if self.dead_time_enabled and float(tau) > 0.0:
            self.state = DetectorState.DEAD
            self.dead_until = float(t) + float(tau)
            self._busy_intervals.append((float(t), float(self.dead_until)))
        else:
            # no dead interval: detector recovers instantly
            self.state = DetectorState.READY
            self.dead_until = float(t)
        if self.afterpulse_enabled and rng is not None and origin != "afterpulse":
            if (
                self.afterpulse_cfg.probability > 0.0
                and rng.random() < self.afterpulse_cfg.probability
            ):
                delay = float(rng.exponential(self.afterpulse_cfg.tau_afterpulse_s))
                self._schedule_afterpulse(t + delay, parent_pulse_id, delay)
                self._n_afterpulse_scheduled += 1

    def _schedule_afterpulse(
        self, t_ap: float, parent_pulse_id: int | None = None, delay_s: float | None = None
    ) -> None:
        entry = AfterpulseEntry(
            t=t_ap,
            parent_pulse_id=int(parent_pulse_id) if parent_pulse_id is not None else -1,
            delay_s=float(delay_s) if delay_s is not None else float("nan"),
        )
        # entries are consumed in strict chronological order
        keys = [e.t for e in self._pending_afterpulses]
        pos = bisect.bisect_left(keys, t_ap)
        self._pending_afterpulses.insert(pos, entry)

    def pop_pending_until(self, t_end: float) -> list[AfterpulseEntry]:
        """Remove and return (in chronological order) every pending afterpulse
        candidate due at or before ``t_end``."""
        out: list[AfterpulseEntry] = []
        keys = [e.t for e in self._pending_afterpulses]
        n = bisect.bisect_right(keys, t_end)
        if n:
            out = self._pending_afterpulses[:n]
            del self._pending_afterpulses[:n]
        return out

    def pending_afterpulse_count(self) -> int:
        return len(self._pending_afterpulses)

    def pending_front(self) -> AfterpulseEntry | None:
        """Earliest pending afterpulse candidate (or None)."""
        return self._pending_afterpulses[0] if self._pending_afterpulses else None

    def pop_next_pending(self) -> AfterpulseEntry | None:
        """Remove and return the earliest pending afterpulse candidate."""
        if not self._pending_afterpulses:
            return None
        return self._pending_afterpulses.pop(0)

    def register_click(self, t: float, rng: Generator | None = None, origin: str = "probe") -> None:
        """Record a click at time t (legacy convenience API).

        Registers an avalanche with the current effective dead time and
        (optionally) afterpulse scheduling. Kept for V0.1 compatibility.
        """
        self._register_avalanche(float(t), rng, origin=origin, tau=self.effective_dead_time_s)

    # ------------------------------------------------------------------ #
    # Statistics / introspection
    # ------------------------------------------------------------------ #
    def get_state(self) -> DetectorState:
        return self.state

    @property
    def total_clicks(self) -> int:
        return self._n_clicks

    @property
    def total_dead_rejected(self) -> int:
        return self._n_dead_rejected

    @property
    def total_afterpulses_scheduled(self) -> int:
        return self._n_afterpulse_scheduled

    def busy_time(self, t0: float, t1: float) -> float:
        """Exact busy time (interval union) inside ``[t0, t1)``.

        Nonparalyzable dead intervals never overlap, so this is the sum of
        clipped interval lengths; the method is written to remain correct for
        future models with overlapping intervals.
        """
        total = 0.0
        for start, end in self._busy_intervals:
            lo = max(start, t0)
            hi = min(end, t1)
            if hi > lo:
                total += hi - lo
        return total

    def prune_history_before(self, t0: float) -> None:
        """Drop busy intervals that end at or before ``t0`` (memory bound)."""
        keep = [iv for iv in self._busy_intervals if iv[1] > t0]
        del self._busy_intervals[len(keep) :]
        self._busy_intervals[:] = keep

    def reset_runtime(self) -> None:
        """Reset to the initial state (used when a fresh run starts)."""
        self.state = DetectorState.READY
        self.last_click_time = -np.inf
        self.dead_until = -np.inf
        self._n_clicks = 0
        self._n_dead_rejected = 0
        self._n_afterpulse_scheduled = 0
        self._pending_afterpulses.clear()
        self._busy_intervals.clear()
        self._tau_eff_s = self._initial_tau_eff()

    # ------------------------------------------------------------------ #
    # Checkpoint serialization
    # ------------------------------------------------------------------ #
    def runtime_state(self) -> dict[str, Any]:
        """Serialize the full temporal state (checkpointing / resume)."""
        return {
            "index": self.index,
            "state": self.state.value,
            "last_click_time": float(self.last_click_time),
            "dead_until": float(self.dead_until),
            "n_clicks": self._n_clicks,
            "n_dead_rejected": self._n_dead_rejected,
            "n_afterpulse_scheduled": self._n_afterpulse_scheduled,
            "tau_eff_s": float(self._tau_eff_s),
            "pending_afterpulses": [e.to_dict() for e in self._pending_afterpulses],
            "busy_intervals": [[float(a), float(b)] for a, b in self._busy_intervals],
        }

    def restore_runtime_state(self, state: dict[str, Any]) -> None:
        self.index = int(state.get("index", self.index))
        self.state = DetectorState(state.get("state", "ready"))
        self.last_click_time = float(state.get("last_click_time", -np.inf))
        self.dead_until = float(state.get("dead_until", -np.inf))
        self._n_clicks = int(state.get("n_clicks", 0))
        self._n_dead_rejected = int(state.get("n_dead_rejected", 0))
        self._n_afterpulse_scheduled = int(state.get("n_afterpulse_scheduled", 0))
        self._tau_eff_s = float(state.get("tau_eff_s", self._initial_tau_eff()))
        self._pending_afterpulses = [
            AfterpulseEntry.from_dict(d) for d in state.get("pending_afterpulses", [])
        ]
        self._pending_afterpulses.sort(key=lambda e: e.t)
        self._busy_intervals = [(float(a), float(b)) for a, b in state.get("busy_intervals", [])]

    # ------------------------------------------------------------------ #
    # Parameters / V0.1-compatible API
    # ------------------------------------------------------------------ #
    def get_parameters(self) -> dict:
        return {
            "label": self.config.label,
            "efficiency": self.efficiency,
            "dark_count_rate_hz": self.dark_count_rate_hz,
            "timing_jitter_s": self.timing_jitter_s,
            "dead_time_enabled": self.dead_time_enabled,
            "dead_time_model": self.dead_time_cfg.model,
            "dead_time_s": self.dead_time_cfg.tau_dead_s,
            "tau_0_s": self.dead_time_cfg.tau_0_s,
            "delta_tau_s": self.dead_time_cfg.delta_tau_s,
            "lambda_c_hz": self.dead_time_cfg.lambda_c_hz,
            "effective_dead_time_s": self.effective_dead_time_s,
            "afterpulse_enabled": self.afterpulse_enabled,
            "p_afterpulse": self.afterpulse_cfg.probability,
            "tau_afterpulse_s": self.afterpulse_cfg.tau_afterpulse_s,
            "gate_width_s": self.gate_width_s,
            "detection_interval_s": self.detection_interval_s,
        }

    # ------------------------------------------------------------------ #
    # V0.1 per-pulse detect() convenience API
    # ------------------------------------------------------------------ #
    def detect(
        self,
        n_photons: int,
        rng: Generator,
        t: float,
    ) -> tuple[bool, ClickSource]:
        """Detect a single pulse arriving at time t with n_photons photons
        (V0.1 semantics: signal and dark contributions sampled independently,
        at most one click; dead-time availability is respected when enabled).
        """
        if not self.is_available(t):
            return False, ClickSource.NONE

        p_sig = self.signal_click_probability(np.asarray([n_photons]))[0]
        sig_clicked = bool(rng.random() < p_sig)
        if sig_clicked:
            offset = float(rng.normal(0.0, self.timing_jitter_s))
            half = self.gate_width_s / 2.0
            sig_clicked = abs(offset) <= half
        dark_clicked = bool(rng.random() < self.dark_click_probability())

        if sig_clicked and dark_clicked:
            source = ClickSource.SIGNAL_AND_DARK
        elif sig_clicked:
            source = ClickSource.SIGNAL
        elif dark_clicked:
            source = ClickSource.DARK
        else:
            source = ClickSource.NONE

        if sig_clicked or dark_clicked:
            self.register_click(t, rng=rng, origin="signal" if sig_clicked else "dark")
        return (sig_clicked or dark_clicked), source
