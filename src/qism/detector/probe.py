"""Free-running detector probe (analytical validation of detector dynamics).

The BB84 pulse train drives the same :class:`~qism.detector.detector.Detector`
state machine through discrete gates. To validate the *detector itself*
against closed-form results, we additionally feed a detector with a
homogeneous Poisson process of candidate events in continuous time — the
"sorted experiment" of dead-time theory:

    observed rate       r_obs  ~=  lambda / (1 + lambda * tau)     (nonparalyzable)
    inter-arrival mean  E[dt]  ~=  1 / r_obs  =  tau + 1/lambda

with ``tau`` the effective dead time, under the documented ideal conditions
(afterpulsing, dark counts, gating and pulse discretization disabled).

The probe is causal: candidates arrive from an *independent* Poisson process
and are offered to the detector even while it is DEAD (the detector records
the dead-time rejection). No feature is fabricated — observed clicks and
their timestamps emerge from the detector state machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.random import Generator

from qism.detector.detector import Detector


@dataclass
class ProbeRun:
    """Record of one free-running probe experiment."""

    t_start: float = 0.0
    t_end: float = 0.0
    offered_times: np.ndarray = field(default_factory=lambda: np.array([]))
    click_times: np.ndarray = field(default_factory=lambda: np.array([]))
    origins: list[str] = field(default_factory=list)  # per click
    dead_rejected: int = 0
    afterpulse_delays: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def duration_s(self) -> float:
        return float(self.t_end - self.t_start)

    @property
    def observed_rate_hz(self) -> float:
        return (
            float(len(self.click_times)) / self.duration_s if self.duration_s > 0 else float("nan")
        )

    @property
    def n_afterpulse_clicks(self) -> int:
        return sum(1 for o in self.origins if o == "afterpulse")

    @property
    def n_primary_clicks(self) -> int:
        return sum(1 for o in self.origins if o != "afterpulse")


def poisson_times(rng: Generator, rate_hz: float, duration_s: float) -> np.ndarray:
    """Sorted arrival times of a homogeneous Poisson process in [0, T)."""
    rate_hz = float(rate_hz)
    duration_s = float(duration_s)
    if rate_hz <= 0.0 or duration_s <= 0.0:
        return np.array([])
    n = int(rng.poisson(rate_hz * duration_s))
    if n <= 0:
        return np.array([])
    return np.sort(rng.uniform(0.0, duration_s, n))


def run_probe(
    detector: Detector,
    rng: Generator,
    *,
    rate_hz: float,
    duration_s: float,
    rate_window_s: float | None = None,
    t0: float = 0.0,
) -> ProbeRun:
    """Offer Poisson candidates (rate ``rate_hz``) to ``detector``.

    ``rate_window_s`` optionally activates the rate-dependent dead-time model
    exactly as the pulse-train driver does: after every window the driver
    estimates the candidate rate from the window just completed and calls
    ``Detector.set_candidate_rate`` for the next window.
    """
    times = poisson_times(rng, rate_hz, duration_s) + t0
    run = ProbeRun(t_start=t0, t_end=t0 + float(duration_s))
    click_ts: list[float] = []
    origins: list[str] = []
    ap_delays: list[float] = []

    if rate_window_s and rate_window_s > 0:
        window = float(rate_window_s)
        n_windows = max(1, int(np.ceil(float(duration_s) / window)))
    else:
        window = float("inf")
        n_windows = 1

    window_idx = 0
    next_boundary = t0 + window
    cand_in_window = 0

    def _offer(t: float, kind: str, **kwargs) -> None:
        """Offer one candidate; record availability / clicks."""
        nonlocal cand_in_window
        res = detector.offer(t, rng, kind=kind, p_click=1.0, **kwargs)
        cand_in_window += 1
        if not res.available:
            run.dead_rejected += 1
        if res.clicked:
            click_ts.append(res.t)
            origins.append(res.origin)
            if res.origin == "afterpulse" and not np.isnan(res.afterpulse_delay_s):
                ap_delays.append(res.afterpulse_delay_s)

    for t in times:
        while t >= next_boundary and window_idx < n_windows - 1:
            # end the current rate window: lambda = candidates / duration
            lam = cand_in_window / max(window, 1e-12)
            detector.set_candidate_rate(lam)
            cand_in_window = 0
            window_idx += 1
            next_boundary = t0 + (window_idx + 1) * window
        # process afterpulse candidates due at/before this arrival FIRST
        # (strict chronology: they were scheduled by earlier avalanches and
        # their trigger time has already arrived)
        for entry in detector.pop_pending_until(t):
            _offer(
                entry.t,
                "afterpulse",
                parent_pulse_id=entry.parent_pulse_id,
                afterpulse_delay_s=entry.delay_s,
            )
        _offer(t, "probe")
    # drain afterpulses still pending at the end of the observation window
    for entry in detector.pop_pending_until(t0 + float(duration_s)):
        _offer(
            entry.t,
            "afterpulse",
            parent_pulse_id=entry.parent_pulse_id,
            afterpulse_delay_s=entry.delay_s,
        )
    # close the final window
    if window_idx < n_windows and float(duration_s) > 0:
        lam = cand_in_window / max(min(window, float(duration_s)), 1e-12)
        detector.set_candidate_rate(lam)

    run.offered_times = np.asarray(times)
    run.click_times = np.asarray(click_ts)
    run.origins = origins
    run.afterpulse_delays = np.asarray(ap_delays)
    return run
