"""Continuous-time stochastic processes and drift models for QISM V0.4.

This module implements the mathematical models for temporal environmental dynamics:
1. Exact discrete integration of the Ornstein-Uhlenbeck (OU) mean-reverting process:
       dx_t = theta * (mu - x_t) dt + sigma * dW_t
   Integrated over Delta_t:
       x_{t+Delta_t} = mu + (x_t - mu) * exp(-theta * Delta_t) +
                       sigma * sqrt((1 - exp(-2 * theta * Delta_t)) / (2 * theta)) * eps,
       where eps ~ N(0, 1).
2. Sinusoidal/periodic diurnal cycles:
       x_diurnal(t) = A * sin(2 * pi * t / P + phi)
3. Composite drift:
       x(t) = x_0 + delta_x_slow(t) + delta_x_fast(t)
4. State serialization and restoration for exact checkpoint recovery.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
from numpy.random import Generator


class OUProcess:
    """Continuous-time Ornstein-Uhlenbeck stochastic process with exact discretization.

    Parameters
    ----------
    mean : float
        Equilibrium / reversion mean (mu).
    theta : float
        Mean-reversion rate [1/s] (theta >= 0).
    sigma : float
        Diffusion volatility [unit / sqrt(s)] (sigma >= 0).
    initial_value : float | None
        Initial value at t=0. Defaults to ``mean``.
    min_val : float | None
        Hard lower bound on the state variable (e.g. 0 for rates/intensities).
    max_val : float | None
        Hard upper bound on the state variable (e.g. 1 for probabilities/efficiencies).
    """

    def __init__(
        self,
        mean: float | None = None,
        theta: float = 0.01,
        sigma: float = 0.01,
        initial_value: float | None = None,
        min_val: float | None = None,
        max_val: float | None = None,
        baseline: float | None = None,
    ) -> None:
        if mean is None and baseline is not None:
            mean = baseline
        elif mean is None:
            mean = 0.0
        self.mean = float(mean)
        self.theta = max(0.0, float(theta))
        self.sigma = max(0.0, float(sigma))
        self.min_val = float(min_val) if min_val is not None else None
        self.max_val = float(max_val) if max_val is not None else None

        init = self.mean if initial_value is None else float(initial_value)
        self.value = self._clamp(init)
        self.t = 0.0

    def current_value(self) -> float:
        """Current state value."""
        return self.value

    def _clamp(self, val: float) -> float:
        if self.min_val is not None and val < self.min_val:
            return self.min_val
        if self.max_val is not None and val > self.max_val:
            return self.max_val
        return val

    def step(self, dt: float, rng: Generator) -> float:
        """Advance the OU process by time step ``dt`` [s] using exact Gaussian transition.

        Parameters
        ----------
        dt : float
            Elapsed duration [s]. Must be >= 0.
        rng : Generator
            Numpy PCG64 random generator for the standard normal innovation.

        Returns
        -------
        float
            Updated value x_{t+dt}.
        """
        if dt <= 0.0:
            return self.value

        self.t += dt
        if self.sigma <= 0.0:
            # Deterministic decay towards mean
            if self.theta > 0.0:
                self.value = self.mean + (self.value - self.mean) * math.exp(-self.theta * dt)
            self.value = self._clamp(self.value)
            return self.value

        if self.theta > 1e-12:
            decay = math.exp(-self.theta * dt)
            mean_step = self.mean + (self.value - self.mean) * decay
            # variance = (sigma^2 / (2*theta)) * (1 - exp(-2*theta*dt))
            var = (self.sigma**2 / (2.0 * self.theta)) * max(0.0, 1.0 - math.exp(-2.0 * self.theta * dt))
            std_step = math.sqrt(max(0.0, var))
        else:
            # Random walk limit (theta -> 0): Wiener process
            mean_step = self.value
            std_step = self.sigma * math.sqrt(dt)

        innovation = float(rng.normal(0.0, 1.0))
        self.value = self._clamp(mean_step + std_step * innovation)
        return self.value

    def get_state(self) -> dict[str, Any]:
        """Serialize current state for checkpointing."""
        return {
            "value": self.value,
            "t": self.t,
            "mean": self.mean,
            "theta": self.theta,
            "sigma": self.sigma,
            "min_val": self.min_val,
            "max_val": self.max_val,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore process state from checkpoint."""
        self.value = float(state["value"])
        self.t = float(state["t"])
        self.mean = float(state["mean"])
        self.theta = float(state["theta"])
        self.sigma = float(state["sigma"])
        self.min_val = float(state["min_val"]) if state.get("min_val") is not None else None
        self.max_val = float(state.get("max_val")) if state.get("max_val") is not None else None


class DiurnalCycle:
    """Sinusoidal cyclic variation model: A * sin(2*pi*t / P + phi).

    Parameters
    ----------
    amplitude : float
        Cycle peak amplitude.
    period_s : float
        Period in seconds (e.g. 86400 s for 24 hours, or shorter test periods).
    phase_rad : float
        Initial phase in radians.
    """

    def __init__(self, amplitude: float = 0.0, period_s: float = 86400.0, phase_rad: float = 0.0) -> None:
        self.amplitude = float(amplitude)
        self.period_s = float(period_s) if period_s > 0.0 else 86400.0
        self.phase_rad = float(phase_rad)

    def value(self, t: float) -> float:
        """Evaluate cyclic offset at time t [s]."""
        if abs(self.amplitude) < 1e-15:
            return 0.0
        return self.amplitude * math.sin((2.0 * math.pi * t / self.period_s) + self.phase_rad)

    def value_at(self, t: float) -> float:
        """Alias for value(t)."""
        return self.value(t)

    def get_state(self) -> dict[str, Any]:
        return {
            "amplitude": self.amplitude,
            "period_s": self.period_s,
            "phase_rad": self.phase_rad,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        self.amplitude = float(state["amplitude"])
        self.period_s = float(state["period_s"])
        self.phase_rad = float(state["phase_rad"])


class EnvironmentalDrift:
    """Composite physical drift combining a baseline, slow cyclic/linear drift, and OU fluctuations.

    x(t) = x_0 + delta_x_slow(t) + delta_x_fast(t)

    where:
      - x_0 is the configured equilibrium baseline.
      - delta_x_slow(t) is a sinusoidal diurnal variation.
      - delta_x_fast(t) is an Ornstein-Uhlenbeck process centered at zero.
    """

    def __init__(
        self,
        baseline: float = 0.0,
        theta: float = 0.01,
        sigma: float = 0.01,
        diurnal_amplitude: float = 0.0,
        diurnal_period_s: float = 86400.0,
        diurnal_phase_rad: float = 0.0,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> None:
        self.baseline = float(baseline)
        self.min_val = float(min_val) if min_val is not None else None
        self.max_val = float(max_val) if max_val is not None else None

        self.diurnal = DiurnalCycle(
            amplitude=diurnal_amplitude,
            period_s=diurnal_period_s,
            phase_rad=diurnal_phase_rad,
        )
        self.ou = OUProcess(
            mean=0.0,
            theta=theta,
            sigma=sigma,
            initial_value=0.0,
            min_val=None,
            max_val=None,
        )
        self.t = 0.0

    def step(self, dt: float, rng: Generator) -> float:
        """Advance time by ``dt`` and return the current total physical value."""
        self.t += dt
        self.ou.step(dt, rng)
        return self.current_value()

    def current_value(self) -> float:
        """Current total value at time self.t, clamped to [min_val, max_val]."""
        val = self.baseline + self.diurnal.value(self.t) + self.ou.value
        if self.min_val is not None and val < self.min_val:
            val = self.min_val
        if self.max_val is not None and val > self.max_val:
            val = self.max_val
        return val

    def get_state(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "t": self.t,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "diurnal": self.diurnal.get_state(),
            "ou": self.ou.get_state(),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        self.baseline = float(state["baseline"])
        self.t = float(state["t"])
        self.min_val = float(state["min_val"]) if state.get("min_val") is not None else None
        self.max_val = float(state.get("max_val")) if state.get("max_val") is not None else None
        self.diurnal.restore_state(state["diurnal"])
        self.ou.restore_state(state["ou"])
