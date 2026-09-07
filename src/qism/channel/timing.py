"""Channel propagation timing.

    t_channel = n_eff * L / c

with n_eff the effective refractive index and c the speed of light. All times
are in seconds. The simulator distinguishes:

    pulse timestamp      — when Alice emits the pulse
    channel delay        — propagation time through the fibre
    expected arrival     — pulse timestamp + channel delay
    measured arrival     — expected arrival + detector jitter (+ attack shifts later)

This separation is essential for future timing attacks.
"""

from __future__ import annotations

import numpy as np

from qism.config import ChannelConfig


class ChannelTiming:
    """Propagation delay through the quantum channel."""

    def __init__(self, config: ChannelConfig):
        self.config = config

    @property
    def channel_delay_s(self) -> float:
        """t_channel = n_eff * L / c [s]."""
        return self.config.n_eff * self.config.distance_m / self.config.speed_of_light_m_per_s

    @staticmethod
    def channel_delay_formula(distance_m: float, n_eff: float, speed_of_light: float) -> float:
        """Exact formula: t = n_eff * L / c."""
        return n_eff * distance_m / speed_of_light

    def expected_arrival(self, pulse_timestamp: np.ndarray) -> np.ndarray:
        """Expected arrival time = pulse timestamp + channel delay."""
        return np.asarray(pulse_timestamp, dtype=np.float64) + self.channel_delay_s

    def get_parameters(self) -> dict:
        return {
            "channel_delay_s": self.channel_delay_s,
            "n_eff": self.config.n_eff,
            "distance_m": self.config.distance_m,
        }
