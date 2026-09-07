"""Channel attenuation model.

    L_total = alpha * L + L_sys            [dB]
    eta_ch  = 10^(-L_total / 10)           (channel transmittance)

Internal calculations use SI units (metres); the configuration carries
human-readable values (km, dB/km). The attenuation law is implemented
exactly as specified.
"""

from __future__ import annotations

import numpy as np

from qism.config import ChannelConfig


class ChannelLoss:
    """Deterministic channel transmittance from distance and loss budgets."""

    def __init__(self, config: ChannelConfig):
        self.config = config

    @property
    def distance_m(self) -> float:
        return self.config.distance_m

    @property
    def total_loss_db(self) -> float:
        """L_total = alpha * L + L_sys [dB]."""
        return self.config.total_loss_db

    def transmittance(self, delta_loss_db: float = 0.0) -> float:
        """eta_ch = 10^(-(L_total + delta_loss_db)/10)."""
        total = self.total_loss_db + float(delta_loss_db)
        return float(10.0 ** (-total / 10.0))

    def transmittance_at(self, distance_km: float) -> float:
        """Transmittance for an arbitrary distance (used by calibration sweeps)."""
        loss = self.config.attenuation_db_per_km * distance_km + self.config.insertion_loss_db
        return float(10.0 ** (-loss / 10.0))

    def get_parameters(self) -> dict:
        return {
            "distance_km": self.config.distance_km,
            "attenuation_db_per_km": self.config.attenuation_db_per_km,
            "insertion_loss_db": self.config.insertion_loss_db,
            "total_loss_db": self.total_loss_db,
            "transmittance": self.transmittance(),
        }

    @staticmethod
    def transmittance_formula(alpha_db_per_km: float, distance_km: float, l_sys_db: float) -> float:
        """Exact formula: eta_ch = 10^(-(alpha*L + L_sys)/10)."""
        return float(10.0 ** (-(alpha_db_per_km * distance_km + l_sys_db) / 10.0))

    # numpy hook so callers can compute arrays without numpy broadcasting issues
    def transmittance_array(self, distance_km: np.ndarray) -> np.ndarray:
        loss = self.config.attenuation_db_per_km * distance_km + self.config.insertion_loss_db
        return np.power(10.0, -loss / 10.0)
