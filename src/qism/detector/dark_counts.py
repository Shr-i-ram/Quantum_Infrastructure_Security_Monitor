"""Dark-count model.

For a detection interval Delta_t:

    P_dark = 1 - exp(-lambda_dark * Delta_t)

computed with the numerically stable form ``-expm1(-x)``. For small
probabilities P_dark ≈ lambda_dark * Delta_t. The probability is never a
fixed constant independent of the configured rate — it scales with both the
dark-count rate and the integration interval.
"""

from __future__ import annotations

import numpy as np


def dark_click_probability(rate_hz: float, interval_s: float) -> float:
    """P_dark = 1 - exp(-lambda * Delta_t), stable for small arguments."""
    if rate_hz < 0.0:
        raise ValueError(f"dark count rate must be >= 0, got {rate_hz}")
    if interval_s <= 0.0:
        raise ValueError(f"detection interval must be > 0, got {interval_s}")
    return float(-np.expm1(-rate_hz * interval_s))
