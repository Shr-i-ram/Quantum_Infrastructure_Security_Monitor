"""Unit discipline helpers and fundamental physical constants.

The simulator performs all internal calculations in SI units (seconds,
metres, Hz, radians). Human-readable configuration values (km, ps, degrees)
are converted at the configuration boundary.

Note on the speed of light: it is exposed here as a physical constant and
also mirrored in the channel configuration (``channel.speed_of_light_m_per_s``)
so that no physical parameter is inaccessible through configuration. Both must
agree; the config value defaults to this constant.
"""

from __future__ import annotations

import re

# Fundamental constants (SI)
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0  # c [m/s]

# Multipliers for human-readable unit strings
_MULTIPLIERS: dict[str, float] = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
    "ps": 1e-12,
    "fs": 1e-15,
    "m": 1.0,
    "cm": 1e-2,
    "km": 1e3,
    "hz": 1.0,
    "khz": 1e3,
    "mhz": 1e6,
    "ghz": 1e9,
}

_UNIT_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*([a-zA-Z]+)?\s*$")


def deg_to_rad(degrees: float) -> float:
    """Convert degrees to radians."""
    import numpy as np

    return float(np.deg2rad(degrees))


def parse_unit(value: str | float | int, default_unit: str = "") -> float:
    """Parse a quantity with an optional human-readable unit suffix.

    Examples: ``parse_unit("25 km") == 25000.0``, ``parse_unit("50 ps") == 5e-11``.
    Plain numbers are interpreted in ``default_unit`` (e.g. metres, seconds).
    """
    if isinstance(value, (int, float)):
        return float(value)
    match = _UNIT_RE.match(str(value))
    if not match:
        raise ValueError(f"Cannot parse quantity: {value!r}")
    number, unit = float(match.group(1)), (match.group(2) or default_unit).lower()
    if unit not in _MULTIPLIERS:
        raise ValueError(f"Unknown unit suffix: {unit!r} (value={value!r})")
    return number * _MULTIPLIERS[unit]
