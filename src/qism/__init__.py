"""QISM — Quantum Infrastructure Security Monitor.

Physics-informed stochastic telemetry simulator for QKD (decoy-state BB84)
implementation monitoring and anomaly-detection research.

Version 0.2 adds detector realism on top of the V0.1 core: a READY/DEAD
detector state machine with nonparalyzable dead time, hard recovery, a
rate-dependent effective dead-time model, exponential afterpulsing with an
event queue that persists across pulses/chunks/checkpoints, and the
corresponding telemetry (dead-time loss, busy fraction, inter-arrival
statistics, detector-state fields, afterpulse ground truth). Environmental
drift (V0.3) and attacks (V0.4+) remain disabled.

The architecture keeps every V0.1 behaviour: when dead time and afterpulsing
are disabled the simulator uses the exact V0.1 vectorized path, so old
configurations reproduce old output.
"""

from __future__ import annotations

import sys

__version__ = "0.6.0"

# The simulator is developed and validated against Python 3.10 only.
REQUIRED_PYTHON = (3, 10)


def verify_python_version() -> None:
    """Explicitly verify that the active interpreter is Python 3.10.x.

    Raises ``RuntimeError`` otherwise. Every CLI entry point calls this before
    doing any work.
    """
    if sys.version_info[:2] != REQUIRED_PYTHON:
        raise RuntimeError(
            "QISM requires Python 3.10.x, but the active interpreter is "
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} "
            f"({sys.executable}).\n"
            "Create the virtual environment with:\n"
            "  Linux/macOS : python3.10 -m venv .venv\n"
            "  Windows     : py -3.10 -m venv .venv\n"
            "then activate it and re-run this command."
        )


__all__ = ["__version__", "verify_python_version"]
