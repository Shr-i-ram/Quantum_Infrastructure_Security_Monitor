"""Shared bootstrap helpers for the QISM CLI scripts.

Adds ``src/`` to ``sys.path`` (so scripts run from a checkout without an
explicit install), verifies the Python 3.10 requirement, and configures
logging to console + a timestamped log file under ``logs/``.
"""

from __future__ import annotations

import logging
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qism import verify_python_version  # noqa: E402


def setup(program_name: str) -> None:
    """Verify Python version and configure logging (console + log file)."""
    verify_python_version()

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{program_name}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logging.getLogger("qism").setLevel(logging.INFO)


def default_config_path() -> pathlib.Path:
    return PROJECT_ROOT / "configs" / "default.yaml"
