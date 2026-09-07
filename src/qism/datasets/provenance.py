"""Dataset provenance tracking and environment capture.

In accordance with Requirements 86-88, 173-175:
Every dataset carries canonical provenance metadata:
- Python version
- QISM simulator version
- Dataset schema version
- Runtime library versions
- Git commit if available
- Deterministic configuration hash
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any

from qism import __version__ as SIMULATOR_VERSION

DATASET_SCHEMA_VERSION = "0.5.0"


def get_git_commit() -> str | None:
    """Best-effort git commit retrieval."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def get_runtime_versions() -> dict[str, str]:
    """Capture versions of packages affecting numerical simulation."""
    pkgs = ("numpy", "scipy", "pandas", "pyarrow", "pydantic", "matplotlib")
    versions: dict[str, str] = {}
    for p in pkgs:
        try:
            mod = __import__(p)
            versions[p] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[p] = "not-installed"
    return versions


def build_provenance_record(
    dataset_id: str,
    master_seed: int,
    config_payload: dict[str, Any],
) -> dict[str, Any]:
    """Create a canonical provenance dictionary for top-level dataset manifests."""
    # Canonical JSON hash
    cfg_json = json.dumps(config_payload, sort_keys=True, separators=(",", ":"), default=str)
    cfg_hash = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()

    return {
        "dataset_id": dataset_id,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "master_seed": master_seed,
        "configuration_hash": cfg_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": get_git_commit(),
        "runtime_dependencies": get_runtime_versions(),
    }
