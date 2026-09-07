"""Dataset provenance manifests.

Every run/dataset carries a manifest so that a dataset generated six months
later is still traceable to the exact configuration, simulator version, seed
and environment. Never lose provenance.
"""

from __future__ import annotations

import json
import logging
import pathlib
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any

from qism import __version__ as SIMULATOR_VERSION
from qism.types import DatasetStatus

log = logging.getLogger("qism.manifest")

#: Dataset schema version. V0.2 added detector-dynamics event columns and
#: window features; V0.1 datasets remain readable but are never silently
#: reinterpreted (simulator_version in their manifest stays 0.1.0).
DATASET_VERSION = "0.2.0"


def package_versions() -> dict[str, str]:
    """Versions of the runtime packages that affect simulation output."""
    versions: dict[str, str] = {}
    for name in ("numpy", "scipy", "pandas", "pyarrow", "pydantic"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "not-installed"
    return versions


def python_version() -> str:
    return platform.python_version()


def git_commit() -> str | None:
    """Best-effort current git commit (None when unavailable)."""
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
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def build_manifest(
    *,
    run_id: str,
    scenario: str,
    family: str,
    master_seed: int,
    seed: int,
    number_of_pulses: int,
    window_size: int,
    config_hash: str,
    status: DatasetStatus,
    varied_parameter: str | None = None,
    varied_value: float | None = None,
    attack_type: str | None = None,
    attack_strength: float | None = None,
    attack_start: int | None = None,
    attack_end: int | None = None,
) -> dict[str, Any]:
    """Build the standard run manifest dictionary."""
    return {
        "dataset_version": DATASET_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "scenario": scenario,
        "family": family,
        "seed": seed,
        "master_seed": master_seed,
        "number_of_pulses": int(number_of_pulses),
        "pulses_completed": 0,
        "window_size": int(window_size),
        "configuration_hash": config_hash,
        "git_commit_if_available": git_commit(),
        "python_version": python_version(),
        "package_versions": package_versions(),
        "status": status.value,
        "varied_parameter": varied_parameter,
        "varied_value": varied_value,
        "attack_type": attack_type,
        "attack_strength": attack_strength,
        "attack_start": attack_start,
        "attack_end": attack_end,
        "files": [],
    }


def write_manifest_atomic(manifest: dict[str, Any], path: pathlib.Path) -> None:
    """Atomically write a manifest (tmp + fsync + rename)."""
    from qism.simulation.checkpoint import atomic_write_text

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(manifest, indent=2, default=str))


def update_manifest_field(manifest_path: pathlib.Path, **updates: Any) -> dict[str, Any]:
    """Read a manifest, apply field updates, write it back atomically."""
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest.update(updates)
    write_manifest_atomic(manifest, manifest_path)
    return manifest


def manifest_ok(manifest: dict[str, Any]) -> bool:
    return manifest.get("status") == DatasetStatus.COMPLETED.value


def is_running(manifest: dict[str, Any]) -> bool:
    return manifest.get("status") in (
        DatasetStatus.RUNNING.value,
        DatasetStatus.CREATED.value,
        DatasetStatus.CHECKPOINTED.value,
        DatasetStatus.INTERRUPTED.value,
    )
