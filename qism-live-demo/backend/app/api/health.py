"""Health and system diagnostics API router."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import time
from typing import Any

from fastapi import APIRouter
import psutil

from ..config import CHECKPOINTS_DIR, DATASET_ROOT, DB_PATH, PREPROCESSORS_DIR, settings
from ..database import get_db_connection
from ..services.inference_service import ModelRegistry, inference_service
from ..services.replay_service import replay_service
from ..services.simulator_service import simulator_service

router = APIRouter(prefix="/health", tags=["Health"])

START_TIME = time.time()


@router.get("")
def get_system_health() -> dict[str, Any]:
    """Retrieve comprehensive system health, component readiness, and runtime telemetry."""
    # Check SQLite DB
    db_ok = False
    try:
        conn = get_db_connection()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        db_ok = True
    except Exception:
        db_ok = False

    # Check Default Model Status
    default_m = settings.default_model_id
    default_ready = (
        default_m in inference_service.loaded_models
        or (CHECKPOINTS_DIR / "pitqsvdd" / f"{default_m}.pt").exists()
    )

    # Check Preprocessor pipeline
    preprocessor_ready = (PREPROCESSORS_DIR / "fitted_pipeline.joblib").exists()

    # Memory and CPU stats
    mem = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())

    # Active session status
    active_mode = "none"
    active_session_id = None
    if simulator_service.status in ("running", "paused"):
        active_mode = "live_simulation"
        active_session_id = simulator_service.session_id
    elif replay_service.status in ("replaying", "paused"):
        active_mode = "recorded_replay"
        active_session_id = replay_service.session_id

    return {
        "status": "HEALTHY" if (db_ok and default_ready and preprocessor_ready) else "DEGRADED",
        "app_name": settings.app_name,
        "version": settings.version,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "active_session": {
            "mode": active_mode,
            "session_id": active_session_id,
            "simulator_status": simulator_service.status,
            "replay_status": replay_service.status,
            "active_model_id": simulator_service.model_id if active_mode == "live_simulation" else replay_service.model_id,
        },
        "components": {
            "database": {
                "status": "READY" if db_ok else "ERROR",
                "path": str(DB_PATH),
            },
            "default_model": {
                "model_id": default_m,
                "name": "PI-TQSVDD (6 Qubits - Flagship)",
                "status": "READY" if default_ready else "MISSING",
                "quantum_backend": "PennyLane lightning.qubit",
                "circuit_depth": 2,
            },
            "preprocessor_pipeline": {
                "status": "READY" if preprocessor_ready else "MISSING",
                "path": str(PREPROCESSORS_DIR / "fitted_pipeline.joblib"),
            },
            "dataset_repository": {
                "status": "READY" if DATASET_ROOT.exists() else "MISSING",
                "path": str(DATASET_ROOT),
            },
        },
        "runtime": {
            "python_version": platform.python_version(),
            "os": platform.platform(),
            "cpu_percent": psutil.cpu_percent(),
            "process_memory_mb": round(proc.memory_info().rss / (1024 * 1024), 2),
            "system_memory_used_pct": mem.percent,
        },
    }
