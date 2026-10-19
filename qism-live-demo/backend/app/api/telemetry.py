"""Telemetry schema and historical buffer API router."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from ..database import get_recent_telemetry
from ..services.inference_service import inference_service
from ..services.replay_service import replay_service
from ..services.simulator_service import simulator_service

router = APIRouter(prefix="/telemetry", tags=["Telemetry"])


@router.get("/schema")
def get_telemetry_schema() -> dict[str, Any]:
    """Retrieve the 76-feature telemetry dictionary with types, units, and groupings."""
    # Group features logically
    feature_names = inference_service.feature_names
    categories = {
        "qber": [f for f in feature_names if "qber" in f.lower()],
        "rates_and_counts": [f for f in feature_names if any(w in f.lower() for w in ["rate", "count", "sifted", "click"])],
        "gains_and_transmittance": [f for f in feature_names if any(w in f.lower() for w in ["gain", "transmittance", "loss"])],
        "detector_dynamics": [f for f in feature_names if any(w in f.lower() for w in ["detector", "dead_time", "afterpulse", "dark"])],
        "timing_and_polarization": [f for f in feature_names if any(w in f.lower() for w in ["timing", "offset", "delay", "polarization", "misalignment"])],
        "environment_sensors": [f for f in feature_names if any(w in f.lower() for w in ["sensor", "temperature", "voltage", "power", "vibration", "humidity"])],
    }

    # Catch remaining
    categorized = set(sum(categories.values(), []))
    categories["other_observable"] = [f for f in feature_names if f not in categorized]

    return {
        "total_features": len(feature_names),
        "features": feature_names,
        "categories": categories,
        "input_representation": "76-dimensional continuous observable telemetry vector",
        "primary_signals": [
            {"id": "qber", "label": "QBER (Quantum Bit Error Rate)", "unit": "fraction", "nominal": "< 0.03"},
            {"id": "sifted_rate", "label": "Sifted Key Rate", "unit": "events/pulse", "nominal": "0.003 - 0.006"},
            {"id": "gain_signal", "label": "Signal Intensity Gain (Q_mu)", "unit": "fraction", "nominal": "0.005 - 0.010"},
            {"id": "gain_decoy", "label": "Decoy Intensity Gain (Q_nu)", "unit": "fraction", "nominal": "0.001 - 0.003"},
            {"id": "detector_imbalance", "label": "Detector Efficiency Imbalance", "unit": "fraction", "nominal": "-0.05 - +0.05"},
        ],
    }


@router.get("/recent")
def get_recent(
    session_id: str | None = None,
    limit: int = Query(default=100, ge=10, le=500),
) -> dict[str, Any]:
    """Retrieve recent telemetry stream data points for visualization and inspection."""
    active_id = session_id
    if not active_id:
        active_id = simulator_service.session_id or replay_service.session_id

    if not active_id:
        return {"session_id": None, "points": [], "count": 0}

    points = get_recent_telemetry(session_id=active_id, limit=limit)
    return {
        "session_id": active_id,
        "points": points,
        "count": len(points),
    }
