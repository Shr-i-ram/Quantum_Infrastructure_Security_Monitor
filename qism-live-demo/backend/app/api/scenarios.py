"""Scenario catalogs router for supported attacks and recorded replays."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ..services.replay_service import ReplayService
from ..services.simulator_service import SimulatorService

router = APIRouter(prefix="/scenarios", tags=["Scenarios"])


@router.get("/attacks")
def get_supported_attacks() -> dict[str, Any]:
    """Retrieve all physically supported causal attack mechanisms in the QKD simulator."""
    attacks = SimulatorService.get_supported_attacks()
    return {
        "attacks": attacks,
        "total_supported": len(attacks),
        "target_subsystems": ["Quantum Channel", "Source Laser", "Bob SPAD Detectors", "Timing Sync"],
    }


@router.get("/replays")
def get_recorded_replays() -> dict[str, Any]:
    """Retrieve catalog of pre-recorded research benchmark runs with provenance."""
    catalog = ReplayService.get_catalog()
    return {
        "replays": catalog,
        "total_scenarios": len(catalog),
        "dataset_source": "Phase X Comprehensive QISM Research Dataset",
    }
