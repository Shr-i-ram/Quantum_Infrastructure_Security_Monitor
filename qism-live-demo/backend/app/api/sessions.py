"""Session control router for live simulation and recorded replay lifecycle."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import settings
from ..services.replay_service import replay_service
from ..services.simulator_service import simulator_service

log = logging.getLogger("qism.demo.api.sessions")
router = APIRouter(prefix="/sessions", tags=["Sessions"])


class StartSimRequest(BaseModel):
    scenario_id: str = "live_default"
    model_id: str = settings.default_model_id
    seed: int = 42


class StartReplayRequest(BaseModel):
    scenario_id: str = "scenario_pns_attack"
    model_id: str = settings.default_model_id
    speed: float = 1.0
    live_inference: bool = True


class InjectAttackRequest(BaseModel):
    attack_type: str
    severity: float = 0.8
    duration_windows: int = 15
    parameters: dict[str, Any] | None = None


class SeekRequest(BaseModel):
    window_index: int


class SpeedRequest(BaseModel):
    speed: float


@router.get("/current")
def get_current_session() -> dict[str, Any]:
    """Retrieve detailed state of the active simulation or replay session."""
    if simulator_service.status in ("running", "paused"):
        return {
            "mode": "live_simulation",
            "status": simulator_service.status,
            "session_id": simulator_service.session_id,
            "scenario_id": simulator_service.scenario_id,
            "model_id": simulator_service.model_id,
            "window_index": simulator_service.window_index,
            "active_attack": simulator_service.active_attack,
            "display_badge": "SIMULATION ENVIRONMENT",
        }
    elif replay_service.status in ("replaying", "paused"):
        return {
            "mode": "recorded_replay",
            "status": replay_service.status,
            "session_id": replay_service.session_id,
            "scenario_id": replay_service.scenario_id,
            "model_id": replay_service.model_id,
            "window_index": replay_service.current_window,
            "total_windows": replay_service.total_windows,
            "speed": replay_service.speed,
            "live_inference": replay_service.live_inference,
            "display_badge": "RECORDED SCENARIO REPLAY",
        }
    else:
        return {
            "mode": "none",
            "status": "idle",
            "session_id": None,
            "scenario_id": None,
            "model_id": settings.default_model_id,
            "window_index": 0,
            "active_attack": None,
            "display_badge": "STANDBY",
        }


@router.post("/start")
async def start_live_simulation(req: StartSimRequest) -> dict[str, Any]:
    """Start a new live physical simulation session."""
    # Ensure replay is stopped
    await replay_service.stop()

    session_id = await simulator_service.start(
        scenario_id=req.scenario_id,
        model_id=req.model_id,
        seed=req.seed,
    )
    return {
        "status": "STARTED",
        "mode": "live_simulation",
        "session_id": session_id,
        "model_id": req.model_id,
    }


@router.post("/replay/start")
async def start_recorded_replay(req: StartReplayRequest) -> dict[str, Any]:
    """Start a recorded scenario replay session with live inference."""
    # Ensure simulator is stopped
    await simulator_service.stop()

    try:
        session_id = await replay_service.start(
            scenario_id=req.scenario_id,
            model_id=req.model_id,
            speed=req.speed,
            live_inference=req.live_inference,
        )
        return {
            "status": "REPLAYING",
            "mode": "recorded_replay",
            "session_id": session_id,
            "scenario_id": req.scenario_id,
            "model_id": req.model_id,
            "speed": req.speed,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/pause")
async def pause_session() -> dict[str, Any]:
    """Pause the active live simulation or replay session."""
    if simulator_service.status == "running":
        await simulator_service.pause()
        return {"status": "PAUSED", "mode": "live_simulation"}
    elif replay_service.status == "replaying":
        await replay_service.pause()
        return {"status": "PAUSED", "mode": "recorded_replay"}
    return {"status": "NOOP", "message": "No active session to pause"}


@router.post("/resume")
async def resume_session() -> dict[str, Any]:
    """Resume a paused session."""
    if simulator_service.status == "paused":
        await simulator_service.resume()
        return {"status": "RESUMED", "mode": "live_simulation"}
    elif replay_service.status == "paused":
        await replay_service.resume()
        return {"status": "RESUMED", "mode": "recorded_replay"}
    return {"status": "NOOP", "message": "No paused session to resume"}


@router.post("/stop")
async def stop_session() -> dict[str, Any]:
    """Stop active session."""
    await simulator_service.stop()
    await replay_service.stop()
    return {"status": "STOPPED"}


@router.post("/reset")
async def reset_session() -> dict[str, Any]:
    """Reset simulator and replay to fresh clean state."""
    await simulator_service.reset()
    await replay_service.stop()
    return {"status": "RESET"}


@router.post("/attacks/inject")
def inject_attack(req: InjectAttackRequest) -> dict[str, Any]:
    """Inject a physical attack into the active live simulation."""
    if simulator_service.status not in ("running", "paused"):
        raise HTTPException(status_code=400, detail="Cannot inject attack: Live simulation is not running.")

    try:
        attack = simulator_service.inject_attack(
            attack_type=req.attack_type,
            severity=req.severity,
            duration_windows=req.duration_windows,
            parameters=req.parameters,
        )
        return {
            "status": "ATTACK_INJECTED",
            "attack": attack,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/attacks/clear")
def clear_attacks() -> dict[str, Any]:
    """Clear active attack from the live simulation."""
    if simulator_service.status not in ("running", "paused"):
        raise HTTPException(status_code=400, detail="Live simulation is not running.")
    simulator_service.clear_attack()
    return {"status": "ATTACK_CLEARED"}


@router.post("/replay/seek")
def seek_replay(req: SeekRequest) -> dict[str, Any]:
    """Seek to a window in recorded replay."""
    ok = replay_service.seek(req.window_index)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Invalid window index {req.window_index}")
    return {"status": "SEEK_OK", "window_index": req.window_index}


@router.post("/replay/speed")
def set_replay_speed(req: SpeedRequest) -> dict[str, Any]:
    """Set playback speed for recorded replay."""
    replay_service.set_speed(req.speed)
    return {"status": "SPEED_UPDATED", "speed": replay_service.speed}
