"""Main FastAPI Application for QISM Live Demonstration Console.

Serves the REST API, WebSocket real-time telemetry stream, and frontend web application.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .api import api_router
from .config import REPO_ROOT, settings
from .database import init_db
from .services.inference_service import inference_service
from .services.replay_service import replay_service
from .services.simulator_service import simulator_service

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("qism.demo")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown management."""
    log.info("=" * 60)
    log.info("Starting QISM Live Demonstration Console Backend")
    log.info("=" * 60)

    # 1. Initialize SQLite Database Schema
    init_db()
    log.info("SQLite database verified and initialized.")

    # 2. Pre-warm default Flagship model (6-qubit PI-TQSVDD)
    try:
        log.info(f"Pre-warming default model: {settings.default_model_id}...")
        inference_service.load_model(settings.default_model_id)
        log.info(f"Successfully loaded default model: {settings.default_model_id}")
    except Exception as e:
        log.warning(f"Could not pre-warm default model on startup: {e}")

    yield

    # Shutdown
    log.info("Shutting down QISM Live Demonstration Console...")
    await simulator_service.stop()
    await replay_service.stop()
    log.info("Clean shutdown complete.")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Live demonstration console for Quantum Infrastructure Security Monitor (QISM).",
    lifespan=lifespan,
)

# CORS Middleware for local frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST API
app.include_router(api_router)


# ---------------------------------------------------------------------------
# WebSocket Real-Time Telemetry Stream
# ---------------------------------------------------------------------------
import math

def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert NaN/Infinity floats and numpy types to JSON-safe primitives."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif hasattr(obj, "item"):
        try:
            return sanitize_for_json(obj.item())
        except Exception:
            return 0.0
    return obj


class ConnectionManager:
    """Manages active WebSocket connections from frontend dashboards."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        log.info(f"WebSocket client connected. Active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            log.info(f"WebSocket client disconnected. Active: {len(self.active_connections)}")

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast sanitized JSON packet to all active clients."""
        clean_msg = sanitize_for_json(message)
        text = json.dumps(clean_msg)
        dead = []
        for ws in self.active_connections:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = ConnectionManager()


# Register subscriber callbacks with both simulator and replay services
async def handle_telemetry_packet(packet: dict[str, Any]) -> None:
    await ws_manager.broadcast(packet)


simulator_service.subscribe(handle_telemetry_packet)
replay_service.subscribe(handle_telemetry_packet)


@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket) -> None:
    """Stream real-time telemetry updates, anomaly scores, and incident alerts."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; can accept client ping or commands
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        log.warning(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Frontend Static Serving
# ---------------------------------------------------------------------------
frontend_dist = REPO_ROOT / "qism-live-demo" / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = frontend_dist / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(frontend_dist / "index.html"))
