"""API Router registration."""

from fastapi import APIRouter

from .benchmark import router as benchmark_router
from .health import router as health_router
from .incidents import router as incidents_router
from .models import router as models_router
from .scenarios import router as scenarios_router
from .sessions import router as sessions_router
from .telemetry import router as telemetry_router

api_router = APIRouter(prefix="/api")

api_router.include_router(health_router)
api_router.include_router(models_router)
api_router.include_router(scenarios_router)
api_router.include_router(sessions_router)
api_router.include_router(incidents_router)
api_router.include_router(telemetry_router)
api_router.include_router(benchmark_router)

__all__ = ["api_router"]
