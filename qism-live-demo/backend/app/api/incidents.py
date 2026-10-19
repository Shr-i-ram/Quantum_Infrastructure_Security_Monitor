"""Incidents management API router."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..services.incident_service import incident_service

router = APIRouter(prefix="/incidents", tags=["Incidents"])


class UpdateStatusRequest(BaseModel):
    status: str  # NEW, ACKNOWLEDGED, INVESTIGATING, RESOLVED


@router.get("")
def list_incidents(
    session_id: str | None = None,
    severity: str | None = None,
    status: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    """Retrieve persistent incidents with optional filters."""
    incidents = incident_service.list_incidents(
        session_id=session_id,
        severity=severity,
        status=status,
        limit=limit,
    )
    return {
        "incidents": incidents,
        "count": len(incidents),
    }


@router.patch("/{incident_id}/status")
def update_status(incident_id: str, req: UpdateStatusRequest) -> dict[str, Any]:
    """Update workflow status of an incident (NEW, ACKNOWLEDGED, INVESTIGATING, RESOLVED)."""
    try:
        updated = incident_service.update_status(incident_id, req.status)
        if not updated:
            raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found.")
        return {
            "status": "SUCCESS",
            "incident_id": incident_id,
            "new_status": req.status.upper(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
