"""Health / readiness endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas import HealthResponse
from app.services.gemini_client import gemini_client
from app.services.model_store import model_store

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report liveness plus whether the model and Gemini client are ready."""
    return HealthResponse(
        status="ok",
        model_loaded=model_store.is_ready(),
        gemini_configured=gemini_client.is_configured,
    )
