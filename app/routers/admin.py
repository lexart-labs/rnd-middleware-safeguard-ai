"""Admin / ops endpoints: retraining control and dataset generation."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks

from app.config import settings
from app.schemas import (
    GenerateDatasetResponse,
    RetrainStatusResponse,
    RetrainTriggerResponse,
)
from app.services.retrain import feedback_buffer
from app.training.generate_dataset import generate_dataset

router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.post("/retrain", response_model=RetrainTriggerResponse)
async def trigger_retrain() -> RetrainTriggerResponse:
    """Flush the feedback buffer and start a retrain immediately."""
    started = feedback_buffer.flush()
    if not started:
        return RetrainTriggerResponse(
            status="skipped", message="Retrain already in progress."
        )
    return RetrainTriggerResponse(
        status="started", message="Retrain triggered manually. Buffer flushed."
    )


@router.get("/retrain/status", response_model=RetrainStatusResponse)
async def retrain_status() -> RetrainStatusResponse:
    """Report whether training is running, buffer size, and the threshold."""
    return RetrainStatusResponse(
        is_training=feedback_buffer.is_training,
        buffer_size=feedback_buffer.size,
        retrain_threshold=settings.retrain_threshold,
    )


@router.post("/generate-dataset", response_model=GenerateDatasetResponse)
async def trigger_generate_dataset(background_tasks: BackgroundTasks) -> GenerateDatasetResponse:
    """Start a background job that regenerates the training dataset via Gemini."""
    background_tasks.add_task(generate_dataset)
    return GenerateDatasetResponse(
        status="started",
        message="Dataset generation running in background. Results appended to dataset.json.",
    )
