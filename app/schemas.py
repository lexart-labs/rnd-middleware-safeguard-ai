"""Pydantic request/response models for the public API.

The ``ChatResponse`` field names (``texto_generado``, etc.) are kept in their
original form because they are part of the established external contract.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CompletionStatus(StrEnum):
    """Possible outcomes of a chat-completion request."""

    ALLOWED = "ALLOWED"
    REJECTED_BY_BUDGET = "REJECTED_BY_BUDGET"
    ALLOWED_PREDICTION_ONLY = "ALLOWED_PREDICTION_ONLY"


class ChatRequest(BaseModel):
    """Inbound chat-completion request."""

    user_id: str = Field(..., description="Identifier of the calling user.")
    prompt: str = Field(..., description="Prompt to estimate and (if allowed) generate.")


class ChatResponse(BaseModel):
    """Outbound chat-completion response.

    ``texto_generado`` is ``None`` whenever generation did not run (budget
    rejection, or upstream fallback).
    """

    status: CompletionStatus
    texto_generado: str | None
    metadata: dict[str, Any]


class RetrainTriggerResponse(BaseModel):
    """Result of a manual retrain trigger."""

    status: str
    message: str


class RetrainStatusResponse(BaseModel):
    """Current state of the background retraining subsystem."""

    is_training: bool
    buffer_size: int
    retrain_threshold: int


class GenerateDatasetResponse(BaseModel):
    """Acknowledgement that dataset generation was scheduled."""

    status: str
    message: str


class HealthResponse(BaseModel):
    """Liveness/readiness probe payload."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    gemini_configured: bool
