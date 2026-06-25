"""Tests for the chat-completion guardrail pipeline."""

from __future__ import annotations

import pytest


def test_allowed_request_returns_generation(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"user_id": "user_123", "prompt": "What is the capital of Australia?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ALLOWED"
    assert data["texto_generado"] == "The capital of Australia is Canberra."
    meta = data["metadata"]
    assert meta["input_tokens"] == 10
    assert meta["actual_output"] == 12
    assert meta["prediction_error"] == 12 - meta["predicted_output"]
    assert meta["model_status"] == "OK"


def test_unknown_user_is_rejected(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"user_id": "ghost", "prompt": "hello"},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Unauthorized client"


def test_budget_guardrail_blocks_before_generation(client, monkeypatch):
    # Force the predictor to estimate more than the whole budget.
    from app.routers import chat
    from app.services.predictor import Prediction

    monkeypatch.setattr(
        chat, "predict_output", lambda prompt: Prediction(999_999, "OK")
    )
    resp = client.post(
        "/v1/chat/completions",
        json={"user_id": "user_123", "prompt": "Write an essay."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "REJECTED_BY_BUDGET"
    assert data["texto_generado"] is None
    assert data["metadata"]["estimado"] > data["metadata"]["disponible"]


def test_gemini_unavailable_falls_back_to_prediction_only(client, monkeypatch):
    from app.services import gemini_client as gc

    async def boom(self, prompt: str):
        raise gc.GeminiUnavailable("upstream down")

    monkeypatch.setattr(gc.GeminiClient, "generate", boom)

    resp = client.post(
        "/v1/chat/completions",
        json={"user_id": "user_123", "prompt": "Explain quantum computing."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ALLOWED_PREDICTION_ONLY"
    assert data["texto_generado"] is None
    assert data["metadata"]["warning"] == "GEMINI_UNAVAILABLE"

    # Budget must NOT have been charged on fallback.
    from app.services.budget import budget_store

    assert budget_store.get("user_123").consumed == 0


def test_count_tokens_failure_returns_502(client, monkeypatch):
    from app.services import gemini_client as gc

    async def boom(self, prompt: str) -> int:
        raise gc.GeminiError("count failed")

    monkeypatch.setattr(gc.GeminiClient, "count_input_tokens", boom)

    resp = client.post(
        "/v1/chat/completions",
        json={"user_id": "user_123", "prompt": "hi"},
    )
    assert resp.status_code == 502
