"""Shared pytest fixtures.

Gemini is fully mocked — tests never hit the network or require an API key.
The model store is loaded from the committed artifacts when available; tests
that exercise the fallback path force it into a not-ready state explicitly.
"""

from __future__ import annotations

import warnings

import pytest
from fastapi.testclient import TestClient

warnings.filterwarnings("ignore")


@pytest.fixture
def client(monkeypatch):
    """A TestClient with Gemini calls mocked and the model store loaded."""
    from app.main import app
    from app.services import gemini_client as gc
    from app.services.budget import budget_store
    from app.services.model_store import model_store

    # Load committed artifacts (real prediction path).
    model_store.load()

    # Deterministic mocked Gemini: input=10 tokens, output=12 tokens.
    async def fake_count(self, prompt: str) -> int:
        return 10

    async def fake_generate(self, prompt: str):
        return gc.GenerationResult(text="The capital of Australia is Canberra.", actual_output=12)

    monkeypatch.setattr(gc.GeminiClient, "count_input_tokens", fake_count)
    monkeypatch.setattr(gc.GeminiClient, "generate", fake_generate)
    monkeypatch.setattr(gc.GeminiClient, "is_configured", property(lambda self: True))

    # Reset the demo user's budget for each test.
    quota = budget_store.get("user_123")
    if quota is not None:
        quota.consumed = 0

    with TestClient(app) as test_client:
        yield test_client
