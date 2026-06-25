"""MoPE output-token predictor.

Wraps the vectorize -> regress step. When the model store is not ready (missing
or corrupt artifacts) it returns a conservative heuristic estimate rather than
dereferencing ``None``, so the guardrail can still make a safe decision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.services.model_store import ModelStore, model_store

logger = logging.getLogger(__name__)

# Conservative fallback: assume a sizeable response when no model is available.
# High by design — overestimating blocks a few good requests; underestimating
# blows budgets.
FALLBACK_OUTPUT_TOKENS = 2048


@dataclass(frozen=True)
class Prediction:
    """Result of an output-token prediction."""

    predicted_output: int
    model_status: str  # "OK" or "FALLBACK_NO_MODEL"


def predict_output(prompt: str, store: ModelStore = model_store) -> Prediction:
    """Estimate output-token count for ``prompt``.

    Falls back to ``FALLBACK_OUTPUT_TOKENS`` with a ``FALLBACK_NO_MODEL`` status
    when artifacts are unavailable.
    """
    if not store.is_ready():
        logger.warning("[predictor] No model loaded — using fallback estimate.")
        return Prediction(FALLBACK_OUTPUT_TOKENS, "FALLBACK_NO_MODEL")

    vectorizer, regressor = store.artifacts
    features = vectorizer.transform([prompt])
    predicted = int(regressor.predict(features)[0])
    return Prediction(max(predicted, 0), "OK")
