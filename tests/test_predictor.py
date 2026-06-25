"""Tests for the MoPE predictor, including the missing-artifact fallback."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")


def test_predict_with_loaded_model():
    from app.services.model_store import model_store
    from app.services.predictor import predict_output

    assert model_store.load() is True
    pred = predict_output("Summarize the benefits of NVMe SSDs.")
    assert pred.model_status == "OK"
    assert pred.predicted_output >= 0


def test_predict_falls_back_when_model_missing():
    from app.services.model_store import ModelStore
    from app.services.predictor import FALLBACK_OUTPUT_TOKENS, predict_output

    empty_store = ModelStore()  # never loaded -> not ready
    pred = predict_output("anything", store=empty_store)
    assert pred.model_status == "FALLBACK_NO_MODEL"
    assert pred.predicted_output == FALLBACK_OUTPUT_TOKENS
