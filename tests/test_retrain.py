"""Tests for the async retraining subsystem."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def _seed_dataset(path: Path, n: int) -> None:
    records = [
        {"prompt": f"sample prompt number {i} about code and data", "output_token": 50 + i}
        for i in range(n)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records), encoding="utf-8")


def test_flush_returns_false_when_already_training(monkeypatch):
    from app.services.retrain import FeedbackBuffer

    buf = FeedbackBuffer()
    buf._is_training = True  # simulate in-progress training
    assert buf.flush() is False


def test_threshold_triggers_retrain_and_hot_reload(monkeypatch, tmp_path):
    from app.config import settings
    from app.services import retrain
    from app.services.model_store import model_store

    # Redirect all artifacts to a temp dir so committed files are untouched.
    monkeypatch.setattr(settings, "dataset_path", tmp_path / "dataset.json")
    monkeypatch.setattr(settings, "model_dir", tmp_path / "models")
    monkeypatch.setattr(settings, "retrain_threshold", 3)
    monkeypatch.setattr(settings, "min_samples_to_train", 2)
    _seed_dataset(settings.dataset_path, 5)

    reloaded = {"count": 0}
    real_reload = model_store.reload

    def counting_reload():
        reloaded["count"] += 1
        return real_reload()

    monkeypatch.setattr(model_store, "reload", counting_reload)

    buf = retrain.FeedbackBuffer()
    for i in range(3):  # hits threshold of 3
        buf.add(f"new prompt {i}", input_tokens=5, actual_tokens=20 + i)

    # Wait for the background retrain to finish.
    deadline = time.time() + 30
    while buf.is_training and time.time() < deadline:
        time.sleep(0.1)

    assert reloaded["count"] >= 1
    assert settings.vectorizer_path.exists()
    assert settings.regressor_path.exists()
    # Dataset grew from 5 seed + 3 new records.
    saved = json.loads(settings.dataset_path.read_text(encoding="utf-8"))
    assert len(saved) == 8
