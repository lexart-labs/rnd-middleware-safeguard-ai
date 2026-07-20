# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A FastAPI middleware that **predicts a Gemini request's output-token cost before
generating**, so per-user token budgets can be enforced without ever overrunning
them. Output length is unknowable up front (the model decides), so a local
**MoPE proxy** (TF-IDF vectorizer + a scikit-learn quantile regressor) estimates
it. "MoPE" (Mixture of Prediction Experts) is the design intent; the current
implementation is a **single quantile regressor** — keep that distinction honest
in docs and comments (see `docs/architecture.md`).

## Commands

```bash
# Run locally
uvicorn app.main:app --reload            # http://localhost:8000, Swagger at /docs

# Run in Docker
docker compose up --build                # serves the api service

# Tests / lint (CI runs both on app/)
pytest -q
pytest tests/test_chat.py::test_name     # single test
ruff check app

# Train / retrain artifacts (writes .pkl into data/models/)
python -m app.training.train_proxy           # offline CLI, from data/dataset.json
docker compose run --rm trainer              # same, in-container (profiles: tools)
```

## Critical constraint: pinned scikit-learn

`scikit-learn` is pinned to **1.8.0** in `requirements.txt`/`pyproject.toml`
because that version produced the committed `.pkl` artifacts in `data/models/`.
Bumping it will likely break unpickling at startup — the store then silently
falls back to a heuristic. Do not upgrade it casually.

## The request hot path (`app/routers/chat.py`)

`POST /v1/chat/completions` runs a fixed 5-step pipeline; understanding it is the
key to the whole service:

1. **Identity** — `budget_store.get(user_id)`; unknown user → `401`.
2. **Input count** — real count via `gemini_client.count_input_tokens` (a cheap
   upstream call). Failure → `502` (`GeminiError`) — the guardrail *needs* a real
   count, so this is a hard error.
3. **Predict** — `predict_output(prompt)` runs locally (vectorize → regress).
4. **Guardrail** — if `input + predicted_output > quota.available`, return
   `REJECTED_BY_BUDGET` **without calling Gemini**. This is the entire point: no
   generation cost is incurred on a reject.
5. **Generate + charge** — otherwise call Gemini, charge the *real* cost, and
   queue feedback via `BackgroundTasks`.

The three response outcomes are the `CompletionStatus` enum: `ALLOWED`,
`REJECTED_BY_BUDGET`, and `ALLOWED_PREDICTION_ONLY` (returned when generation
fails — `GeminiUnavailable` — so the request degrades to an estimate instead of
erroring, and the user is **not** charged).

## Degrade-don't-crash architecture

The service is designed to start and keep serving even when dependencies are
missing; each subsystem reports its state via `/health`:

- **Missing/corrupt model artifacts** → `model_store.is_ready()` is false;
  `predict_output` returns `FALLBACK_OUTPUT_TOKENS` (2048, deliberately high — over-
  predicting blocks a few requests, under-predicting blows budgets).
- **Missing `GEMINI_API_KEY`** → client unconfigured; upstream calls raise.

The two distinct Gemini exceptions encode this policy: `GeminiError` (token count
failed → 502, hard) vs. `GeminiUnavailable` (generation failed → graceful
degrade). Preserve that split when touching `gemini_client.py`.

## Quantile regression, not mean regression

`app/training/pipeline.py::train` is the **single source of truth** for model
building — both the offline CLI (`train_proxy`) and the async retrainer
(`services/retrain.py`) call it, so they can never drift. It uses
`GradientBoostingRegressor(loss="quantile", alpha=0.85)`: the model intentionally
predicts an **upper bound**, because under-prediction is what exhausts budgets.
Don't "fix" this to ordinary regression.

## Retraining loop (`app/services/retrain.py`)

Completed requests feed `(prompt, actual_output)` into a thread-safe
`FeedbackBuffer`. At `RETRAIN_THRESHOLD` records it schedules a retrain on a
**single-worker ThreadPoolExecutor** (CPU-bound training must not block the async
event loop). Retrain appends to `dataset.json`, refits via the shared pipeline,
writes `*_new.pkl` then `os.replace`s them (atomic swap), and hot-reloads through
`model_store.reload()`. Manual trigger: `POST /v1/admin/retrain`; status:
`GET /v1/admin/retrain/status`.

## Conventions & shared state

- **Config**: all env-driven values live in `app/config.py` as one typed
  `Settings` object. Import the shared `settings` instance — never read
  `os.environ` directly in app code.
- **Singletons**: `model_store`, `budget_store`, `gemini_client`, and
  `feedback_buffer` are process-wide singletons, each guarded by its own
  `threading.Lock`. `budget_store` is in-memory and seeded with demo user
  `user_123`; the interface is intentionally minimal so it can be swapped for
  Redis/DB later without touching routers.
- **External API contract**: `ChatResponse` field names are Spanish
  (`texto_generado`, `presupuesto_restante`, etc.) on purpose — they are the
  established external contract. Don't rename them.

## Tests

`tests/conftest.py` fully mocks Gemini (no network, no API key needed):
`count_input_tokens` → 10, `generate` → 12 output tokens, and resets `user_123`'s
budget per test. Tests that exercise the fallback path force `model_store` into a
not-ready state explicitly.

## Data assets

Committed `.pkl` artifacts under `data/models/` make the service run out of the
box. `data/dataset.json` and `data/prompts.json` are **not** committed — download
from Kaggle (see README) only when regenerating the dataset or retraining from
scratch. Docker mounts only `./data` into the container (keeps `.env`/`.git` out).
