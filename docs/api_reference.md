# API Reference

Base URL: `http://localhost:8000`. Interactive docs at `/docs` (Swagger) and
`/redoc`.

| Method | Path | Description |
|---|---|---|
| `GET`  | `/health` | Liveness/readiness probe. |
| `POST` | `/v1/chat/completions` | Estimate cost, enforce budget, generate. |
| `POST` | `/v1/admin/retrain` | Flush the feedback buffer and retrain now. |
| `GET`  | `/v1/admin/retrain/status` | Retrain state, buffer size, threshold. |
| `POST` | `/v1/admin/generate-dataset` | Background: rebuild the dataset via Gemini. |

---

## `GET /health`

```json
{ "status": "ok", "model_loaded": true, "gemini_configured": true }
```

`model_loaded` is `false` when the `.pkl` artifacts could not be loaded (the
service then serves with a conservative fallback estimate). `gemini_configured`
is `false` when no API key is set.

---

## `POST /v1/chat/completions`

**Request:**
```json
{ "user_id": "user_123", "prompt": "Explain the difference between TCP and UDP." }
```

The response `status` is one of three values:

### `ALLOWED` — generated normally (HTTP 200)
```json
{
  "status": "ALLOWED",
  "texto_generado": "TCP is a connection-oriented protocol...",
  "metadata": {
    "input_tokens": 12,
    "predicted_output": 180,
    "actual_output": 154,
    "prediction_error": -26,
    "latencia_ms": 921.44,
    "presupuesto_restante": 9834,
    "model_status": "OK"
  }
}
```
`prediction_error = actual_output − predicted_output`. `model_status` is `"OK"`
or `"FALLBACK_NO_MODEL"` (no artifacts loaded).

### `REJECTED_BY_BUDGET` — blocked before generation (HTTP 200)
Gemini is **never called**; no cost is incurred.
```json
{
  "status": "REJECTED_BY_BUDGET",
  "texto_generado": null,
  "metadata": {
    "error": "Safeguard Triggered: Estimated output exceeds remaining budget allowance.",
    "estimado": 967,
    "disponible": 10,
    "model_status": "OK"
  }
}
```

### `ALLOWED_PREDICTION_ONLY` — upstream generation failed (HTTP 200)
The guardrail passed but Gemini was unavailable. The estimate is returned with a
warning flag; **the budget is not charged and no feedback is recorded** (there is
no real `actual_output` to learn from).
```json
{
  "status": "ALLOWED_PREDICTION_ONLY",
  "texto_generado": null,
  "metadata": {
    "warning": "GEMINI_UNAVAILABLE",
    "detail": "Upstream generation failed; returning MoPE estimate only. Budget not charged.",
    "input_tokens": 12,
    "predicted_output": 180,
    "estimated_total": 192,
    "model_status": "OK",
    "latencia_ms": 305.1
  }
}
```

### Error cases

| Condition | HTTP | Body |
|---|---|---|
| Unknown `user_id` | 401 | `{"detail": "Unauthorized client"}` |
| Gemini `count_tokens` failure | 502 | `{"detail": "Inference Engine Error: ..."}` |

Budget rejection is **200**, not 4xx: it is an expected business outcome, not an
error.

---

## `POST /v1/admin/retrain`

Flushes the feedback buffer and starts a retrain immediately.
```bash
curl -X POST http://localhost:8000/v1/admin/retrain
```
```json
{ "status": "started", "message": "Retrain triggered manually. Buffer flushed." }
```
If training is already running: `{ "status": "skipped", "message": "Retrain already in progress." }`.

## `GET /v1/admin/retrain/status`
```json
{ "is_training": false, "buffer_size": 12, "retrain_threshold": 50 }
```

## `POST /v1/admin/generate-dataset`

Starts a background job that queries Gemini over `data/prompts.json` and appends
results to `data/dataset.json`. Requires `prompts.json` to be present (download
from Kaggle — see the [retraining guide](retraining_guide.md)).
```json
{ "status": "started", "message": "Dataset generation running in background. Results appended to dataset.json." }
```
