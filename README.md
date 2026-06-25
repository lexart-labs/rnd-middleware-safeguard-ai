# MoPE Predictive Middleware — Token-Budget Safeguard

A marginal-latency predictive middleware that estimates how many tokens a Gemini
request will consume **before** running the expensive generation call, so you can
enforce per-user budgets without ever blowing through them.

It uses a lightweight **Mixture of Prediction Experts (MoPE)** proxy — a TF-IDF
vectorizer plus a quantile-regression model — to predict output verbosity, and a
continuous feedback loop that retrains the model on real request outcomes to
correct prediction drift over time.

> **Status:** research / reference implementation. The current proxy is a single
> quantile regressor; the multi-expert routing described as "MoPE" is the design
> direction, documented honestly in [`docs/architecture.md`](docs/architecture.md).

---

## The problem

When you build on top of an LLM, you bill and limit users by **tokens**. Input
tokens are easy — you can count them up front. **Output** tokens are not: the
model decides how long its answer is, and you only learn the true cost *after*
you've already been charged. A user near their limit can issue one verbose-prompt
request and blow their entire budget.

## How it works

For every `POST /v1/chat/completions`, the middleware runs a five-step pipeline:

1. **Identity** — look up the user's budget; reject unknown users (401).
2. **Input count** — ask Gemini for the real input-token count (a cheap call).
3. **MoPE prediction** — vectorize the prompt and predict output tokens locally
   (negligible latency, no network).
4. **Guardrail** — if `input + predicted_output > available_budget`, reject
   **before** calling Gemini. No generation cost is incurred.
5. **Generate + feedback** — otherwise call Gemini, charge the real cost, and
   asynchronously record `(prompt, actual_output)` to improve the model.

The regressor is trained with **quantile regression** (pinball loss, q=0.85), so
it deliberately predicts toward the upper bound — under-prediction is what
exhausts budgets, so the model is intentionally conservative.

See [`docs/architecture.md`](docs/architecture.md) for the full diagram and the
honest mapping of design intent vs. current implementation.

## Quickstart

```bash
# 1. Clone and configure
git clone <your-repo-url> && cd rnd-middleware-safeguard-ai
cp .env.example .env          # then add your GEMINI_API_KEY

# 2. Get the data assets (see "Datasets & models" below)
#    Place prompts.json and dataset.json under data/

# 3. Run
docker compose up --build
```

The API is then at `http://localhost:8000` (Swagger UI at `/docs`).

Local (without Docker):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Send a request:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "user_123", "prompt": "Explain the difference between TCP and UDP."}'
```

## Configuration

All settings come from environment variables (see [`.env.example`](.env.example)):

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — (required) | Google Gemini API key. Without it, the service runs prediction-only. |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | Model used to serve completions. |
| `DATASET_MODEL` | `gemini-3.1-flash-lite-preview` | Cheaper model used to build the dataset. |
| `RETRAIN_THRESHOLD` | `50` | New feedback records that trigger an automatic retrain. |
| `MIN_SAMPLES_TO_TRAIN` | `5` | Minimum total dataset size before training runs. |
| `REQUEST_DELAY_SECONDS` | `6.0` | Delay between Gemini calls during dataset generation. |
| `DEFAULT_USER_LIMIT` | `10000` | Token budget for the seeded demo user. |

## Datasets & models

- **Pre-trained models** (`data/models/*.pkl`) ship **committed** in this repo, so
  the service runs out of the box with working weights.

  > scikit-learn is pinned to **1.8.0** in `requirements.txt` — the version that
  > produced the committed artifacts. Other versions may fail to unpickle them.

- **`prompts.json` and `dataset.json`**. Download them from Kaggle and place them under `data/`:

  | File | Destination | Kaggle link |
  |---|---|---|
  | `prompts.json` | `data/prompts.json` | _https://www.kaggle.com/datasets/whendelmorais/rnd-middleware-safeguard-ai-data_ |
  | `dataset.json` | `data/dataset.json` | _https://www.kaggle.com/datasets/whendelmorais/rnd-middleware-safeguard-ai-data_ |

  You only need these to **(re)generate** the dataset or **retrain** from scratch;
  serving works with the committed `.pkl` files alone.

## Documentation

- [Architecture](docs/architecture.md) — components, request flow, design intent vs. reality.
- [API reference](docs/api_reference.md) — every route, request/response, error cases.
- [Retraining guide](docs/retraining_guide.md) — Kaggle download, training, the three retrain triggers, hot-swap.

## Roadmap

- True multi-expert MoPE routing per domain (currently a single regressor).
- Local tokenizer to replace the remote `count_tokens` round-trip.
- Persistent quota store + feedback buffer (Redis; an opt-in `cache` service is wired in compose).
- Migrate off the deprecated `google-generativeai` SDK to `google-genai`.

## License

Apache-2.0 — see [LICENSE](LICENSE).
