# Retraining Guide

The MoPE model is not static. After the initial training, every successful
request feeds real `(prompt, actual_output)` data back into the system. When
enough new data accumulates, the model retrains in the background and is
hot-swapped into the running server — no restart.

## Data assets

The repo **ships committed pre-trained artifacts** (`data/models/*.pkl`), so
serving works out of the box. You only need the source data below to regenerate
the dataset or retrain from scratch.

| File | Place at | Source |
|---|---|---|
| `vectorizer_mope.pkl` | `data/models/` |this repo |
| `regressor_mope.pkl` | `data/models/` | this repo |
| `prompts.json` | `data/prompts.json` | Kaggle — _https://www.kaggle.com/datasets/whendelmorais/rnd-middleware-safeguard-ai-data_ |
| `dataset.json` | `data/dataset.json` | Kaggle — _https://www.kaggle.com/datasets/whendelmorais/rnd-middleware-safeguard-ai-data_ |

> The committed `.pkl` files were produced with **scikit-learn 1.8.0** (pinned in
> `requirements.txt`). A different version may fail to unpickle them.

### Download from Kaggle

1. Download `prompts.json` and `dataset.json` from the Kaggle dataset (links above).
2. Place them under `data/` as shown in the table.

## Initial / from-scratch training

With `dataset.json` in place, build fresh artifacts:

```bash
python -m app.training.train_proxy
# or, in Docker:
docker compose run --rm trainer
```

This reads `data/dataset.json`, fits a TF-IDF vectorizer + quantile regressor via
the shared pipeline (`app/training/pipeline.py`), and writes the `.pkl` files to
`data/models/`.

### Generating a dataset from prompts

If you have `prompts.json` but not `dataset.json`, build one by querying Gemini
(requires `GEMINI_API_KEY`):

```bash
python -m app.training.generate_dataset
# or trigger it on a running server:
curl -X POST http://localhost:8000/v1/admin/generate-dataset
```

Each prompt is sent to `DATASET_MODEL`; the real output-token count is recorded.
Calls are spaced by `REQUEST_DELAY_SECONDS` to respect rate limits.

## The three retrain triggers

| Trigger | How | Where it's handled |
|---|---|---|
| **Automatic (threshold)** | After `RETRAIN_THRESHOLD` new feedback records accumulate. | `FeedbackBuffer.add` in `app/services/retrain.py` |
| **API** | `POST /v1/admin/retrain` flushes the buffer and retrains now. | `app/routers/admin.py` |
| **Manual (CLI)** | `python -m app.training.train_proxy` / `docker compose run --rm trainer`. | `app/training/train_proxy.py` |

## How the hot-swap works

A background retrain (`_run_retrain`) runs on a single-worker thread pool and:

1. Loads `dataset.json` and appends the buffered records, then persists it.
2. Skips if the total is below `MIN_SAMPLES_TO_TRAIN`.
3. Trains a fresh vectorizer + regressor via the shared pipeline.
4. Writes `vectorizer_new.pkl` / `regressor_new.pkl`, then `os.replace`s them
   onto the production paths (atomic on Unix — never a half-written file).
5. Calls `model_store.reload()` to swap the in-memory artifacts under a lock.

In-flight requests finish on the old model; every subsequent request uses the new
one. If anything fails mid-retrain, the previous known-good model keeps serving.

## Configuration

| Parameter | Env var | Default | Description |
|---|---|---|---|
| Retrain threshold | `RETRAIN_THRESHOLD` | `50` | New records needed to auto-trigger. |
| Min samples to train | `MIN_SAMPLES_TO_TRAIN` | `5` | Floor below which training is skipped. |
| Request delay | `REQUEST_DELAY_SECONDS` | `6.0` | Spacing between dataset-gen calls. |

## Why quantile regression?

A mean regressor predicts the *average* output length — but for budget
enforcement the average is dangerous, since half the time the real output is
longer. Quantile regression at the 85th percentile makes the model deliberately
overshoot, trading a few unnecessary blocks for far fewer budget overruns.

## Caveat: in-memory buffer

The feedback buffer lives in memory and is lost on restart. Persisting it (e.g.
to the opt-in Redis `cache` service) is on the roadmap.
