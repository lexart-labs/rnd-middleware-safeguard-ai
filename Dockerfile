# syntax=docker/dockerfile:1

# --- Builder: install dependencies into a wheel cache -------------------------
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Runtime: slim image, non-root, no secrets baked in ----------------------
FROM python:3.11-slim AS runtime

# Create an unprivileged user.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Bring in installed packages from the builder.
COPY --from=builder /install /usr/local

# Application code (data/ and .env are excluded via .dockerignore and supplied
# via the compose volume mount / env_file at runtime).
COPY app ./app

USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
