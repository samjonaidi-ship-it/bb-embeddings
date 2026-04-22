#!/bin/bash
# bb-embeddings | start.sh | v1.1.0 | 2026-04-21 | BB
# Start uvicorn immediately so Railway healthcheck can probe /healthz.
# Model warm-up runs inside FastAPI startup event (load_model() in main.py);
# /healthz returns 503 until load_model() completes, then 200.
# healthcheckTimeout in railway.json = 180s to cover CPU model load.
set -e

echo "[bb-embeddings] starting uvicorn on port ${PORT:-8080}..."
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8080}" --workers 1
