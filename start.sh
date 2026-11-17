#!/bin/bash
# bb-embeddings | start.sh | v1.0.1 | 2026-11-17 | BB
# Warmup pass then start uvicorn.
# The warm-up ensures the first real request hits a hot model (no 503 cold-start leak).
# Spec: EMBEDDING_SERVICE.md §7 (cold start mitigation).
set -e

echo "[bb-embeddings] running warmup forward pass..."
python3 -c "
import io
from PIL import Image
from model import load_model, encode_image
# A small solid-gray JPEG, made here by Pillow. The hand-written byte list
# this replaced was not a valid JPEG, so the warmup crashed and the service
# never started (CI run 44137328318).
buf = io.BytesIO()
Image.new('RGB', (224, 224), (128, 128, 128)).save(buf, format='JPEG')
buf.seek(0)
m = load_model()
e = encode_image(m, buf)
print(f'[bb-embeddings] warmup OK - dim={len(e)} norm={sum(x*x for x in e)**0.5:.4f}')
"

echo "[bb-embeddings] starting uvicorn on port ${PORT:-8080}..."
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8080}" --workers 1
