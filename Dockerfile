# bb-embeddings | Dockerfile | v1.0.2 | 2026-11-17 | BB
# Pre-bakes DINOv2 ViT-S/14 weights into the image to eliminate CDN dependency
# at cold-start. Spec: EMBEDDING_SERVICE.md §7 (cold start mitigation).

FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow + torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-fetch DINOv2 weights into image (avoids Facebook CDN call at runtime)
# torch.hub caches to TORCH_HOME/hub/checkpoints/
ENV TORCH_HOME=/root/.cache/torch
# model.py's own loader does the fetch, so the cached repo dir and the runtime
# load can never disagree on the ref. skip_validation keeps the GitHub API
# (rate-limited) out of the build.
COPY model.py .
RUN python3 -c "from model import load_model; load_model(); print('DINOv2 weights cached OK')"

COPY . .
RUN chmod +x start.sh

# Port injected by Railway via $PORT env var; default 8080 for local dev
EXPOSE 8080

CMD ["bash", "start.sh"]
