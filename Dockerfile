# bb-embeddings | Dockerfile | v1.0.0 | 2026-04-21 | BB
# Pre-bakes DINOv2 ViT-S/14 weights into the image to eliminate CDN dependency
# at cold-start. Spec: EMBEDDING_SERVICE.md §7 (cold start mitigation).

FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow + torch (libgl1 replaces libgl1-mesa-glx in Debian Bookworm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-fetch DINOv2 weights into image (avoids Facebook CDN call at runtime)
# torch.hub caches to TORCH_HOME/hub/checkpoints/
ENV TORCH_HOME=/root/.cache/torch
RUN python3 -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True, verbose=False); print('DINOv2 weights cached OK')"

COPY . .
RUN chmod +x start.sh

# Port injected by Railway via $PORT env var; default 8080 for local dev
EXPOSE 8080

CMD ["bash", "start.sh"]
