# bb-embeddings | main.py | v1.0.0 | 2026-04-21 | BB
# FastAPI service: POST /embed -> DINOv2 ViT-S/14 384-dim L2-normalized vector.
# Spec: EMBEDDING_SERVICE.md v1.0
# Auth: x-embed-token header equality check (EMBED_SERVICE_TOKEN env var).
# Never logs request bodies (tap crops may contain crew faces).

import os, time, base64, io, logging
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from model import load_model, encode_image

logging.basicConfig(level=os.getenv("LOG_LEVEL", "info").upper(),
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bb-embeddings")

EMBED_TOKEN  = os.getenv("EMBED_SERVICE_TOKEN", "")
MAX_BYTES    = 256 * 1024      # 256 KB post-base64 cap
_boot_start  = time.monotonic()

app = FastAPI(title="bb-embeddings", version="1.0.0")

# ── Model (loaded at startup) ──────────────────────────────────────────────
_model = None
_model_loaded = False

@app.on_event("startup")
async def startup():
    global _model, _model_loaded
    t0 = time.monotonic()
    _model = load_model()
    _model_loaded = True
    ms = round((time.monotonic() - t0) * 1000)
    logger.info({"event": "embed.cold_start", "boot_ms": ms})

# ── Schemas ────────────────────────────────────────────────────────────────
class EmbedRequest(BaseModel):
    image_b64:  str
    request_id: str = ""

class EmbedResponse(BaseModel):
    request_id:   str
    embedding:    list[float]
    model:        str
    model_version:str
    dim:          int
    inference_ms: int

# ── Auth helper ────────────────────────────────────────────────────────────
def _check_token(tok: str):
    if not EMBED_TOKEN:
        return          # no token configured → open (dev mode)
    if tok != EMBED_TOKEN:
        raise HTTPException(status_code=401, detail="invalid x-embed-token")

# ── Routes ─────────────────────────────────────────────────────────────────
@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest,
                x_embed_token: str = Header(default="")):
    if not _model_loaded:
        return JSONResponse(status_code=503,
                            content={"error": "model_loading"})
    _check_token(x_embed_token)

    raw_b64 = req.image_b64.strip()
    if len(raw_b64) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="payload too large")

    try:
        img_bytes = base64.b64decode(raw_b64)
        img_buf   = io.BytesIO(img_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="malformed base64 or image")

    t0 = time.monotonic()
    try:
        embedding = encode_image(_model, img_buf)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"image decode failed: {e}")
    inference_ms = round((time.monotonic() - t0) * 1000)

    logger.info({"event": "embed.request",
                 "request_id": req.request_id,
                 "bytes_in": len(raw_b64),
                 "status": 200,
                 "inference_ms": inference_ms})

    return EmbedResponse(
        request_id    = req.request_id,
        embedding     = embedding,
        model         = "dinov2_vits14",
        model_version = "1.0",
        dim           = len(embedding),
        inference_ms  = inference_ms,
    )

@app.get("/healthz")
async def healthz():
    return {
        "ok":           _model_loaded,
        "model_loaded": _model_loaded,
        "uptime_s":     round(time.monotonic() - _boot_start),
    }

@app.get("/version")
async def version():
    import torch
    return {
        "service":       "bb-embeddings",
        "version":       "1.0.0",
        "model":         "dinov2_vits14",
        "torch":         torch.__version__,
    }
