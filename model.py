# bb-embeddings | model.py | v1.0.0 | 2026-04-21 | BB
# DINOv2 ViT-S/14 loader + inference.
# Spec: EMBEDDING_SERVICE.md §3 (224-crop classification preset, not 518 _reg variant).
# Output: 384-dim float32 L2-normalized vector, ready for pgvector cosine similarity.

import torch
import torchvision.transforms as T
from PIL import Image

# Official DINOv2 classification preprocessing (Oquab et al. 2023).
# Resize shortest side to 256, center-crop 224, normalize with ImageNet stats.
_TRANSFORM = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def load_model():
    """
    Load DINOv2 ViT-S/14 from torch.hub (weights pre-baked into Docker image).
    Runs on CPU; inference ~40ms on a modern Railway shared-vCPU.
    """
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                            pretrained=True, verbose=False)
    model.eval()
    # Warm-up forward pass (reduces first-request latency seen by clients).
    with torch.no_grad():
        _ = model(torch.zeros(1, 3, 224, 224))
    return model


def encode_image(model, img_buf) -> list[float]:
    """
    Encode a JPEG/PNG file-like object to a 384-dim L2-normalized embedding.

    Args:
        model:   loaded DINOv2 model from load_model()
        img_buf: file-like with .read() (BytesIO or open file)

    Returns:
        list[float] — 384 values, L2-normalized (norm ≈ 1.0 ± 1e-5)
    """
    img = Image.open(img_buf).convert("RGB")
    tensor = _TRANSFORM(img).unsqueeze(0)          # (1, 3, 224, 224)

    with torch.no_grad():
        features = model(tensor)                   # (1, 384)

    # L2-normalize
    norm = features.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = (features / norm).squeeze(0)      # (384,)

    return normalized.tolist()
