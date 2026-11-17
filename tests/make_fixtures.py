# bb-embeddings | tests/make_fixtures.py | v1.0.0 | 2026-11-17 | BB
# Provenance only: this wrote the images in tests/fixtures/. The committed image
# FILES are the fixtures; do not re-run this and re-commit, because another Pillow
# or numpy version can write different bytes and silently move every golden vector.
# Needs Pillow and numpy (not the service image).

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261117)


def gradient(w, h):
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :].repeat(h, 0)
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None].repeat(w, 1)
    return Image.fromarray(np.stack([x, y, 255 - (x + y) / 2], axis=2).astype(np.uint8), "RGB")


def checker(n, cell):
    i = np.indices((n, n)).sum(axis=0) // cell % 2
    a = np.where(i[..., None] == 1, np.array([230, 230, 220]), np.array([30, 40, 60]))
    return Image.fromarray(a.astype(np.uint8), "RGB")


def noise(w, h):
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


def shapes(w, h, bg):
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([w * 0.1, h * 0.15, w * 0.45, h * 0.7], fill=(200, 16, 46))
    d.ellipse([w * 0.5, h * 0.2, w * 0.9, h * 0.8], fill=(26, 26, 26))
    d.polygon([(w * 0.2, h * 0.9), (w * 0.5, h * 0.55), (w * 0.8, h * 0.9)], fill=(240, 200, 40))
    d.line([(0, h * 0.05), (w, h * 0.95)], fill=(20, 120, 200), width=5)
    return img


def scene(w, h):
    # photo-like: soft shapes over low-frequency noise, the closest synthetic stand-in for a tap crop
    base = noise(w // 8, h // 8).resize((w, h), Image.BICUBIC)
    img = Image.blend(base, shapes(w, h, (90, 110, 100)), 0.6).filter(ImageFilter.GaussianBlur(2))
    grain = np.asarray(img, dtype=np.int16) + rng.integers(-8, 9, (h, w, 3))
    return Image.fromarray(np.clip(grain, 0, 255).astype(np.uint8), "RGB")


def rgba(n):
    img = shapes(n, n, (245, 245, 245)).convert("RGBA")
    alpha = np.linspace(40, 255, n, dtype=np.uint8)[None, :].repeat(n, 0)
    img.putalpha(Image.fromarray(alpha, "L"))
    return img


gradient(320, 240).save(os.path.join(OUT, "gradient_320x240.png"))
checker(256, 16).save(os.path.join(OUT, "checker_256.png"))
noise(224, 224).save(os.path.join(OUT, "noise_224.png"))
shapes(400, 300, (235, 235, 230)).save(os.path.join(OUT, "shapes_400x300.jpg"), quality=90)
shapes(180, 420, (60, 70, 90)).save(os.path.join(OUT, "tall_180x420.jpg"), quality=85)
scene(512, 384).save(os.path.join(OUT, "scene_512x384.jpg"), quality=88)
gradient(32, 32).save(os.path.join(OUT, "tiny_32.png"))
rgba(300).save(os.path.join(OUT, "alpha_300.png"))
print("wrote", sorted(os.listdir(OUT)))
