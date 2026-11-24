"""Image sizing for the embedder; port of DigUp Sources/LlamaRuntime/ImageSizing.swift."""

from __future__ import annotations

import io
import math

from PIL import Image


def target_size(w: int, h: int, budget: int) -> tuple[int, int]:
    max_patches = budget * 9  # 3x3 pooling, 16 px patches
    f = math.sqrt(max_patches * 256 / (w * h))
    tw, th = int(f * w / 48) * 48, int(f * h / 48) * 48
    cap = budget * 48
    if tw == 0:
        tw, th = 48, min(th, cap)
    if th == 0:
        th, tw = 48, min(tw, cap)
    return tw, th


def fit_down(img: Image.Image, max_size: tuple[int, int], resample: Image.Resampling) -> Image.Image:
    """Downscale `img` to fit inside `max_size` (matching Image.thumbnail without pre-copying)."""
    x, y = max_size
    if x >= img.width and y >= img.height:
        return img
    aspect = img.width / img.height
    if x / y >= aspect:
        x = max(min(math.floor(y * aspect), math.ceil(y * aspect), key=lambda n: abs(aspect - n / y)), 1)
    else:
        y = max(min(math.floor(x / aspect), math.ceil(x / aspect),
                    key=lambda n: 0 if n == 0 else abs(aspect - x / n)), 1)
    return img.resize((x, y), resample, reducing_gap=2.0)


def prepare_png(img: Image.Image, budget: int) -> bytes:
    size = target_size(img.width, img.height, budget)
    buf = io.BytesIO()
    img.resize(size, Image.Resampling.BICUBIC).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()
