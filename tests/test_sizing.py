"""Tests for image sizing, fit_down parity, and PNG encoding."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from mediadig import sizing


def test_target_size_multiples_and_budget() -> None:
    for w, h, budget in [(1920, 1080, 560), (1080, 1920, 140), (4000, 30, 560), (30, 4000, 140), (48, 48, 560)]:
        tw, th = sizing.target_size(w, h, budget)
        assert tw >= 48 and th >= 48
        assert tw % 48 == 0 and th % 48 == 0
        assert (tw // 48) * (th // 48) <= budget


def test_fit_down_matches_pillow_thumbnail_exact_pixels() -> None:
    rng = np.random.default_rng(7)
    for size in [(2400, 1350), (1350, 2400), (1600, 1600), (800, 600), (3000, 47), (47, 3000)]:
        arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        im = Image.fromarray(arr, mode="RGB")

        for max_size, resample in [((1600, 1600), Image.Resampling.BICUBIC), ((96, 96), Image.Resampling.LANCZOS)]:
            got = sizing.fit_down(im, max_size, resample)
            ref = im.copy()
            ref.thumbnail(max_size, resample)
            assert got.size == ref.size
            assert np.array_equal(np.asarray(got), np.asarray(ref))


def test_prepare_png_is_lossless_vs_default_compression() -> None:
    rng = np.random.default_rng(3)
    im = Image.fromarray(rng.integers(0, 256, (600, 800, 3), dtype=np.uint8), mode="RGB")
    raw = sizing.prepare_png(im, 140)
    decoded = Image.open(io.BytesIO(raw))

    ref_buf = io.BytesIO()
    im.resize(sizing.target_size(800, 600, 140), Image.Resampling.BICUBIC).save(ref_buf, format="PNG")
    ref = Image.open(io.BytesIO(ref_buf.getvalue()))

    assert decoded.size == ref.size
    assert np.array_equal(np.asarray(decoded), np.asarray(ref))
