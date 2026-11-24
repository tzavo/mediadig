"""Sample corpus benchmark test for initial indexing pipeline and search speed."""

from __future__ import annotations

import os
import time
from pathlib import Path

import docx
import numpy as np
from PIL import Image, ImageDraw

from mediadig import config, indexer, search


def _populate_sample_corpus(root: Path, n_images: int = 30, n_txt: int = 30, n_docx: int = 10) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    topics = [
        ("cat", "orange cat sleeping on keyboard meme"),
        ("dog", "golden retriever catching frisbee in park"),
        ("invoice", "quarterly cloud hosting invoice pdf summary"),
        ("czech", "příliš žluťoučký kůň úpěl ďábelské ódy poznámka"),
        ("code", "python sqlite wal benchmark and numpy vector search"),
    ]
    old = time.time() - 30.0

    for i in range(n_images):
        arr = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        im = Image.fromarray(arr, mode="RGB")
        draw = ImageDraw.Draw(im)
        draw.rectangle((10, 10, 220, 60), fill=(255, 255, 255))
        p = root / f"img_{i:03d}_{topics[i % len(topics)][0]}.png"
        im.save(p, format="PNG", compress_level=1)
        os.utime(p, (old, old))

    for i in range(n_txt):
        tag, body = topics[i % len(topics)]
        p = root / f"note_{i:03d}_{tag}.txt"
        p.write_text(f"Document {i}: {body}.\n" + ("Additional context sentence. " * 20))
        os.utime(p, (old, old))

    for i in range(n_docx):
        tag, body = topics[i % len(topics)]
        d = docx.Document()
        d.add_paragraph(f"Report {i}: {body}")
        tbl = d.add_table(rows=2, cols=2)
        tbl.cell(0, 0).text = "metric"
        tbl.cell(0, 1).text = f"value_{i}"
        tbl.cell(1, 0).text = tag
        tbl.cell(1, 1).text = body
        p = root / f"doc_{i:03d}_{tag}.docx"
        d.save(str(p))
        os.utime(p, (old, old))


def test_sample_indexing_and_search_performance(tmp_path: Path, monkeypatch) -> None:
    """Builds a 70-file mixed sample (PNG/TXT/DOCX), indexes it through the real pipeline
    with deterministic fake OCR/embed latency, then times load_state and search queries."""
    root = tmp_path / "sample_roots"
    data_dir = tmp_path / "sample_data"
    _populate_sample_corpus(root)

    rng = np.random.default_rng(99)

    def _unit_vec(seed_text: str) -> np.ndarray:
        h = abs(hash(seed_text)) % (2**32)
        v = np.random.default_rng(h).standard_normal(config.EMBED_DIM).astype(np.float32)
        return v / np.linalg.norm(v)

    class BenchClient:
        def embed_texts(self, texts: list[str]) -> np.ndarray:
            return np.stack([_unit_vec(t) for t in texts])

        def embed_image_png(self, png: bytes) -> np.ndarray:
            return _unit_vec(str(len(png)))

    class BenchServer:
        def __init__(self, cfg: config.Config):
            pass

        def __enter__(self) -> BenchClient:
            return BenchClient()

        def __exit__(self, *exc) -> None:
            pass

    class BenchOcr:
        def __init__(self, models_dir: Path):
            pass

        def text(self, img: Image.Image) -> str:
            return "sample meme caption cat dog"

    monkeypatch.setattr(indexer, "IndexServer", BenchServer)
    monkeypatch.setattr(indexer, "Ocr", BenchOcr)

    cfg = config.Config((root,), data_dir, data_dir / "thumbs")
    t0 = time.perf_counter()
    rc = indexer.run(cfg)
    index_ms = (time.perf_counter() - t0) * 1000
    assert rc == config.EXIT_OK

    t1 = time.perf_counter()
    st = search.load_state(cfg.db)
    load_ms = (time.perf_counter() - t1) * 1000
    assert len(st.files) == 70

    qv = rng.standard_normal(config.EMBED_DIM).astype(np.float32)
    qv /= np.linalg.norm(qv)
    fake_embed = lambda _q: qv

    queries = ["cat keyboard", "invoice summary", "vid dog", "prilis zlutoucky kun", "python sqlite vector"]
    t2 = time.perf_counter()
    for q in queries:
        res = search.search(st, q, n=10, min_score=0.0, embed=fake_embed)
        assert res["semantic"] is True
    avg_search_ms = (time.perf_counter() - t2) * 1000 / len(queries)

    # Confirm actual matches come back for text/image queries
    assert len(search.search(st, "keyboard meme", n=5, min_score=0.0, embed=fake_embed)["results"]) > 0
    assert len(search.search(st, "prilis zlutoucky kun", n=5, min_score=0.0, embed=fake_embed)["results"]) > 0
    assert len(search.search(st, "doc report", n=5, min_score=0.0, embed=fake_embed)["results"]) > 0

    print(
        f"\n[sample-bench] 70 files (30 PNG, 30 TXT, 10 DOCX): "
        f"index={index_ms:.1f}ms load_state={load_ms:.2f}ms avg_search={avg_search_ms:.2f}ms"
    )
