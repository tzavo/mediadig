"""Tests for indexer walk, reconcile, write_file, and staged run pipeline."""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import numpy as np
from PIL import Image

from mediadig import config, db, extract, indexer
from mediadig.llama import EmbedServerDown


def _age(path: Path, seconds: float = 10.0) -> None:
    t = time.time() - seconds
    os.utime(path, (t, t))


def test_walk_skips_dotdirs_and_symlink_dirs(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "ok").mkdir(parents=True)
    (root / ".git").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()

    f_img = root / "ok" / "a.png"
    f_img.write_bytes(b"x")
    (root / ".git" / "ignored.png").write_bytes(b"x")
    (root / "ok" / "notes.bin").write_bytes(b"x")
    ext_file = outside / "b.txt"
    ext_file.write_text("hi")
    (root / "link_dir").symlink_to(outside)
    (root / "link_file.txt").symlink_to(ext_file)
    (root / "broken.txt").symlink_to(tmp_path / "does_not_exist")

    found = indexer.walk((root,))
    assert str(f_img) in found
    assert str(root / "link_file.txt") in found
    assert str(root / ".git" / "ignored.png") not in found
    assert str(root / "link_dir" / "b.txt") not in found
    assert str(root / "ok" / "notes.bin") not in found
    assert str(root / "broken.txt") not in found


def test_reconcile_moves_deletes_and_deferral(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    a = root / "a.txt"
    b = root / "b.txt"
    fresh = root / "fresh.txt"
    a.write_text("alpha")
    b.write_text("beta")
    fresh.write_text("just written")
    _age(a)
    _age(b)

    con = db.connect(tmp_path / "idx.sqlite")
    try:
        walked = indexer.walk((root,))
        ready, deferred, deleted = indexer.reconcile(con, walked)
        assert {e.path for e in ready} == {str(a), str(b)}
        assert deferred == 1
        assert deleted == 0

        # Persist a and b
        for e in ready:
            indexer.write_file(con, e, "ok", None, None, [extract.Seg("chunk", text="t")], [None])

        # Move a -> a_moved, delete b, age fresh
        a_moved = root / "a_moved.txt"
        a.rename(a_moved)
        b.unlink()
        _age(fresh)

        walked2 = indexer.walk((root,))
        ready2, deferred2, deleted2 = indexer.reconcile(con, walked2)
        assert [e.path for e in ready2] == [str(fresh)]
        assert deferred2 == 0
        assert deleted2 == 1
        paths = {r[0] for r in con.execute("SELECT path FROM files")}
        assert paths == {str(a_moved)}
    finally:
        con.close()


def test_staged_indexer_run_end_to_end_and_reentrant(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "roots"
    root.mkdir()
    img_path = root / "meme.png"
    Image.new("RGB", (200, 120), color=(40, 120, 200)).save(img_path)
    txt_path = root / "note.txt"
    txt_path.write_text("Hello world from a test note about cats and dogs.")
    small_img = root / "tiny.png"
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(small_img)
    for p in (img_path, txt_path, small_img):
        _age(p)

    class FakeClient:
        def embed_texts(self, texts: list[str]) -> np.ndarray:
            v = np.zeros((len(texts), config.EMBED_DIM), np.float32)
            v[:, 0] = 1.0
            return v

        def embed_image_png(self, png: bytes) -> np.ndarray:
            v = np.zeros(config.EMBED_DIM, np.float32)
            v[1] = 1.0
            return v

    class FakeServer:
        def __init__(self, cfg: config.Config):
            self.cfg = cfg

        def __enter__(self) -> FakeClient:
            return FakeClient()

        def __exit__(self, *exc) -> None:
            pass

    class FakeOcr:
        def __init__(self, models_dir: Path):
            pass

        def text(self, img: Image.Image) -> str:
            return "MEME CAPTION"

    monkeypatch.setattr(indexer, "IndexServer", FakeServer)
    monkeypatch.setattr(indexer, "Ocr", FakeOcr)

    cfg = config.Config((root,), tmp_path / "data", tmp_path / "data" / "thumbs")
    rc1 = indexer.run(cfg)
    assert rc1 == config.EXIT_OK

    # Second run in the same process verifies ChildTracker re-entrancy & idle fast path
    txt2 = root / "note2.txt"
    txt2.write_text("Second note added later.")
    _age(txt2)
    rc2 = indexer.run(cfg)
    assert rc2 == config.EXIT_OK

    con = db.connect(cfg.db, readonly=True)
    try:
        rows = dict(con.execute("SELECT path, status FROM files"))
        assert rows[str(img_path)] == "ok"
        assert rows[str(txt_path)] == "ok"
        assert rows[str(txt2)] == "ok"
        assert rows[str(small_img)] == "error"
        segs = con.execute("SELECT count(*) FROM segments").fetchone()[0]
        assert segs == 3
    finally:
        con.close()


def test_media_prefetch_and_fatal_embed_cancellation(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "roots"
    root.mkdir()
    vid = root / "clip.mp4"
    vid.write_bytes(b"fake mp4")
    txt = root / "a.txt"
    txt.write_text("some text")
    _age(vid)
    _age(txt)

    def fake_extract_frames(path: Path, work_dir: Path, tracker=None) -> extract.Frames:
        d = work_dir / "f0"
        d.mkdir(parents=True, exist_ok=True)
        fp = d / "000001.jpg"
        Image.new("RGB", (128, 128), color=(10, 200, 30)).save(fp, format="JPEG")
        return extract.Frames(d, [(0.5, fp)])

    exited = False

    class FailingClient:
        def embed_texts(self, texts: list[str]) -> np.ndarray:
            raise EmbedServerDown("gone")

        def embed_image_png(self, png: bytes) -> np.ndarray:
            v = np.zeros(config.EMBED_DIM, np.float32)
            v[0] = 1.0
            return v

    class TrackingServer:
        def __init__(self, cfg: config.Config):
            pass

        def __enter__(self) -> FailingClient:
            return FailingClient()

        def __exit__(self, *exc) -> None:
            nonlocal exited
            exited = True

    class FakeOcr:
        def __init__(self, models_dir: Path):
            pass

        def text(self, img: Image.Image) -> str:
            return "frame text"

    monkeypatch.setattr(extract, "extract_frames", fake_extract_frames)
    monkeypatch.setattr(indexer, "IndexServer", TrackingServer)
    monkeypatch.setattr(indexer, "Ocr", FakeOcr)

    prev_handler = signal.getsignal(signal.SIGTERM)
    cfg = config.Config((root,), tmp_path / "data", tmp_path / "data" / "thumbs")
    rc = indexer.run(cfg)
    assert rc == config.EXIT_FATAL
    assert exited is True
    assert signal.getsignal(signal.SIGTERM) is prev_handler
    assert not cfg.frames_dir.exists()
