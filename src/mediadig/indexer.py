"""`mediadig index`: reconcile the roots with the DB and process new or changed files."""

from __future__ import annotations

import fcntl
import hashlib
import os
import queue
import shutil
import signal
import sqlite3
import stat
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mediadig import config, db, extract
from mediadig.llama import EmbedClient, EmbedServerDown, IndexServer
from mediadig.ocr import Ocr, OcrInitError

if TYPE_CHECKING:
    from PIL import Image

EXIT_OK = config.EXIT_OK
EXIT_FATAL = config.EXIT_FATAL
EXIT_RETRY = config.EXIT_RETRY
RECENT_NS = 3_000_000_000
PREFETCH = 4
PIPELINE_DEPTH = 4
MEDIA_KINDS = ("gif", "video")


@dataclass
class Entry:
    path: str
    kind: str
    size: int
    mtime_ns: int
    dev: int
    ino: int


@dataclass
class _Extracted:
    n: int
    entry: Entry
    segs: list[extract.Seg] | None = None
    thumb: Image.Image | None = None
    t0: float = 0.0
    error: Exception | None = None


def log(msg: str) -> None:
    print(msg, flush=True)


def walk(roots: tuple[Path, ...]) -> dict[str, Entry]:
    found: dict[str, Entry] = {}
    stack = [str(r) for r in roots]
    while stack:
        dirpath = stack.pop()
        try:
            it = os.scandir(dirpath)
        except OSError:
            continue
        with it:
            for de in it:
                name = de.name
                try:
                    if de.is_dir(follow_symlinks=False):
                        if not name.startswith("."):
                            stack.append(de.path)
                        continue
                    kind = config.EXT_KIND.get(os.path.splitext(name)[1].lower())
                    if kind is None:
                        continue
                    st = de.stat(follow_symlinks=True)
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                found[de.path] = Entry(de.path, kind, st.st_size, st.st_mtime_ns, st.st_dev, st.st_ino)
    return found


def thumb_name(e: Entry) -> str:
    return hashlib.sha1(f"{e.dev}:{e.ino}:{e.size}:{e.mtime_ns}".encode()).hexdigest()[:20] + ".png"


def delete_orphan_thumbs(con: sqlite3.Connection, thumbs_dir: Path) -> None:
    if not thumbs_dir.is_dir():
        return
    keep = {r[0] for r in con.execute("SELECT thumb FROM files WHERE thumb IS NOT NULL")}
    for f in thumbs_dir.glob("*.png"):
        if str(f) not in keep:
            f.unlink(missing_ok=True)


def check_fingerprint(con: sqlite3.Connection, thumbs_dir: Path) -> None:
    if db.get_meta(con, "fingerprint") == config.FINGERPRINT:
        return
    log(f"fingerprint changed -> clearing index ({config.FINGERPRINT})")
    with db.transaction(con):
        con.execute("DELETE FROM files")
        db.set_meta(con, "fingerprint", config.FINGERPRINT)
    if thumbs_dir.is_dir():
        for f in thumbs_dir.glob("*.png"):
            f.unlink(missing_ok=True)


def reconcile(con: sqlite3.Connection, walked: dict[str, Entry]) -> tuple[list[Entry], int, int]:
    """Apply moves and deletes; return (queue, deferred count, deleted count)."""
    rows = {r[0]: r[1:] for r in con.execute("SELECT path, id, size, mtime_ns, dev, ino FROM files")}
    missing = [p for p in rows if p not in walked]
    new = {p: e for p, e in walked.items() if p not in rows}
    by_ident = {(e.dev, e.ino, e.size, e.mtime_ns): p for p, e in new.items()}

    moved = deleted = 0
    work_queue: list[Entry] = []
    with db.transaction(con):
        for p in missing:
            fid, size, mtime_ns, dev, ino = rows[p]
            target = by_ident.pop((dev, ino, size, mtime_ns), None)
            if target is not None:
                con.execute("UPDATE files SET path=? WHERE id=?", (target, fid))
                del new[target]
                moved += 1
            else:
                con.execute("DELETE FROM files WHERE id=?", (fid,))
                deleted += 1
        for p, e in walked.items():
            row = rows.get(p)
            if p in new:
                work_queue.append(e)
            elif row is not None:
                fid, size, mtime_ns, dev, ino = row
                if (size, mtime_ns) != (e.size, e.mtime_ns):
                    work_queue.append(e)
                elif (dev, ino) != (e.dev, e.ino):
                    con.execute("UPDATE files SET dev=?, ino=? WHERE id=?", (e.dev, e.ino, fid))

    now = time.time_ns()
    ready = [e for e in work_queue if now - e.mtime_ns >= RECENT_NS]
    deferred = len(work_queue) - len(ready)
    if moved or deleted or deferred:
        log(f"moved={moved} deleted={deleted} deferred={deferred}")
    order = {k: i for i, k in enumerate(config.KIND_ORDER)}
    ready.sort(key=lambda e: (order[e.kind], e.path))
    return ready, deferred, deleted


def write_file(con: sqlite3.Connection, e: Entry, status: str, error: str | None, thumb: str | None,
               segs: list[extract.Seg], vecs: list[np.ndarray | None]) -> None:
    with db.transaction(con):
        fid = con.execute(
            "INSERT INTO files(path, kind, size, mtime_ns, dev, ino, status, error, thumb, indexed_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET kind=excluded.kind, size=excluded.size, "
            "mtime_ns=excluded.mtime_ns, dev=excluded.dev, ino=excluded.ino, status=excluded.status, "
            "error=excluded.error, thumb=excluded.thumb, indexed_at=excluded.indexed_at RETURNING id",
            (e.path, e.kind, e.size, e.mtime_ns, e.dev, e.ino, status, error, thumb, int(time.time())),
        ).fetchone()[0]
        con.execute("DELETE FROM segments WHERE file_id=?", (fid,))
        con.executemany(
            "INSERT INTO segments(file_id, kind, t, page, text, vec) VALUES(?,?,?,?,?,?)",
            [(fid, s.kind, s.t, s.page, s.text, None if v is None else v.astype("<f2").tobytes())
             for s, v in zip(segs, vecs)],
        )


def embed_segments(client: EmbedClient, segs: list[extract.Seg]) -> list[np.ndarray | None]:
    vecs: list[np.ndarray | None] = [None] * len(segs)
    text_idx = [i for i, s in enumerate(segs) if s.doc_text is not None]
    if text_idx:
        tv = client.embed_texts([segs[i].doc_text for i in text_idx])
        for i, v in zip(text_idx, tv):
            vecs[i] = v
    for i, s in enumerate(segs):
        if s.png is not None:
            vecs[i] = client.embed_image_png(s.png)
    return vecs


def _sigterm(signum, frame):
    raise SystemExit(143)


def run(cfg: config.Config, limit: int | None = None) -> int:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    lock_f = open(cfg.lock, "w")
    try:
        try:
            fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("mediadig: index already running", file=sys.stderr)
            return EXIT_RETRY
        prev_sig = None
        is_main = threading.current_thread() is threading.main_thread()
        if is_main:
            prev_sig = signal.signal(signal.SIGTERM, _sigterm)

        con = db.connect(cfg.db)
        try:
            check_fingerprint(con, cfg.thumbs_dir)
            roots = tuple(r for r in cfg.roots if r.is_dir())
            for r in cfg.roots:
                if r not in roots:
                    log(f"WARNING root missing: {r}")
            walked = walk(roots)
            work_queue, deferred, deleted = reconcile(con, walked)
            done_code = EXIT_RETRY if deferred else EXIT_OK
            if limit is not None:
                work_queue = work_queue[:limit]
            log(f"walked={len(walked)} queued={len(work_queue)}")
            if not work_queue:
                if deleted:
                    delete_orphan_thumbs(con, cfg.thumbs_dir)
                return done_code

            cfg.thumbs_dir.mkdir(parents=True, exist_ok=True)
            ocr = Ocr(cfg.ocr_models)
            tracker = extract.ChildTracker()
            media = deque(i for i, e in enumerate(work_queue) if e.kind in MEDIA_KINDS)
            pool: ThreadPoolExecutor | None = None
            pending: deque[tuple[int, Future]] = deque()

            def refill() -> None:
                assert pool is not None
                while len(pending) < PREFETCH and media:
                    i = media.popleft()
                    pending.append((i, pool.submit(extract.extract_frames, Path(work_queue[i].path), cfg.frames_dir, tracker)))

            t_start = time.monotonic()
            try:
                if media:
                    # Lock held: anything left here is from a run that was killed (SIGKILL, power loss).
                    shutil.rmtree(cfg.frames_dir, ignore_errors=True)
                    cfg.frames_dir.mkdir()
                    pool = ThreadPoolExecutor(max_workers=PREFETCH)
                    refill()

                try:
                    server = IndexServer(cfg)
                    client = server.__enter__()
                except RuntimeError as err:
                    log(f"FATAL {err}")
                    return EXIT_FATAL

                out_q: queue.Queue[_Extracted | None] = queue.Queue(maxsize=PIPELINE_DEPTH)
                cancel = threading.Event()

                def _put(item: _Extracted | None) -> None:
                    while not cancel.is_set():
                        try:
                            out_q.put(item, timeout=0.1)
                            return
                        except queue.Full:
                            continue

                def _extract_worker() -> None:
                    try:
                        for n, e in enumerate(work_queue, 1):
                            if cancel.is_set():
                                return
                            t0 = time.monotonic()
                            frames = None
                            try:
                                if e.kind in MEDIA_KINDS:
                                    i, fut = pending.popleft()
                                    assert work_queue[i] is e
                                    refill()
                                    frames = fut.result()
                                    segs, thumb_img = extract.process_frames(frames, ocr, cancel=cancel)
                                elif e.kind == "pdf":
                                    segs, thumb_img = extract.process_pdf(Path(e.path), ocr, cancel=cancel)
                                else:
                                    segs, thumb_img = extract.PROCESSORS[e.kind](Path(e.path), ocr)
                                _put(_Extracted(n, e, segs=segs, thumb=thumb_img, t0=t0))
                            except OcrInitError as err:
                                _put(_Extracted(n, e, t0=t0, error=err))
                                return
                            except Exception as err:
                                _put(_Extracted(n, e, t0=t0, error=err))
                            finally:
                                if frames is not None:
                                    frames.cleanup()
                    finally:
                        _put(None)

                worker = threading.Thread(target=_extract_worker, name="extract", daemon=True)
                worker.start()
                try:
                    while True:
                        item = out_q.get()
                        if item is None:
                            break
                        e = item.entry
                        try:
                            if item.error is not None:
                                raise item.error
                            assert item.segs is not None
                            vecs = embed_segments(client, item.segs)
                            thumb = None
                            if item.thumb is not None:
                                tp = cfg.thumbs_dir / thumb_name(e)
                                item.thumb.save(tp, format="PNG")
                                thumb = str(tp)
                            write_file(con, e, "ok", None, thumb, item.segs, vecs)
                            log(f"[{item.n}/{len(work_queue)}] {e.kind} {e.path} segs={len(item.segs)} {time.monotonic() - item.t0:.2f}s")
                        except (OcrInitError, EmbedServerDown) as err:
                            log(f"FATAL {err}")
                            return EXIT_FATAL
                        except extract.Skipped as err:
                            write_file(con, e, "skipped", str(err), None, [], [])
                            log(f"[{item.n}/{len(work_queue)}] SKIP {e.path}: {err}")
                        except Exception as err:
                            msg = str(err).strip() or type(err).__name__
                            write_file(con, e, "error", msg[:500], None, [], [])
                            log(f"ERROR {e.path}: {msg[:300]}")
                finally:
                    cancel.set()
                    tracker.stop()
                    # Stop llama-server before waiting on the extractor thread so SIGTERM cannot orphan GPU state.
                    server.__exit__(None, None, None)
                    while not out_q.empty():
                        try:
                            out_q.get_nowait()
                        except queue.Empty:
                            break
                    worker.join(timeout=30)

                delete_orphan_thumbs(con, cfg.thumbs_dir)
                log(f"done: {len(work_queue)} files in {time.monotonic() - t_start:.1f}s")
                return done_code
            finally:
                tracker.stop()
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
                    shutil.rmtree(cfg.frames_dir, ignore_errors=True)
        finally:
            con.close()
            if is_main and prev_sig is not None:
                signal.signal(signal.SIGTERM, prev_sig)
    finally:
        lock_f.close()
