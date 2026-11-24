"""`mediadig serve`: search API on a UNIX socket, file watcher and index scheduler."""

from __future__ import annotations

import json
import signal
import socketserver
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlsplit

import watchfiles

from mediadig import config, db
from mediadig.search import State, load_state, make_embedder, search

RELOAD_EVERY = 120.0
RETRY_DEFERRED = 10.0
RETRY_FATAL = 300.0
STOP_WAIT = 30.0


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class App:
    def __init__(self, cfg: config.Config, passthrough: list[str]):
        self.cfg = cfg
        self.passthrough = passthrough
        self.state: State = State()
        self.embed = make_embedder(cfg.embed_sock, timeout=1.0)
        self.dirty = threading.Event()
        self.stop = threading.Event()
        self.proc: subprocess.Popen | None = None
        self.last_exit: int | None = None
        self._ver_con: sqlite3.Connection | None = None
        self._db_ino: tuple[int, int] | None = None
        self._data_version: int | None = None

    def _close_ver_con(self) -> None:
        if self._ver_con is not None:
            self._ver_con.close()
            self._ver_con = None
        self._db_ino = None
        self._data_version = None

    def close(self) -> None:
        self._close_ver_con()
        close_embed = getattr(self.embed, "close", None)
        if close_embed is not None:
            close_embed()

    def reload(self, force: bool = False) -> bool:
        if not self.cfg.db.exists():
            self._close_ver_con()
            if force or self.state.files:
                self.state = State()
                return True
            return False
        try:
            st_stat = self.cfg.db.stat()
            ino = (st_stat.st_dev, st_stat.st_ino)
            if self._ver_con is None or self._db_ino != ino:
                self._close_ver_con()
                self._ver_con = db.connect(self.cfg.db, readonly=True)
                self._db_ino = ino
                self._data_version = None
            ver = self._ver_con.execute("PRAGMA data_version").fetchone()[0]
        except (OSError, sqlite3.Error):
            self._close_ver_con()
            ver = None
        if not force and ver is not None and ver == self._data_version:
            return False
        t0 = time.monotonic()
        try:
            st = load_state(self.cfg.db, strict=True)
        except sqlite3.OperationalError as err:
            log(f"reload deferred: {err}")
            return False
        self.state = st
        self._data_version = ver
        log(f"state loaded: files={len(st.files)} vectors={len(st.V)} texts={len(st.corpus)} "
            f"in {time.monotonic() - t0:.2f}s")
        return True

    def status(self) -> dict:
        files = segments = errors = skipped = 0
        if self.cfg.db.exists():
            con = db.connect(self.cfg.db, readonly=True)
            try:
                counts = dict(con.execute("SELECT status, count(*) FROM files GROUP BY status"))
                files = counts.get("ok", 0)
                errors = counts.get("error", 0)
                skipped = counts.get("skipped", 0)
                segments = con.execute("SELECT count(*) FROM segments").fetchone()[0]
            except Exception:
                pass
            finally:
                con.close()
        return {"files": files, "segments": segments, "errors": errors, "skipped": skipped,
                "indexing": self.proc is not None, "last_index_exit": self.last_exit}

    # --- watcher -----------------------------------------------------------------

    def watch(self) -> None:
        roots = tuple(r for r in self.cfg.roots if r.is_dir())
        if not roots:
            log("watcher: no existing roots")
            return
        for changes in watchfiles.watch(*(str(r) for r in roots), debounce=2000, recursive=True, stop_event=self.stop):
            for _, path in changes:
                if config.is_watched_path(path, roots):
                    self.dirty.set()
                    break

    # --- scheduler ---------------------------------------------------------------

    def schedule(self) -> None:
        run_now = True
        retry_at: float | None = None
        last_reload = 0.0
        while not self.stop.is_set():
            now = time.monotonic()
            if self.proc is None:
                if run_now or self.dirty.is_set() or (retry_at is not None and now >= retry_at):
                    self.dirty.clear()
                    run_now, retry_at = False, None
                    argv = [sys.executable, "-m", "mediadig", "index", *self.passthrough]
                    log(f"index run: {' '.join(argv)}")
                    self.proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL)
                    last_reload = now
            else:
                rc = self.proc.poll()
                if rc is None:
                    if now - last_reload >= RELOAD_EVERY:
                        self.reload()
                        last_reload = now
                else:
                    self.proc = None
                    self.last_exit = rc
                    log(f"index run exited {rc}")
                    self.reload()
                    if rc == config.EXIT_RETRY:
                        retry_at = now + RETRY_DEFERRED
                    elif rc != 0:
                        retry_at = now + RETRY_FATAL
            self.stop.wait(0.5)

    def stop_indexer(self) -> None:
        p = self.proc
        if p is None or p.poll() is not None:
            return
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(STOP_WAIT)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


class Handler(BaseHTTPRequestHandler):
    server: "Server"

    def log_message(self, format: str, *args) -> None:  # client_address is empty on UNIX sockets
        sys.stderr.write("http: " + (format % args) + "\n")

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        app = self.server.app
        url = urlsplit(self.path)
        qs = parse_qs(url.query)
        if url.path == "/search":
            q = qs.get("q", [None])[0]
            if q is None:
                return self._json(400, {"error": "missing q"})
            try:
                n = max(1, min(50, int(qs.get("n", ["10"])[0])))
                min_score = float(qs.get("min", ["0"])[0])
            except ValueError:
                return self._json(400, {"error": "bad n or min"})
            return self._json(200, search(app.state, q, n, min_score, app.embed))
        if url.path == "/status":
            return self._json(200, app.status())
        return self._json(404, {"error": "not found"})


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    app: App


def serve(cfg: config.Config, passthrough: list[str]) -> int:
    sock = cfg.search_sock
    sock.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    sock.unlink(missing_ok=True)

    app = App(cfg, passthrough)
    app.reload()
    srv = Server(str(sock), Handler)
    srv.app = app

    def on_term(signum, frame):
        app.stop.set()

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    threads = [
        threading.Thread(target=srv.serve_forever, name="http", daemon=True),
        threading.Thread(target=app.watch, name="watch", daemon=True),
        threading.Thread(target=app.schedule, name="schedule", daemon=True),
    ]
    for t in threads:
        t.start()
    log(f"serving on {sock}; roots={[str(r) for r in cfg.roots]}")
    while not app.stop.is_set():
        app.stop.wait(1.0)
    log("stopping")
    threads[2].join(5)
    app.stop_indexer()
    srv.shutdown()
    srv.server_close()
    app.close()
    sock.unlink(missing_ok=True)
    return 0
