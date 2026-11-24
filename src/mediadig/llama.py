"""llama-server client over UNIX sockets, and the indexer's own multimodal server."""

from __future__ import annotations

import base64
import http.client
import json
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from mediadig import config

if TYPE_CHECKING:
    import numpy as np

TEXT_BATCH = 16
_POOL_MAX = 8
_STALE_CONN_ERRORS = (http.client.RemoteDisconnected, http.client.BadStatusLine, BrokenPipeError, ConnectionResetError)


def query_prompt(q: str) -> str:
    return f"task: search result | query: {q}"


def doc_prompt(title: str, chunk: str) -> str:
    return f"title: {title} | text: {chunk}"


class EmbedError(Exception):
    """The server answered, but not with a usable embedding (per-file error)."""


class EmbedServerDown(RuntimeError):
    """The server could not be reached or timed out (fatal for an index run)."""


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: Path | str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self.unix_path = str(path)

    def connect(self) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        try:
            s.connect(self.unix_path)
        except BaseException:
            s.close()
            raise
        self.sock = s


def request(sock: Path | str, method: str, url: str, body: bytes | None, timeout: float) -> tuple[int, bytes]:
    conn = UnixHTTPConnection(sock, timeout)
    try:
        headers = {"Host": "localhost"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        conn.request(method, url, body=body, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def _check(vec: list[float]) -> np.ndarray:
    import numpy as np

    v = np.asarray(vec, dtype=np.float32)
    if v.shape != (config.EMBED_DIM,) or not np.isfinite(v).all():
        raise EmbedError("bad embedding")
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise EmbedError("bad embedding")
    return v / n


class EmbedClient:
    def __init__(self, sock: Path | str, timeout: float):
        self.sock = sock
        self.timeout = timeout
        self._lock = threading.Lock()
        self._pool: list[UnixHTTPConnection] = []
        self._closed = False

    def _checkout(self) -> tuple[UnixHTTPConnection, bool]:
        with self._lock:
            if self._pool:
                return self._pool.pop(), True
        return UnixHTTPConnection(self.sock, self.timeout), False

    def _checkin(self, conn: UnixHTTPConnection, keep: bool) -> None:
        if keep:
            with self._lock:
                if not self._closed and len(self._pool) < _POOL_MAX:
                    self._pool.append(conn)
                    return
        conn.close()

    def _post(self, body: bytes) -> tuple[int, bytes]:
        headers = {"Host": "localhost", "Content-Type": "application/json", "Connection": "keep-alive"}
        conn, reused = self._checkout()
        for attempt in range(2):
            try:
                conn.request("POST", "/v1/embeddings", body=body, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
                self._checkin(conn, keep=not resp.will_close)
                return resp.status, data
            except _STALE_CONN_ERRORS:
                self._checkin(conn, keep=False)
                if attempt == 0 and reused:
                    conn, reused = self._checkout()
                    continue
                raise
            except (OSError, http.client.HTTPException):
                self._checkin(conn, keep=False)
                raise
        raise AssertionError("unreachable")

    def close(self) -> None:
        with self._lock:
            self._closed = True
            conns = list(self._pool)
            self._pool.clear()
        for conn in conns:
            conn.close()

    def _embed(self, inputs: list) -> list[np.ndarray]:
        body = json.dumps({"input": inputs}).encode()
        try:
            status, data = self._post(body)
        except (OSError, http.client.HTTPException) as e:
            raise EmbedServerDown(f"embed server unreachable: {e}") from e
        if status != 200:
            raise EmbedError(f"embed HTTP {status}: {data[:200].decode(errors='replace')}")
        try:
            items = sorted(json.loads(data)["data"], key=lambda d: d["index"])
            vecs = [_check(d["embedding"]) for d in items]
        except (ValueError, KeyError, TypeError) as e:
            raise EmbedError(f"bad embedding response: {e}") from e
        if len(vecs) != len(inputs):
            raise EmbedError("bad embedding")
        return vecs

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        import numpy as np

        out: list[np.ndarray] = []
        for i in range(0, len(texts), TEXT_BATCH):
            out.extend(self._embed(texts[i : i + TEXT_BATCH]))
        return np.stack(out) if out else np.zeros((0, config.EMBED_DIM), np.float32)

    def embed_image_png(self, png: bytes) -> np.ndarray:
        url = "data:image/png;base64," + base64.b64encode(png).decode()
        return self._embed([{"content": [{"type": "image_url", "image_url": {"url": url}}]}])[0]


class IndexServer:
    """Multimodal llama-server owned by one index run; stopped on exit to free the GPU."""

    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self.sock = cfg.index_embed_sock(os.getpid())
        self.proc: subprocess.Popen | None = None
        self.client: EmbedClient | None = None

    def __enter__(self) -> EmbedClient:
        self.sock.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.sock.unlink(missing_ok=True)
        argv = [
            config.LLAMA_SERVER,
            "-m", str(config.TEXT_MODEL), "--mmproj", str(config.MMPROJ_MODEL),
            "--embeddings", "-ngl", "99", "-fa", "off", "-np", "1",
            "-c", "2048", "-b", "2048", "-ub", "2048",
            "--image-min-tokens", "1", "--image-max-tokens", "1120",
            "--host", str(self.sock),
        ]
        with open(self.cfg.index_log, "wb") as log:
            self.proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            self._wait_ready()
        except BaseException:
            self._stop()
            raise
        self.client = EmbedClient(self.sock, timeout=120)
        return self.client

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                break
            try:
                status, _ = request(self.sock, "GET", "/health", None, 2.0)
                if status == 200:
                    return
            except (OSError, http.client.HTTPException):
                pass
            time.sleep(0.1)
        raise RuntimeError("index embed server did not start; see index-embed.log")

    def _stop(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.sock.unlink(missing_ok=True)

    def __exit__(self, *exc) -> None:
        self._stop()
