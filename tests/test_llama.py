"""Tests for lazy numpy import and EmbedClient keep-alive / reconnect."""

from __future__ import annotations

import json
import socketserver
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from mediadig import config
from mediadig.llama import EmbedClient


def test_request_import_does_not_load_numpy() -> None:
    code = "import sys; from mediadig.llama import request; assert 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_embed_client_keepalive_and_reconnect(tmp_path: Path) -> None:
    sock = tmp_path / "embed.sock"
    connections = 0
    requests_seen = 0
    drop_after = {2}  # force server to close socket after 2nd request to exercise reconnect

    class FakeHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args) -> None:
            pass

        def setup(self) -> None:
            nonlocal connections
            connections += 1
            super().setup()

        def do_POST(self) -> None:
            nonlocal requests_seen
            requests_seen += 1
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            n = len(payload["input"])
            vec = [1.0] + [0.0] * (config.EMBED_DIM - 1)
            body = json.dumps({"data": [{"index": i, "embedding": vec} for i in range(n)]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if requests_seen in drop_after:
                # Close abruptly without telling client via Connection: close, simulating stale keep-alive
                self.close_connection = True
            self.end_headers()
            self.wfile.write(body)

    class UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

    srv = UnixServer(str(sock), FakeHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    client = EmbedClient(sock, timeout=5.0)
    try:
        # First 2 calls share 1 keep-alive connection
        v1 = client.embed_texts(["a"])
        v2 = client.embed_texts(["b"])
        assert v1.shape == (1, config.EMBED_DIM)
        assert v2.shape == (1, config.EMBED_DIM)
        assert connections == 1

        # 3rd call hits the closed keep-alive socket, transparently reconnects, and succeeds
        v3 = client.embed_texts(["c"])
        v4 = client.embed_texts(["d"])
        assert v3.shape == (1, config.EMBED_DIM)
        assert v4.shape == (1, config.EMBED_DIM)
        assert connections == 2
        assert requests_seen == 4

        # 10 short-lived threads (like ThreadingMixIn HTTP requests) reuse the pooled connection
        for idx in range(10):
            err: list[BaseException] = []

            def worker(i: int = idx) -> None:
                try:
                    assert client.embed_texts([f"t{i}"]).shape == (1, config.EMBED_DIM)
                except BaseException as exc:
                    err.append(exc)

            wt = threading.Thread(target=worker)
            wt.start()
            wt.join()
            assert not err
        assert connections == 2
        assert len(client._pool) == 1
    finally:
        client.close()
        assert len(client._pool) == 0
        srv.shutdown()
        srv.server_close()
