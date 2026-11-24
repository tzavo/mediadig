"""Tests for daemon change-aware reloads and status aggregation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from mediadig import config, db, extract, indexer, server
from mediadig.server import App


def test_app_reload_skips_noop_and_detects_writes_and_replacement(tmp_path: Path) -> None:
    cfg = config.Config((tmp_path,), tmp_path / "data", tmp_path / "data" / "thumbs")
    cfg.data_dir.mkdir(parents=True)
    app = App(cfg, [])
    try:
        # No DB yet -> no-op
        assert app.reload() is False

        # Create DB and write 1 file
        con = db.connect(cfg.db)
        e1 = indexer.Entry("/r/a.txt", "txt", 10, 100, 1, 1)
        indexer.write_file(con, e1, "ok", None, None, [extract.Seg("chunk", text="alpha")], [None])
        con.close()

        assert app.reload() is True
        assert len(app.state.files) == 1
        # Unchanged DB -> skips reload
        assert app.reload() is False

        # External write -> PRAGMA data_version bumps -> triggers reload
        con = db.connect(cfg.db)
        e2 = indexer.Entry("/r/b.txt", "txt", 12, 200, 1, 2)
        indexer.write_file(con, e2, "ok", None, None, [extract.Seg("chunk", text="beta")], [None])
        e3 = indexer.Entry("/r/c.txt", "txt", 5, 300, 1, 3)
        indexer.write_file(con, e3, "error", "corrupt", None, [], [])
        e4 = indexer.Entry("/r/d.mp4", "video", 99, 400, 1, 4)
        indexer.write_file(con, e4, "skipped", "too long", None, [], [])
        con.close()

        assert app.reload() is True
        assert len(app.state.files) == 2

        # Verify single-query status counts
        st = app.status()
        assert st["files"] == 2
        assert st["errors"] == 1
        assert st["skipped"] == 1
        assert st["segments"] == 2

        # Replace DB file with a fresh inode -> detected and reloaded
        cfg.db.unlink()
        for suffix in ("-wal", "-shm"):
            p = cfg.db.with_name(cfg.db.name + suffix)
            p.unlink(missing_ok=True)
        con = db.connect(cfg.db)
        indexer.write_file(con, e1, "ok", None, None, [extract.Seg("chunk", text="only one")], [None])
        con.close()

        assert app.reload() is True
        assert len(app.state.files) == 1
    finally:
        app.close()


def test_app_reload_preserves_state_on_transient_operational_error(tmp_path: Path, monkeypatch) -> None:
    cfg = config.Config((tmp_path,), tmp_path / "data", tmp_path / "data" / "thumbs")
    cfg.data_dir.mkdir(parents=True)
    con = db.connect(cfg.db)
    e1 = indexer.Entry("/r/a.txt", "txt", 10, 100, 1, 1)
    indexer.write_file(con, e1, "ok", None, None, [extract.Seg("chunk", text="alpha")], [None])
    con.close()

    app = App(cfg, [])
    try:
        assert app.reload() is True
        assert len(app.state.files) == 1

        # External write bumps PRAGMA data_version, but load_state transiently fails
        con = db.connect(cfg.db)
        e2 = indexer.Entry("/r/b.txt", "txt", 12, 200, 1, 2)
        indexer.write_file(con, e2, "ok", None, None, [extract.Seg("chunk", text="beta")], [None])
        con.close()

        real_load = server.load_state
        monkeypatch.setattr(server, "load_state", lambda *a, **kw: (_ for _ in ()).throw(sqlite3.OperationalError("locked")))
        assert app.reload() is False
        assert len(app.state.files) == 1  # previous valid state preserved

        # Once transient error clears, the next reload picks up the new version
        monkeypatch.setattr(server, "load_state", real_load)
        assert app.reload() is True
        assert len(app.state.files) == 2
    finally:
        app.close()
