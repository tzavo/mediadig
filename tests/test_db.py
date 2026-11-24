"""Tests for SQLite schema, migrations, index plans, and transactions."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mediadig import db


def test_schema_migration_drops_dead_indexes_and_sets_version(tmp_path: Path) -> None:
    dbp = tmp_path / "legacy.sqlite"
    # Simulate an existing v0 database with the legacy indexes
    raw = sqlite3.connect(dbp)
    raw.executescript(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE files(
          id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, kind TEXT NOT NULL,
          size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, dev INTEGER NOT NULL, ino INTEGER NOT NULL,
          status TEXT NOT NULL, error TEXT, thumb TEXT, indexed_at INTEGER NOT NULL);
        CREATE INDEX files_inode ON files(dev, ino);
        CREATE TABLE segments(
          id INTEGER PRIMARY KEY,
          file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
          kind TEXT NOT NULL, t REAL, page INTEGER, text TEXT NOT NULL DEFAULT '', vec BLOB);
        CREATE INDEX segments_file ON segments(file_id);
        """
    )
    raw.close()

    con = db.connect(dbp)
    try:
        assert con.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
        indexes = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        assert "files_inode" not in indexes
        assert "segments_file" not in indexes
        assert "segments_file_order" in indexes
        assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert con.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL

        plan = [
            r[3]
            for r in con.execute(
                "EXPLAIN QUERY PLAN "
                "SELECT f.id, f.path, f.kind, f.thumb, s.kind, s.t, s.page, s.text, s.vec "
                "FROM segments s JOIN files f ON f.id = s.file_id WHERE f.status = 'ok' "
                "ORDER BY f.id, s.page, s.t, s.id"
            )
        ]
        assert not any("TEMP B-TREE" in step for step in plan), plan
    finally:
        con.close()


def test_transaction_commits_and_rolls_back(tmp_path: Path) -> None:
    con = db.connect(tmp_path / "tx.sqlite")
    try:
        with db.transaction(con):
            db.set_meta(con, "k", "v1")
        assert db.get_meta(con, "k") == "v1"

        with pytest.raises(RuntimeError):
            with db.transaction(con):
                db.set_meta(con, "k", "v2")
                raise RuntimeError("boom")
        assert db.get_meta(con, "k") == "v1"
        assert not con.in_transaction
    finally:
        con.close()


def test_readonly_connect_rejects_writes(tmp_path: Path) -> None:
    dbp = tmp_path / "ro.sqlite"
    db.connect(dbp).close()
    ro = db.connect(dbp, readonly=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            db.set_meta(ro, "x", "y")
    finally:
        ro.close()
