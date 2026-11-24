"""SQLite schema and connections."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files(
  id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, kind TEXT NOT NULL,
  size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, dev INTEGER NOT NULL, ino INTEGER NOT NULL,
  status TEXT NOT NULL,
  error TEXT, thumb TEXT,
  indexed_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS segments(
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  t REAL, page INTEGER,
  text TEXT NOT NULL DEFAULT '',
  vec BLOB);
CREATE INDEX IF NOT EXISTS segments_file_order ON segments(file_id, page, t, id);
"""


def _migrate(con: sqlite3.Connection) -> None:
    ver = con.execute("PRAGMA user_version").fetchone()[0]
    if ver < SCHEMA_VERSION:
        con.execute("DROP INDEX IF EXISTS files_inode")
        con.execute("DROP INDEX IF EXISTS segments_file")
        con.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


def connect(path: Path, readonly: bool = False) -> sqlite3.Connection:
    """Autocommit connection; callers wrap writes in `transaction(con)`."""
    con = sqlite3.connect(path, timeout=10, isolation_level=None, check_same_thread=False)
    try:
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=10000")
        if readonly:
            con.execute("PRAGMA query_only=1")
        else:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            con.executescript(SCHEMA)
            _migrate(con)
    except BaseException:
        con.close()
        raise
    return con


@contextmanager
def transaction(con: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    con.execute("BEGIN IMMEDIATE")
    try:
        yield con
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    else:
        con.execute("COMMIT")


def get_meta(con: sqlite3.Connection, key: str) -> str | None:
    row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
