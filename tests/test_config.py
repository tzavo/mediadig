"""Tests for config socket isolation and shared path filtering."""

from __future__ import annotations

from pathlib import Path

from mediadig import config


def test_search_sock_isolation(tmp_path: Path) -> None:
    default_sock = config.search_sock_for(None)
    assert default_sock == config.search_sock_for(config.DEFAULT_DATA_DIR)
    assert default_sock.name == "search.sock"

    a = config.search_sock_for(tmp_path / "dev-a")
    b = config.search_sock_for(tmp_path / "dev-b")
    assert a != default_sock
    assert b != default_sock
    assert a != b
    assert a == config.search_sock_for(tmp_path / "dev-a")
    assert a == config.search_sock_for(str(tmp_path / "sub" / ".." / "dev-a"))
    assert len(str(a)) < 100  # fits comfortably under Linux AF_UNIX 107-byte sun_path limit


def test_config_search_sock_matches_helper(tmp_path: Path) -> None:
    cfg = config.load(roots=[str(tmp_path)], data_dir=str(tmp_path / "data"))
    assert cfg.search_sock == config.search_sock_for(tmp_path / "data")


def test_is_watched_path(tmp_path: Path) -> None:
    root = tmp_path / ".outer-dot" / "roots"
    root.mkdir(parents=True)
    roots = (root,)

    # Supported files inside normal dirs (even when root itself has a dot ancestor)
    assert config.is_watched_path(root / "cat.jpg", roots)
    assert config.is_watched_path(root / "sub" / "clip.MP4", roots)
    assert config.is_watched_path(root / "sub" / ".hidden.png", roots)

    # Unsupported extensions ignored
    assert not config.is_watched_path(root / "notes.tmp", roots)
    assert not config.is_watched_path(root / "archive.zip", roots)

    # Dot-directories under root ignored
    assert not config.is_watched_path(root / ".git" / "objects" / "ab", roots)
    assert not config.is_watched_path(root / ".thumbnails" / "thumb.png", roots)
    assert not config.is_watched_path(root / "sub" / ".Trash" / "old.jpg", roots)
    assert not config.is_watched_path(root / ".hidden_dir", roots)

    # Extensionless regular files ignored, but directories and deleted paths trigger reconcile
    extless_file = root / "Makefile"
    extless_file.write_text("all:\n")
    assert not config.is_watched_path(extless_file, roots)

    sub_dir = root / "memes"
    sub_dir.mkdir()
    assert config.is_watched_path(sub_dir, roots)
    assert config.is_watched_path(root / "deleted_dir", roots)

    # Dotted non-hidden directories must still trigger; hidden dotted dirs must not
    dotted_dir = root / "photos.2025"
    dotted_dir.mkdir()
    assert config.is_watched_path(dotted_dir, roots)
    hidden_dotted = root / ".hidden.2025"
    hidden_dotted.mkdir()
    assert not config.is_watched_path(hidden_dotted, roots)
