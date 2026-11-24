"""Tests for text normalization, query parsing, and chunking."""

from __future__ import annotations

import re
import unicodedata

from mediadig import textnorm


def _ref_normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.compile(r"[\W_]+").sub(" ", s.casefold()).strip()


def test_normalize_parity_ascii_and_unicode() -> None:
    samples = [
        "",
        "   ---___   ",
        "Hello, World! 123_456",
        "ALL CAPS & Symbols: $100 / 50%",
        "Příliš žluťoučký kůň úpěl ďábelské ódy",
        "Café naïve résumé",
        "Straße vs STRASSE",
        "line1\n\tline2\r\nline3",
    ]
    for s in samples:
        assert textnorm.normalize(s) == _ref_normalize(s)


def test_parse_query_kind_words_and_stopwords() -> None:
    q = textnorm.parse_query("vid funny cat on the table")
    assert q.kinds == frozenset({"video"})
    assert q.text == "funny cat on the table"
    assert q.terms == ["funny", "cat", "table"]

    q2 = textnorm.parse_query("kočka fotky")
    assert q2.kinds == frozenset({"image"})
    assert q2.terms == ["kocka"]

    # Kind word in the middle stays an ordinary term
    q3 = textnorm.parse_query("when the video game loads")
    assert q3.kinds == frozenset()
    assert "video" in q3.terms

    # All-stopword query keeps original words
    q4 = textnorm.parse_query("to be or not to be")
    assert q4.terms == ["to", "be", "or", "not"]


def test_chunk_text_overlap_and_whitespace() -> None:
    text = " ".join(f"word{i}" for i in range(400))
    chunks = textnorm.chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 5
    assert all("\n" not in c and "  " not in c for c in chunks)
    # Adjacent chunks overlap
    assert chunks[0].split()[-1] in chunks[1]
