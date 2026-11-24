"""Text normalization, query terms and chunking."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_NON_ALNUM = re.compile(r"[\W_]+")


def normalize(s: str) -> str:
    if s.isascii():
        return _NON_ALNUM.sub(" ", s.lower()).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return _NON_ALNUM.sub(" ", s.casefold()).strip()


STOPWORDS = frozenset(
    "a an the is are was were be been to of in on at for and or it its this that i you he she we they "
    "me my your with as by from not no do does did so if but "
    "de je jsem jsi jsou to se si na v ve z ze k ke s o u do za po pro ale jak co".split()
)

_DOC_KINDS = frozenset({"docx", "pdf", "txt"})
# Normalized word -> file kinds it restricts the search to (see parse_query).
KIND_WORDS: dict[str, frozenset[str]] = {
    **dict.fromkeys(
        ["image", "images", "img", "imgs", "picture", "pictures", "pic", "pics", "photo", "photos",
         "obrazek", "obrazky", "fotka", "fotky", "png", "jpg", "jpeg", "jfif", "webp"],
        frozenset({"image"})),
    **dict.fromkeys(
        ["video", "videos", "videa", "vid", "vids", "clip", "clips", "mp4", "webm", "mov", "mkv", "m4v"],
        frozenset({"video"})),
    **dict.fromkeys(["gif", "gifs"], frozenset({"gif"})),
    **dict.fromkeys(["pdf", "pdfs"], frozenset({"pdf"})),
    "docx": frozenset({"docx"}),
    "txt": frozenset({"txt"}),
    **dict.fromkeys(["doc", "docs", "document", "documents", "dokument", "dokumenty"], _DOC_KINDS),
}


@dataclass(frozen=True)
class Query:
    text: str  # original query minus the type words, for the embedder
    norm: str  # normalized `text`, for phrase matching
    terms: list[str]  # normalized words to match, stopwords removed when others remain
    kinds: frozenset[str]  # restrict results to these file kinds; empty = all kinds


def parse_query(q: str) -> Query:
    """A type word ("img", "vid", "gif", "pdf", ...) as the first or last word of a multi-word
    query restricts the results to that type. Elsewhere it is an ordinary word, so a caption
    like "when the video game loads" still matches as text."""
    words = q.split()
    kinds: frozenset[str] = frozenset()
    for end in (0, -1):
        if len(words) < 2:
            break
        k = KIND_WORDS.get(normalize(words[end]))
        if k:
            kinds |= k
            del words[end]
    text = " ".join(words)
    norm = normalize(text)
    rest = norm.split()
    terms = [w for w in rest if w not in STOPWORDS] or rest
    return Query(text, norm, list(dict.fromkeys(terms)), kinds)


def chunk_text(text: str, size: int = 1800, overlap: int = 200, max_chunks: int = 2000) -> list[str]:
    t = " ".join(text.split())
    out: list[str] = []
    start = 0
    while start < len(t) and len(out) < max_chunks:
        end = start + size
        if end < len(t):
            sp = t.rfind(" ", end - overlap, end)
            if sp > start:
                end = sp
        chunk = t[start:end].strip()
        if chunk:
            out.append(chunk)
        if end >= len(t):
            break
        start += size - overlap
    return out
