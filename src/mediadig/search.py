"""In-memory search state and scoring (adapted from DigUp Sources/DigUpKit/Searcher.swift)."""

from __future__ import annotations

import functools
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from rapidfuzz import fuzz, process

from mediadig import config, db
from mediadig.llama import EmbedClient, EmbedError, EmbedServerDown, query_prompt
from mediadig.textnorm import normalize, parse_query

Z_GATE = 2.0
STD_FLOOR = 0.02
GROUP_MIN_ROWS = 20
WORD_WEIGHT = 4.0
COVERAGE_GATE = 0.5
TOP_CUT = 0.5
PHRASE_MIN_LEN = 8
PHRASE_SCORE = 90
SHORT_TERM_LEN = 4
SHORT_TERM_SCORE = 100
LONG_TERM_SCORE = 85

VISUAL, TEXT = 0, 1
_GROUP = {"image": VISUAL, "frame": VISUAL, "page_image": VISUAL, "chunk": TEXT}
_KIND_ID = {k: i for i, k in enumerate(config.KIND_ORDER)}
_VEC_CHUNK = 4096


def _file_ranges(owner: np.ndarray, nf: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build contiguous per-file slice bounds from a monotonically grouped `owner` array."""
    pos = np.full(nf, -1, np.int64)
    if not len(owner):
        empty = np.zeros(0, np.int64)
        return empty, empty, empty, pos
    changes = np.flatnonzero(owner[1:] != owner[:-1]) + 1
    starts = np.empty(len(changes) + 1, np.int64)
    starts[0] = 0
    starts[1:] = changes
    ends = np.empty_like(starts)
    ends[:-1] = changes
    ends[-1] = len(owner)
    files = owner[starts]
    if len(np.unique(files)) != len(files):
        raise ValueError("owner array must be grouped contiguously by file")
    pos[files] = np.arange(len(files), dtype=np.int64)
    return files, starts, ends, pos


@dataclass
class State:
    files: list[tuple[str, str, str | None]] = field(default_factory=list)  # (path, kind, thumb)
    V: np.ndarray = field(default_factory=lambda: np.zeros((0, config.EMBED_DIM), np.float32))
    vec_seg: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    vec_file: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    vec_group: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int8))
    seg_t: list[float | None] = field(default_factory=list)
    seg_page: list[int | None] = field(default_factory=list)
    seg_text: list[str] = field(default_factory=list)
    corpus: list[str] = field(default_factory=list)
    corpus_seg: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    corpus_file: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    corpus_len: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    file_kind: np.ndarray = field(init=False)
    vec_files: np.ndarray = field(init=False)
    vec_starts: np.ndarray = field(init=False)
    vec_ends: np.ndarray = field(init=False)
    vec_file_pos: np.ndarray = field(init=False)
    corpus_files: np.ndarray = field(init=False)
    corpus_starts: np.ndarray = field(init=False)
    corpus_ends: np.ndarray = field(init=False)
    corpus_file_pos: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        nf = len(self.files)
        if len(self.corpus_len) != len(self.corpus):
            self.corpus_len = np.asarray([len(c) for c in self.corpus], np.int64)
        self.file_kind = np.asarray([_KIND_ID.get(f[1], -1) for f in self.files], np.int8)
        self.vec_files, self.vec_starts, self.vec_ends, self.vec_file_pos = _file_ranges(self.vec_file, nf)
        self.corpus_files, self.corpus_starts, self.corpus_ends, self.corpus_file_pos = _file_ranges(self.corpus_file, nf)


def load_state(db_path: Path, strict: bool = False) -> State:
    if not db_path.exists():
        return State()
    try:
        con = db.connect(db_path, readonly=True)
    except sqlite3.OperationalError:
        if strict:
            raise
        return State()
    try:
        cur = con.execute(
            "SELECT f.id, f.path, f.kind, f.thumb, s.kind, s.t, s.page, s.text, s.vec "
            "FROM segments s JOIN files f ON f.id = s.file_id WHERE f.status = 'ok' "
            "ORDER BY f.id, s.page, s.t, s.id"
        )
        files: list[tuple[str, str, str | None]] = []
        file_idx: dict[int, int] = {}
        vecs: list[bytes] = []
        vec_seg: list[int] = []
        vec_file: list[int] = []
        vec_group: list[int] = []
        seg_t: list[float | None] = []
        seg_page: list[int | None] = []
        seg_text: list[str] = []
        corpus: list[str] = []
        corpus_seg: list[int] = []
        corpus_file: list[int] = []
        seen: set[str] = set()
        for fid, path, fkind, thumb, skind, t, page, text, vec in cur:
            fi = file_idx.get(fid)
            if fi is None:
                fi = file_idx[fid] = len(files)
                files.append((path, fkind, thumb))
                seen = set()
            si = len(seg_text)
            seg_t.append(t)
            seg_page.append(page)
            seg_text.append(text)
            if vec is not None:
                vecs.append(vec)
                vec_seg.append(si)
                vec_file.append(fi)
                vec_group.append(_GROUP[skind])
            tn = normalize(text)
            if tn and tn not in seen:
                seen.add(tn)
                corpus.append(tn)
                corpus_seg.append(si)
                corpus_file.append(fi)
    except sqlite3.OperationalError:
        if strict:
            raise
        return State()
    finally:
        con.close()
    if vecs:
        V = np.empty((len(vecs), config.EMBED_DIM), np.float32)
        for i in range(0, len(vecs), _VEC_CHUNK):
            chunk = vecs[i : i + _VEC_CHUNK]
            V[i : i + len(chunk)] = np.frombuffer(b"".join(chunk), dtype="<f2").reshape(-1, config.EMBED_DIM)
    else:
        V = np.zeros((0, config.EMBED_DIM), np.float32)
    return State(
        files=files,
        V=V,
        vec_seg=np.asarray(vec_seg, np.int64),
        vec_file=np.asarray(vec_file, np.int64),
        vec_group=np.asarray(vec_group, np.int8),
        seg_t=seg_t,
        seg_page=seg_page,
        seg_text=seg_text,
        corpus=corpus,
        corpus_seg=np.asarray(corpus_seg, np.int64),
        corpus_file=np.asarray(corpus_file, np.int64),
        corpus_len=np.asarray([len(c) for c in corpus], np.int64),
    )


def make_embedder(sock: Path, timeout: float = 1.0) -> Callable[[str], np.ndarray]:
    client = EmbedClient(sock, timeout)

    @functools.lru_cache(maxsize=128)
    def embed(q: str) -> np.ndarray:
        return client.embed_texts([query_prompt(q)])[0]

    embed.close = client.close  # type: ignore[attr-defined]
    return embed


def _fuzzy_matches(state: State, needle: str, threshold: int) -> np.ndarray:
    """Boolean mask of corpus texts matching `needle` at or above `threshold`."""
    n = len(state.corpus)
    if threshold >= 100:
        return np.fromiter((needle in c for c in state.corpus), dtype=bool, count=n)
    out = np.zeros(n, bool)
    nlen = len(needle)
    long_idx = np.flatnonzero(state.corpus_len >= nlen)
    min_short = (nlen * threshold) // (200 - threshold)
    short_idx = np.flatnonzero((state.corpus_len < nlen) & (state.corpus_len >= min_short))
    for idx, scorer in ((long_idx, fuzz.partial_ratio), (short_idx, fuzz.ratio)):
        if len(idx):
            texts = state.corpus if len(idx) == n else [state.corpus[i] for i in idx]
            scores = process.cdist([needle], texts, scorer=scorer, score_cutoff=threshold,
                                   workers=-1, dtype=np.uint8)[0]
            out[idx] = scores > 0
    return out


def search(state: State, q: str, n: int = 10, min_score: float = 0.0,
           embed: Callable[[str], np.ndarray] | None = None) -> dict:
    out: dict = {"query": q, "semantic": False, "results": []}
    pq = parse_query(q)
    if len(pq.norm) < 2:
        return out
    terms = pq.terms
    nf = len(state.files)

    # Meaning
    fz = np.zeros(nf, np.float32)
    z: np.ndarray | None = None
    if embed is not None:
        try:
            qv = embed(pq.text[:300])
            out["semantic"] = True
        except (EmbedError, EmbedServerDown):
            qv = None
        if qv is not None and len(state.V):
            s = state.V @ qv
            z = np.empty_like(s)
            for g in (VISUAL, TEXT):
                m = state.vec_group == g
                if not m.any():
                    continue
                sm = s[m]
                sub = sm if len(sm) >= GROUP_MIN_ROWS else s
                z[m] = (sm - sub.mean()) / max(float(sub.std()), STD_FLOOR)
            fz[state.vec_files] = np.maximum.reduceat(z, state.vec_starts)

    # Words
    fc = np.zeros(nf, np.float32)
    cov: np.ndarray | None = None
    if state.corpus:
        cov = np.zeros(len(state.corpus), np.float32)
        for term in terms:
            thr = SHORT_TERM_SCORE if len(term) <= SHORT_TERM_LEN else LONG_TERM_SCORE
            cov += _fuzzy_matches(state, term, thr)
        cov /= len(terms)
        if len(pq.norm) >= PHRASE_MIN_LEN:
            cov[_fuzzy_matches(state, pq.norm, PHRASE_SCORE)] = 1.0
        fc[state.corpus_files] = np.maximum.reduceat(cov, state.corpus_starts)

    # Candidates and score
    score = np.maximum(fz, 0) + WORD_WEIGHT * fc**1.5
    eligible = (fz >= Z_GATE) | (fc >= COVERAGE_GATE)
    if pq.kinds:
        kind_ids = [_KIND_ID[k] for k in pq.kinds if k in _KIND_ID]
        eligible &= np.isin(state.file_kind, kind_ids)
    cand = np.flatnonzero(eligible)
    if not len(cand):
        return out
    cand = cand[np.argsort(-score[cand], kind="stable")]
    top = float(score[cand[0]])
    results = []
    for i in cand:
        sc = float(score[i])
        if sc < TOP_CUT * top or sc < min_score or len(results) >= n:
            break
        path, kind, thumb = state.files[i]
        words, meaning = fc[i] >= COVERAGE_GATE, fz[i] >= Z_GATE
        if words:
            assert cov is not None
            p = state.corpus_file_pos[i]
            s0, s1 = state.corpus_starts[p], state.corpus_ends[p]
            seg = int(state.corpus_seg[s0 + int(np.argmax(cov[s0:s1]))])
        else:
            assert z is not None
            p = state.vec_file_pos[i]
            s0, s1 = state.vec_starts[p], state.vec_ends[p]
            seg = int(state.vec_seg[s0 + int(np.argmax(z[s0:s1]))])
        results.append({
            "path": path,
            "name": os.path.basename(path),
            "kind": kind,
            "score": round(sc, 3),
            "match": "both" if words and meaning else "words" if words else "meaning",
            "snippet": " ".join(state.seg_text[seg].split())[:100] if words else "",
            "t": state.seg_t[seg],
            "page": state.seg_page[seg],
            "thumb": thumb,
        })
    out["results"] = results
    return out
