"""Tests for search state loading, scoring equivalence, and fallback behavior."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from rapidfuzz import fuzz, process

from mediadig import config, db, search
from mediadig.llama import EmbedError, EmbedServerDown
from mediadig.textnorm import parse_query


def _reference_search(state: search.State, q: str, n: int, min_score: float, embed) -> dict:
    """Original search implementation kept as an oracle for parity checks."""
    out: dict = {"query": q, "semantic": False, "results": []}
    pq = parse_query(q)
    if len(pq.norm) < 2:
        return out
    terms = pq.terms
    nf = len(state.files)

    def _old_fuzzy(needle: str, threshold: int) -> np.ndarray:
        res = np.zeros(len(state.corpus), np.float32)
        long_idx = np.flatnonzero(state.corpus_len >= len(needle))
        short_idx = np.flatnonzero(state.corpus_len < len(needle))
        for idx, scorer in ((long_idx, fuzz.partial_ratio), (short_idx, fuzz.ratio)):
            if len(idx):
                texts = state.corpus if len(idx) == len(state.corpus) else [state.corpus[i] for i in idx]
                res[idx] = process.cdist([needle], texts, scorer=scorer, score_cutoff=threshold,
                                         workers=-1, dtype=np.uint8)[0]
        return res

    def _old_best(values: np.ndarray, owner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        best = np.full(nf, -np.inf, np.float32)
        row = np.full(nf, -1, np.int64)
        if len(values):
            order = np.argsort(-values, kind="stable")
            files, first = np.unique(owner[order], return_index=True)
            best[files] = values[order[first]]
            row[files] = order[first]
        return best, row

    fz = np.zeros(nf, np.float32)
    fz_seg = np.full(nf, -1, np.int64)
    if embed is not None:
        try:
            qv = embed(pq.text[:300])
            out["semantic"] = True
        except EmbedError:
            qv = None
        if qv is not None and len(state.V):
            s = state.V @ qv
            z = np.empty_like(s)
            for g in (search.VISUAL, search.TEXT):
                m = state.vec_group == g
                if not m.any():
                    continue
                sub = s[m] if m.sum() >= search.GROUP_MIN_ROWS else s
                z[m] = (s[m] - sub.mean()) / max(float(sub.std()), search.STD_FLOOR)
            best, row = _old_best(z, state.vec_file)
            has = row >= 0
            fz[has] = best[has]
            fz_seg[has] = state.vec_seg[row[has]]

    fc = np.zeros(nf, np.float32)
    fc_seg = np.full(nf, -1, np.int64)
    if state.corpus:
        cov = np.zeros(len(state.corpus), np.float32)
        for term in terms:
            thr = search.SHORT_TERM_SCORE if len(term) <= search.SHORT_TERM_LEN else search.LONG_TERM_SCORE
            cov += _old_fuzzy(term, thr) >= thr
        cov /= len(terms)
        if len(pq.norm) >= search.PHRASE_MIN_LEN:
            cov[_old_fuzzy(pq.norm, search.PHRASE_SCORE) >= search.PHRASE_SCORE] = 1.0
        best, row = _old_best(cov, state.corpus_file)
        has = row >= 0
        fc[has] = best[has]
        fc_seg[has] = state.corpus_seg[row[has]]

    score = np.maximum(fz, 0) + search.WORD_WEIGHT * fc**1.5
    eligible = (fz >= search.Z_GATE) | (fc >= search.COVERAGE_GATE)
    if pq.kinds:
        file_kinds = np.asarray([f[1] for f in state.files], object)
        eligible &= np.isin(file_kinds, list(pq.kinds))
    cand = np.flatnonzero(eligible)
    if not len(cand):
        return out
    cand = cand[np.argsort(-score[cand], kind="stable")]
    top = float(score[cand[0]])
    results = []
    for i in cand:
        sc = float(score[i])
        if sc < search.TOP_CUT * top or sc < min_score or len(results) >= n:
            break
        path, kind, thumb = state.files[i]
        words, meaning = fc[i] >= search.COVERAGE_GATE, fz[i] >= search.Z_GATE
        seg = fc_seg[i] if words else fz_seg[i]
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


def _build_synthetic_db(db_path: Path, n_files: int = 80) -> tuple[np.ndarray, search.State]:
    con = db.connect(db_path)
    rng = np.random.default_rng(123)
    vocab = [
        "cat", "dog", "meme", "funny", "kocka", "pes", "programming",
        "restaurant", "invoice", "screenshot", "coffee", "bicycle", "mountain",
    ]
    all_vecs = []
    with db.transaction(con):
        for fi in range(n_files):
            kind = config.KIND_ORDER[fi % len(config.KIND_ORDER)]
            fid = con.execute(
                "INSERT INTO files(path, kind, size, mtime_ns, dev, ino, status, error, thumb, indexed_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (f"/media/item_{fi:03d}.{kind}", kind, 1000 + fi, 1000, 1, fi, "ok", None, None, 1000),
            ).lastrowid
            n_segs = 1 + (fi % 4)
            for k in range(n_segs):
                v = rng.standard_normal(config.EMBED_DIM).astype(np.float32)
                v /= np.linalg.norm(v)
                all_vecs.append(v)
                skind = "chunk" if kind in ("pdf", "docx", "txt") else ("frame" if kind in ("gif", "video") else "image")
                # Mix short, medium, duplicate, and Czech texts to exercise every branch
                if k == 0 and fi % 7 == 0:
                    txt = vocab[fi % len(vocab)][:3]
                elif k == 1 and fi % 5 == 0:
                    txt = "Příliš žluťoučký kůň programming invoice"
                else:
                    w1 = vocab[(fi + k) % len(vocab)]
                    w2 = vocab[(fi * 3 + k) % len(vocab)]
                    txt = f"{w1} {w2} sample caption {fi}"
                vec_blob = None if (fi % 19 == 0 and k == 0) else v.astype("<f2").tobytes()
                t_val = float(k * 2) if skind == "frame" else None
                p_val = k + 1 if skind == "chunk" else None
                con.execute(
                    "INSERT INTO segments(file_id, kind, t, page, text, vec) VALUES(?,?,?,?,?,?)",
                    (fid, skind, t_val, p_val, txt, vec_blob),
                )
    con.close()
    return np.stack(all_vecs), search.load_state(db_path)


def test_search_exact_parity_with_reference(tmp_path: Path) -> None:
    vecs, st = _build_synthetic_db(tmp_path / "parity.sqlite", n_files=120)
    queries = [
        "cat",
        "cat dog",
        "programming invoice",
        "programing invoce",  # fuzzy typos
        "prilis zlutoucky kun",
        "vid cat funny",
        "img screenshot coffee",
        "pdf invoice",
        "zz",  # no word matches, meaning only
        "a",   # too short
    ]
    for qi, q in enumerate(queries):
        qv = vecs[(qi * 11) % len(vecs)]
        embed = lambda _text, v=qv: v
        got = search.search(st, q, n=15, min_score=0.0, embed=embed)
        ref = _reference_search(st, q, n=15, min_score=0.0, embed=embed)
        assert got == ref, f"Mismatch on query {q!r}"


def test_search_word_fallback_on_embed_error(tmp_path: Path) -> None:
    _, st = _build_synthetic_db(tmp_path / "fb.sqlite", n_files=30)

    def broken_embed(q: str) -> np.ndarray:
        raise EmbedError("down")

    res = search.search(st, "programming invoice", n=10, min_score=0.0, embed=broken_embed)
    assert res["semantic"] is False
    assert len(res["results"]) > 0
    assert all(r["match"] == "words" for r in res["results"])

    def down_embed(q: str) -> np.ndarray:
        raise EmbedServerDown("unreachable")

    res2 = search.search(st, "programming invoice", n=10, min_score=0.0, embed=down_embed)
    assert res2 == res


def test_search_edge_states() -> None:
    # Empty state returns empty results cleanly
    empty = search.State()
    qv = np.zeros(config.EMBED_DIM, np.float32)
    qv[0] = 1.0
    res = search.search(empty, "hello world", n=10, min_score=0.0, embed=lambda _q: qv)
    assert res == {"query": "hello world", "semantic": True, "results": []}

    # Vector-only file + text-only file
    V = np.zeros((2, config.EMBED_DIM), np.float32)
    V[0, 0] = 1.0
    V[1, 1] = 1.0
    st = search.State(
        files=[("/a.jpg", "image", None), ("/b.jpg", "image", None), ("/c.txt", "txt", None)],
        V=V,
        vec_seg=np.asarray([0, 1], np.int64),
        vec_file=np.asarray([0, 1], np.int64),
        vec_group=np.asarray([search.VISUAL, search.VISUAL], np.int8),
        seg_t=[None, None, None],
        seg_page=[None, None, None],
        seg_text=["", "", "unique keyword match"],
        corpus=["unique keyword match"],
        corpus_seg=np.asarray([2], np.int64),
        corpus_file=np.asarray([2], np.int64),
    )
    r_words = search.search(st, "unique keyword", n=5, min_score=0.0, embed=None)
    assert [r["path"] for r in r_words["results"]] == ["/c.txt"]

    # Non-grouped owner array raises explicitly instead of silently mis-scoring
    with pytest.raises(ValueError, match="grouped contiguously"):
        search.State(
            files=[("/a.jpg", "image", None), ("/b.jpg", "image", None)],
            V=np.zeros((3, config.EMBED_DIM), np.float32),
            vec_seg=np.asarray([0, 1, 2], np.int64),
            vec_file=np.asarray([0, 1, 0], np.int64),
            vec_group=np.asarray([0, 0, 0], np.int8),
        )
