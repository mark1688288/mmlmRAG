"""Pure helpers for hybrid retrieval.

``fts_match_query`` keeps protected spans only and never reads the corpus.
``rrf_merge`` ranks by list position. Neither function calls Ollama.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

RRF_K = 60
_MAX_SPAN = 64
_MAX_SPANS = 8
_MIN_SPAN = 3

_CLAUSE = re.compile(chr(0x7B2C) + r"[0-9]{1,6}" + chr(0x689D))
_AMOUNT_GROUPED = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_AMOUNT_DECIMAL = re.compile(r"\d+\.\d+")
_MODEL = re.compile(r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)+")
_ASCII = re.compile(r"[A-Za-z0-9]{3,}")

_PRIORITY_CLAUSE = 0
_PRIORITY_AMOUNT = 1
_PRIORITY_MODEL = 2
_PRIORITY_ASCII = 3


@dataclass(frozen=True, slots=True)
class MergedHit:
    """One direct hit after reciprocal rank fusion.

    ``distance`` is missing when the chunk was not in the vector list.
    ``origin`` is ``vector``, ``fts``, or ``both``.
    """

    chunk_id: str
    distance: float | None
    origin: str


def fts_match_query(question: str) -> str | None:
    """Return a quoted ``OR`` query, or ``None`` when nothing is protected.

    Spans are clause numbers, amounts, model ids, and ASCII runs of at
    least three characters. Longer spans win overlaps. The result never
    passes the raw question to ``MATCH``.
    """
    if not isinstance(question, str) or question == "":
        return None
    phrases: list[str] = []
    seen: set[str] = set()
    for text in _protected_spans(question):
        if text in seen:
            continue
        seen.add(text)
        phrases.append(_fts_phrase(text))
        if len(phrases) == _MAX_SPANS:
            break
    if not phrases:
        return None
    return " OR ".join(phrases)


def rrf_score(ranks: Sequence[int], k: int = RRF_K) -> float:
    """Sum reciprocal ranks. ``k`` is the fusion constant, not ``--k``."""
    return sum(1.0 / (k + rank) for rank in ranks)


def rrf_merge(
    vector: Sequence[tuple[str, float]],
    fts: Sequence[str],
    limit: int,
) -> list[MergedHit]:
    """Fuse two ranked lists and keep at most ``limit`` hits.

    Rank is the 1-based list position. The caller orders ``vector`` by
    ascending cosine distance and ``fts`` by ascending bm25. A higher
    fused score comes first. On a tie, a hit with a vector distance comes
    before a full-text-only hit, then the smaller distance, then
    ``chunk_id``.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    vector_rank: dict[str, int] = {}
    distance_for: dict[str, float] = {}
    for index, (chunk_id, distance) in enumerate(vector, start=1):
        if chunk_id in vector_rank:
            continue
        vector_rank[chunk_id] = index
        distance_for[chunk_id] = float(distance)
    fts_rank: dict[str, int] = {}
    for index, chunk_id in enumerate(fts, start=1):
        if chunk_id in fts_rank:
            continue
        fts_rank[chunk_id] = index
    ranked: list[tuple[float, float | None, str, str]] = []
    for chunk_id in set(vector_rank) | set(fts_rank):
        ranks: list[int] = []
        in_vector = chunk_id in vector_rank
        in_fts = chunk_id in fts_rank
        if in_vector:
            ranks.append(vector_rank[chunk_id])
        if in_fts:
            ranks.append(fts_rank[chunk_id])
        if in_vector and in_fts:
            origin = "both"
        elif in_vector:
            origin = "vector"
        else:
            origin = "fts"
        distance = distance_for.get(chunk_id)
        ranked.append((rrf_score(ranks), distance, chunk_id, origin))
    ranked.sort(
        key=lambda item: (
            -item[0],
            item[1] is None,
            0.0 if item[1] is None else item[1],
            item[2],
        )
    )
    return [
        MergedHit(chunk_id=chunk_id, distance=distance, origin=origin)
        for _score, distance, chunk_id, origin in ranked[:limit]
    ]


def _protected_spans(question: str) -> list[str]:
    found: list[tuple[int, int, int, str]] = []
    patterns = (
        (_CLAUSE, _PRIORITY_CLAUSE),
        (_AMOUNT_GROUPED, _PRIORITY_AMOUNT),
        (_AMOUNT_DECIMAL, _PRIORITY_AMOUNT),
        (_MODEL, _PRIORITY_MODEL),
        (_ASCII, _PRIORITY_ASCII),
    )
    for pattern, priority in patterns:
        for match in pattern.finditer(question):
            found.append((match.start(), match.end(), priority, match.group()))
    found.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[2]))
    kept: list[tuple[int, int, str]] = []
    for start, end, _priority, text in found:
        if end - start < _MIN_SPAN:
            continue
        if any(start < kept_end and end > kept_start for kept_start, kept_end, _kept in kept):
            continue
        clipped = text[:_MAX_SPAN]
        if len(clipped) < _MIN_SPAN:
            continue
        kept.append((start, end, clipped))
    kept.sort(key=lambda item: item[0])
    return [text for _start, _end, text in kept]


def _fts_phrase(text: str) -> str:
    return '"' + text.replace('"', '""') + '"'
