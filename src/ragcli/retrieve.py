"""Pure helpers for hybrid retrieval and yes/no rerank scores.

``fts_match_query`` keeps protected spans only and never reads the corpus.
``rrf_merge`` ranks by list position. ``yes_no_score`` only reads token
logprobs. None of these functions calls Ollama.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

RRF_K = 60
RERANK_MODEL = "qwen3-reranker:0.6b"
RERANK_TOP_LOGPROBS = 20
RERANK_OVERHEAD = 200
RERANK_PROBE_MARGIN = 0.2
RERANK_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)
# The installed tag must render this Go template. The client does not append it.
RERANK_TEMPLATE = (
    "{{- if .System }}<|im_start|>system\n"
    "{{ .System }}<|im_end|>\n"
    "{{ end }}\n"
    "{{- range .Messages }}\n"
    '{{- if eq .Role "user" }}<|im_start|>user\n'
    "{{ .Content }}<|im_end|>\n"
    '{{ else if eq .Role "assistant" }}<|im_start|>assistant\n'
    "{{ .Content }}<|im_end|>\n"
    "{{ end }}\n"
    "{{- end }}<|im_start|>assistant\n"
    "{{if not .Think}}<think>\n"
    "\n"
    "</think>\n"
    "\n"
    "{{ end }}\n"
)
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


def yes_no_score(top: Sequence[tuple[str, float]]) -> float | None:
    """Return P(yes) from the two-token softmax, or ``None`` if either is missing.

    Token text must be exactly ``yes`` or ``no``. Case and surrounding spaces
    are different tokens. The sampled ``content`` is not a score.
    """
    yes = next((logp for token, logp in top if token == "yes"), None)
    no = next((logp for token, logp in top if token == "no"), None)
    if yes is None or no is None:
        return None
    pivot = yes if yes > no else no
    ey = math.exp(yes - pivot)
    en = math.exp(no - pivot)
    return ey / (ey + en)


def rerank_user(instruction: str, query: str, document: str) -> str:
    """User message for one yes/no score. The caller does not add a think block."""
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def template_accepts_rerank(template: str) -> bool:
    """True when the tag template closes thinking before the first token.

    A spaced ``{{ if not .Think }}`` still counts. The empty think block must
    contain the substring ``<think>`` plus a blank line plus ``</think>``.
    """
    if not isinstance(template, str):
        return False
    return "not .Think" in template and "<think>\n\n</think>" in template


def _probe_text(points: tuple[int, ...]) -> str:
    return "".join(chr(point) for point in points)


_PROBE_ZH_QUERY = _probe_text((0x6CD5, 0x570B, 0x9996, 0x90FD))
_PROBE_ZH_RELEVANT = _probe_text(
    (0x6CD5, 0x570B, 0x7684, 0x9996, 0x90FD, 0x662F, 0x5DF4, 0x9ECE, 0x3002)
)
_PROBE_ZH_IRRELEVANT = _probe_text(
    (0x9019, 0x4EFD, 0x83DC, 0x55AE, 0x6709, 0x8336, 0x548C, 0x98EF, 0x3002)
)
RERANK_PROBE_PAIRS = (
    (
        "capital of France",
        "Paris is the capital of France.",
        "The menu lists tea and rice.",
    ),
    (_PROBE_ZH_QUERY, _PROBE_ZH_RELEVANT, _PROBE_ZH_IRRELEVANT),
)


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
