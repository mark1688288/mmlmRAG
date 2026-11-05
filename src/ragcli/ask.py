"""Answer one question from the active collection.

The query string uses the Qwen3-Embedding instruction. Document text does
not. ``Query:`` has no space before the question. Hits whose best cosine
distance is above the configured gate never load a generator, do not query
FTS, and do not expand neighboring chunks. When hybrid retrieval is on,
vector hits and FTS hits are merged with RRF. Same-document chunks at
ordinal ±1 then follow those direct hits. Optional reranking is off unless
requested. A skip leaves the same direct hits as reranking off. Optional
verification is off unless requested. It asks the loaded generator for an
exact PASS or FAIL and does not retrieve again. A chunk that does not fit
is left out whole, and if the first chunk does not fit the prompt is not
sent.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ragcli.chunking import estimate_tokens
from ragcli.config import Config
from ragcli.errors import (
    EXIT_DISK,
    EXIT_MISMATCH,
    EXIT_MODEL_MISSING,
    EXIT_NO_COLLECTION,
    EXIT_OLLAMA_UNREACHABLE,
    EXIT_QUESTION,
    EXIT_RESOURCES,
    EXIT_UNCLASSIFIED,
    EXIT_USAGE,
    RagError,
)
from ragcli.i18n import t
from ragcli.models_lock import ollama_lock
from ragcli.ollama import (
    KEEP_ALIVE_BATCH,
    KEEP_ALIVE_DONE,
    ChatMessage,
    OllamaClient,
    available_bytes,
    choose_generator,
    valid_model_name,
)
from ragcli.prompt import (
    Excerpt,
    format_excerpt,
    format_user_message,
    is_refusal,
    refusal_sentence,
    system_prompt,
    verify_system_prompt,
    verify_user_message,
)
from ragcli.retrieve import (
    RERANK_MODEL,
    RERANK_OVERHEAD,
    RERANK_PROBE_MARGIN,
    RERANK_PROBE_PAIRS,
    RERANK_SYSTEM,
    RERANK_TOP_LOGPROBS,
    MergedHit,
    fts_match_query,
    parse_verdict,
    rerank_user,
    rrf_merge,
    template_accepts_rerank,
    yes_no_score,
)
from ragcli.store.chroma import ChromaIndex, QueryHit
from ragcli.store.sqlite import Chunk, Run, SqliteStore

logger = logging.getLogger("ragcli.ask")

QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
SAFETY_TOKENS = 128

_KNOWN_NATIVE = {
    "qwen3-embedding:0.6b": 1024,
    "qwen3-embedding:4b": 2560,
    "qwen3-embedding:8b": 4096,
}
_OLLAMA_CODES = frozenset(
    {
        EXIT_OLLAMA_UNREACHABLE,
        EXIT_MODEL_MISSING,
        EXIT_RESOURCES,
        EXIT_UNCLASSIFIED,
    }
)
_PRIVATE_FILE = 0o600


@dataclass
class _Outcome:
    code: int = 0
    chunk_count: int = 0
    model_name: str | None = None
    refused_count: int = 0
    ollama_error_count: int = 0
    ask_latency_ms: int | None = None


def query_text(question: str) -> str:
    """Return the embedding input for a question.

    The instruction is the English line from the Qwen3-Embedding model card.
    There is no space between ``Query:`` and the question.
    """
    if not isinstance(question, str) or question == "":
        raise RagError(EXIT_USAGE, t("ask.question_missing"))
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery:{question}"


def document_embed_text(text: str) -> str:
    """Return chunk text unchanged. Documents do not take the query prefix."""
    if not isinstance(text, str):
        raise RagError(EXIT_UNCLASSIFIED, t("ask.chunk_text_type"))
    return text


def require_question_limit(question: str, limit: int) -> None:
    """Reject a question over ``limit`` estimated tokens. Do not shorten it."""
    if estimate_tokens(question) > limit:
        raise RagError(EXIT_QUESTION, t("ask.question_too_long", limit=limit))


def retrieval_budget(
    *,
    num_ctx: int,
    system: str,
    question: str,
    num_predict: int,
) -> int:
    """Tokens left for excerpts after the system prompt, question, and margin.

    128 is the fixed safety margin. ``num_predict`` reserves the answer.
    """
    return (
        num_ctx
        - estimate_tokens(system)
        - estimate_tokens(question)
        - num_predict
        - SAFETY_TOKENS
    )


def pack_excerpts(excerpts: Sequence[Excerpt], budget: int) -> list[Excerpt]:
    """Keep the caller order. Do not sort. Do not skip a later shorter chunk.

    The first item that does not fit stops the pack. If that item is also
    the first item, exit 9 and do not build a partial prompt.
    """
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise RagError(EXIT_UNCLASSIFIED, t("ask.budget_type"))
    packed: list[Excerpt] = []
    used = 0
    for item in excerpts:
        cost = estimate_tokens(format_excerpt(len(packed) + 1, item))
        if used + cost > budget:
            if not packed:
                raise RagError(EXIT_QUESTION, t("ask.excerpt_unfit"))
            break
        packed.append(item)
        used += cost
    return packed


def run_ask(
    config: Config,
    question: str,
    *,
    k: int | None = None,
    model: str | None = None,
    allow_fallback: bool = True,
    show_chunks: bool = False,
    as_json: bool = False,
    hybrid: bool | None = None,
    neighbors: bool | None = None,
    rerank: bool | None = None,
    verify: bool | None = None,
    lock_timeout: float | None = None,
    client: OllamaClient | None = None,
    available: int | None = None,
    clock: Callable[[], str] | None = None,
    client_factory: Callable | None = None,
    free_bytes: Callable[[], int] | None = None,
) -> int:
    """Ask one question. Returns a process exit code. Does not embed the corpus."""
    cleaned = _clean_question(question)
    if model is not None and not valid_model_name(model):
        raise RagError(EXIT_USAGE, t("model.name_invalid"))
    limit = _resolve_k(config, k)
    use_hybrid = _resolve_hybrid(config, hybrid)
    use_neighbors = _resolve_neighbors(config, neighbors)
    use_rerank = _resolve_rerank(config, rerank)
    use_verify = _resolve_verify(config, verify)
    _require_disk(config, free_bytes)
    with _session(config, clock, client_factory=client_factory) as (store, index, outcome):
        _execute(
            config,
            store,
            index,
            outcome,
            cleaned,
            k=limit,
            model=model,
            allow_fallback=allow_fallback,
            show_chunks=show_chunks,
            as_json=as_json,
            hybrid=use_hybrid,
            neighbors=use_neighbors,
            rerank=use_rerank,
            verify=use_verify,
            lock_timeout=lock_timeout,
            client=client,
            available=available,
        )
        return outcome.code


def _execute(
    config: Config,
    store: SqliteStore,
    index: ChromaIndex,
    outcome: _Outcome,
    question: str,
    *,
    k: int,
    model: str | None,
    allow_fallback: bool,
    show_chunks: bool,
    as_json: bool,
    hybrid: bool,
    neighbors: bool,
    rerank: bool,
    verify: bool,
    lock_timeout: float | None,
    client: OllamaClient | None,
    available: int | None,
) -> None:
    if store.has_unfinished_chunks():
        raise RagError(EXIT_MISMATCH, t("ask.unfinished_chunks"))
    active = store.get_active()
    if active is None:
        raise RagError(EXIT_NO_COLLECTION, t("ask.no_active_collection"))
    require_question_limit(question, config.max_question_tokens)
    info = index.describe(active.collection)
    if (
        info.embed_model != active.embed_model
        or info.dimensions != active.dimensions
        or info.embed_num_ctx != active.embed_num_ctx
    ):
        raise RagError(EXIT_MISMATCH, t("ask.metadata_mismatch"))
    logger.info("%s", t("log.query_collection", name=active.collection))
    if not active.embed_model.startswith("qwen3-embedding"):
        print(t("ask.prefix_warning"), file=sys.stderr)
    ollama = client if client is not None else OllamaClient(config.ollama_base_url)
    with ollama_lock(config.data_dir, timeout=lock_timeout):
        width = k
        if rerank and _rerank_can_score(config, ollama, outcome, available):
            width = config.rerank_candidates
        vector = _embed_query(
            ollama,
            outcome,
            active.embed_model,
            query_text(question),
            active.embed_num_ctx,
        )
        native = _expected_native(config, active.embed_model)
        if native is not None and len(vector) != native:
            raise RagError(EXIT_MISMATCH, t("ask.native_width"))
        hits = index.query(
            vector,
            model=active.embed_model,
            dimensions=active.dimensions,
            embed_num_ctx=active.embed_num_ctx,
            n_results=width,
        )
        ordered = sorted(hits, key=lambda hit: hit.distance)
        if not ordered or ordered[0].distance > config.max_cosine_distance:
            if show_chunks:
                _print_hits([_excerpt(hit) for hit in ordered])
            logger.info("%s", t("log.no_close_hit"))
            _emit(
                refusal_sentence(),
                refused=True,
                model=None,
                sources=[],
                as_json=as_json,
            )
            outcome.refused_count = 1
            return
        fts_ids = _fts_ids(store, question, width) if hybrid else []
        if width == config.rerank_candidates and rerank:
            selected = _apply_rerank(
                config,
                store,
                ollama,
                outcome,
                question,
                ordered,
                fts_ids,
                k=k,
                hybrid=hybrid,
            )
        else:
            selected = _select_hits(
                store, ordered, fts_ids, limit=k, hybrid=hybrid
            )
        if neighbors:
            selected = _append_neighbors(store, selected)
        if show_chunks:
            _print_hits(selected)
        prompt = system_prompt()
        budget = retrieval_budget(
            num_ctx=config.generate_num_ctx,
            system=prompt,
            question=question,
            num_predict=config.num_predict,
        )
        packed = pack_excerpts(selected, budget)
        content = _generate(
            config,
            ollama,
            outcome,
            format_user_message(question, packed),
            system=prompt,
            preferred=config.generate_model if model is None else model,
            allow_fallback=allow_fallback,
            available=available,
            hold=verify,
        )
        outcome.chunk_count = len(packed)
        if verify:
            answer, refused = _apply_verdict(
                config,
                ollama,
                outcome,
                question,
                packed,
                content,
            )
        else:
            refused = is_refusal(content)
            answer = refusal_sentence() if refused else content
            if refused:
                logger.info("%s", t("log.model_refused"))
                outcome.refused_count = 1
    _emit(answer, refused=refused, model=outcome.model_name, sources=packed, as_json=as_json)


def _embed_query(
    client: OllamaClient,
    outcome: _Outcome,
    model: str,
    text: str,
    num_ctx: int,
) -> list[float]:
    vector: list[float] | None = None
    body_error: BaseException | None = None
    try:
        vectors = client.embed(model, [text], num_ctx=num_ctx, keep_alive=KEEP_ALIVE_DONE)
        if len(vectors) != 1:
            raise RagError(EXIT_UNCLASSIFIED, t("ask.embed_count"))
        vector = vectors[0]
    except RagError as exc:
        body_error = exc
        _note_ollama(outcome, exc)
        raise
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        _release_if_loaded(client, model, body_error, outcome)
    if vector is None:
        raise RagError(EXIT_UNCLASSIFIED, t("ask.embed_count"))
    return vector


def _generate(
    config: Config,
    client: OllamaClient,
    outcome: _Outcome,
    user: str,
    *,
    system: str,
    preferred: str,
    allow_fallback: bool,
    available: int | None,
    hold: bool = False,
) -> str:
    try:
        installed = client.tags()
    except RagError as exc:
        _note_ollama(outcome, exc)
        raise
    free = available_bytes() if available is None else available
    chosen = choose_generator(
        preferred=preferred,
        fallback=config.generate_fallback_model,
        tags=installed,
        available=free,
        nine_b_min_bytes=config.nine_b_min_bytes,
        four_b_min_bytes=config.four_b_min_bytes,
        allow_fallback=allow_fallback,
    )
    outcome.model_name = chosen
    logger.info("%s", t("log.generator", name=chosen))
    result = None
    body_error: BaseException | None = None
    try:
        result = client.chat(
            chosen,
            (
                ChatMessage("system", system),
                ChatMessage("user", user),
            ),
            num_ctx=config.generate_num_ctx,
            num_predict=config.num_predict,
            temperature=config.temperature,
            keep_alive=KEEP_ALIVE_BATCH if hold else KEEP_ALIVE_DONE,
        )
    except RagError as exc:
        body_error = exc
        _note_ollama(outcome, exc)
        raise
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        if not hold or body_error is not None:
            _release_if_loaded(client, chosen, body_error, outcome)
    if result is None or result.thinking_only or result.content.strip() == "":
        if hold:
            _release_if_loaded(client, chosen, None, outcome)
        raise RagError(EXIT_UNCLASSIFIED, t("ask.empty_generation"))
    return result.content


def _apply_verdict(
    config: Config,
    client: OllamaClient,
    outcome: _Outcome,
    question: str,
    packed: Sequence[Excerpt],
    draft: str,
) -> tuple[str, bool]:
    """Check one draft with the generator that is already loaded.

    The model stays loaded from the first chat. This function unloads it
    before returning. An exact FAIL becomes the refusal sentence. Any other
    text is exit 1. A prompt that does not fit leaves the draft unchanged.
    A model refusal is not checked again. This does not retrieve.
    """
    model = outcome.model_name
    if not isinstance(model, str) or model == "":
        raise RagError(EXIT_UNCLASSIFIED, t("ask.empty_generation"))
    system = verify_system_prompt()
    user = verify_user_message(question, packed, draft)
    body_error: BaseException | None = None
    verdict_result = None
    if is_refusal(draft):
        mode = "refusal"
    elif not _verify_fits(config, system, user):
        mode = "unfit"
    else:
        mode = "chat"
    try:
        if mode == "chat":
            try:
                verdict_result = client.chat(
                    model,
                    (
                        ChatMessage("system", system),
                        ChatMessage("user", user),
                    ),
                    num_ctx=config.generate_num_ctx,
                    num_predict=config.verify_num_predict,
                    temperature=0,
                    keep_alive=KEEP_ALIVE_DONE,
                )
            except RagError as exc:
                body_error = exc
                _note_ollama(outcome, exc)
                raise
            except BaseException as exc:
                body_error = exc
                raise
    finally:
        _release_if_loaded(client, model, body_error, outcome)
    if mode == "refusal":
        logger.info("%s", t("log.model_refused"))
        outcome.refused_count = 1
        return refusal_sentence(), True
    if mode == "unfit":
        logger.warning("%s", t("ask.verify_unfit"))
        return draft, False
    if (
        verdict_result is None
        or verdict_result.thinking_only
        or verdict_result.content.strip() == ""
    ):
        raise RagError(EXIT_UNCLASSIFIED, t("ask.empty_generation"))
    verdict = parse_verdict(verdict_result.content)
    if verdict == "pass":
        logger.info("%s", t("log.verify_pass"))
        return draft, False
    if verdict == "fail":
        logger.warning("%s", t("log.verify_fail"))
        outcome.refused_count = 1
        return refusal_sentence(), True
    raise RagError(EXIT_UNCLASSIFIED, t("ask.verify_invalid"))


def _verify_fits(config: Config, system: str, user: str) -> bool:
    """True when the second prompt and its short answer fit in ``num_ctx``.

    The sum is both prompts, ``verify_num_predict``, and the fixed 128-token
    margin. Equality still fits. The excerpts and the draft are not shortened.
    """
    cost = (
        estimate_tokens(system)
        + estimate_tokens(user)
        + config.verify_num_predict
        + SAFETY_TOKENS
    )
    return cost <= config.generate_num_ctx


def _release_if_loaded(
    client: OllamaClient,
    model: str,
    body_error: BaseException | None,
    outcome: _Outcome,
) -> None:
    try:
        loaded = client.ps()
    except RagError as exc:
        _note_ollama(outcome, exc)
        if body_error is None:
            raise
        return
    names = [item.name for item in loaded]
    if model not in names:
        if names and body_error is None:
            raise RagError(EXIT_RESOURCES, t("ask.models_still_loaded"))
        return
    try:
        client.release_stage(model)
    except RagError as exc:
        logger.error("%s", t("log.unload_failed", model=model))
        _note_ollama(outcome, exc)
        if body_error is None:
            raise


def _note_ollama(outcome: _Outcome, exc: RagError) -> None:
    if exc.code in _OLLAMA_CODES:
        outcome.ollama_error_count += 1


def _fts_ids(store: SqliteStore, question: str, limit: int) -> list[str]:
    """Ranked full-text ids, or an empty list when the question has no span."""
    match = fts_match_query(question)
    if match is None:
        return []
    return [chunk_id for chunk_id, _score in store.search_fts(match, limit)]


def _select_hits(
    store: SqliteStore,
    ordered: Sequence[QueryHit],
    fts_ids: Sequence[str],
    *,
    limit: int,
    hybrid: bool,
    announce: bool = True,
) -> list[Excerpt]:
    """Return direct hits in pack order. A closed gate never reaches here.

    ``ordered`` is nearest-first. ``fts_ids`` is best-bm25-first. ``limit``
    keeps the prefix of each list before fusion, so a wide retrieval can be
    reduced to the same hits reranking-off would have produced.
    """
    vector_rows = list(ordered[:limit])
    if not hybrid:
        return [_excerpt(hit) for hit in vector_rows]
    chosen_fts = list(fts_ids[:limit])
    merged = rrf_merge(
        [(hit.chunk_id, hit.distance) for hit in vector_rows],
        chosen_fts,
        limit,
    )
    by_id = {hit.chunk_id: hit for hit in vector_rows}
    selected: list[Excerpt] = []
    for item in merged:
        hit = by_id.get(item.chunk_id)
        if hit is not None:
            selected.append(_excerpt(hit, origin=item.origin))
            continue
        loaded = _fts_excerpt(store, item)
        if loaded is not None:
            selected.append(loaded)
    if announce:
        logger.info(
            "%s",
            t(
                "log.hybrid",
                vector_count=len(vector_rows),
                fts_count=len(chosen_fts),
                merged_count=len(selected),
            ),
        )
    return selected


def _rerank_can_score(
    config: Config,
    client: OllamaClient,
    outcome: _Outcome,
    available: int | None,
) -> bool:
    """True only when the wide retrieval is allowed to load the reranker.

    A skip is logged and does not load the model. Connection errors still
    fail the question. A missing tag, a rejected template, a failed memory
    gate, or a context that cannot hold one pair all return false.
    """
    if (
        config.max_question_tokens + config.chunk_max_tokens + RERANK_OVERHEAD
        > config.rerank_num_ctx
    ):
        logger.warning("%s", t("log.rerank_skipped"))
        return False
    model = config.rerank_model
    try:
        installed = client.tags()
    except RagError as exc:
        _note_ollama(outcome, exc)
        raise
    if model not in installed:
        logger.warning("%s", t("log.rerank_skipped"))
        return False
    try:
        template = client.show_template(model)
    except RagError as exc:
        if exc.code == EXIT_OLLAMA_UNREACHABLE:
            _note_ollama(outcome, exc)
            raise
        logger.warning("%s", t("log.rerank_skipped"))
        return False
    if not template_accepts_rerank(template):
        logger.warning("%s", t("log.rerank_skipped"))
        return False
    try:
        allowed = _rerank_memory_ok(config, model, available)
    except RagError as exc:
        _note_ollama(outcome, exc)
        raise
    if not allowed:
        logger.warning("%s", t("log.rerank_skipped"))
        return False
    return True


def _rerank_memory_ok(config: Config, model: str, available: int | None) -> bool:
    """The default 0.6B tag skips the gate. Other names use 4B or 9B bytes."""
    if model == RERANK_MODEL:
        return True
    if model == config.generate_fallback_model:
        need = config.four_b_min_bytes
    else:
        need = config.nine_b_min_bytes
    free = available_bytes() if available is None else available
    if isinstance(free, bool) or not isinstance(free, int) or free < 0:
        raise RagError(EXIT_RESOURCES, t("ollama.memory_unavailable"))
    return free >= need


def _apply_rerank(
    config: Config,
    store: SqliteStore,
    client: OllamaClient,
    outcome: _Outcome,
    question: str,
    ordered: Sequence[QueryHit],
    fts_ids: Sequence[str],
    *,
    k: int,
    hybrid: bool,
) -> list[Excerpt]:
    """Score the wide list, or rebuild the rerank-off hits from its prefixes.

    The reranker is unloaded before this returns. A dirty ``ps`` is exit 13.
    Any other scoring failure keeps the prefix fusion and exit 0.
    """
    candidates = _select_hits(
        store,
        ordered,
        fts_ids,
        limit=config.rerank_candidates,
        hybrid=hybrid,
        announce=False,
    )
    kept: list[Excerpt] | None = None
    try:
        try:
            kept = _score_candidates(
                client,
                config.rerank_model,
                config.rerank_num_ctx,
                question,
                candidates,
                k,
            )
        except RagError as exc:
            _note_ollama(outcome, exc)
            kept = None
        if kept is None:
            logger.warning("%s", t("log.rerank_blocked"))
            return _select_hits(store, ordered, fts_ids, limit=k, hybrid=hybrid)
        if hybrid:
            logger.info(
                "%s",
                t(
                    "log.hybrid",
                    vector_count=len(ordered),
                    fts_count=len(fts_ids),
                    merged_count=len(candidates),
                ),
            )
        logger.info("%s", t("log.rerank_used", count=len(kept)))
        return kept
    finally:
        _release_reranker(client, config.rerank_model, outcome)


def _score_candidates(
    client: OllamaClient,
    model: str,
    num_ctx: int,
    question: str,
    candidates: Sequence[Excerpt],
    k: int,
) -> list[Excerpt] | None:
    """Return the top ``k`` excerpts, or ``None`` when any score is missing."""
    if not _probe_accepts(client, model, num_ctx):
        return None
    scored: list[tuple[float, int, Excerpt]] = []
    for index, excerpt in enumerate(candidates):
        score = _yes_no(client, model, num_ctx, question, excerpt.text)
        if score is None:
            return None
        scored.append((score, index, excerpt))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:k]]


def _probe_accepts(client: OllamaClient, model: str, num_ctx: int) -> bool:
    """Four fixed pairs must score, and each relevant side must lead by 0.2."""
    for query, relevant, irrelevant in RERANK_PROBE_PAIRS:
        rel = _yes_no(client, model, num_ctx, query, relevant)
        if rel is None:
            return False
        irr = _yes_no(client, model, num_ctx, query, irrelevant)
        if irr is None or rel < irr + RERANK_PROBE_MARGIN:
            return False
    return True


def _yes_no(
    client: OllamaClient,
    model: str,
    num_ctx: int,
    query: str,
    document: str,
) -> float | None:
    """One yes/no probability. An unfit pair is not sent and has no score."""
    user = rerank_user(QUERY_INSTRUCTION, query, document)
    if estimate_tokens(RERANK_SYSTEM) + estimate_tokens(user) + RERANK_OVERHEAD > num_ctx:
        return None
    result = client.chat(
        model,
        (
            ChatMessage("system", RERANK_SYSTEM),
            ChatMessage("user", user),
        ),
        num_ctx=num_ctx,
        num_predict=1,
        temperature=1,
        keep_alive=KEEP_ALIVE_BATCH,
        logprobs=True,
        top_logprobs=RERANK_TOP_LOGPROBS,
    )
    if result.thinking.strip() != "":
        return None
    if result.logprobs is None or len(result.logprobs) != 1:
        return None
    return yes_no_score(result.logprobs[0])


def _release_reranker(client: OllamaClient, model: str, outcome: _Outcome) -> None:
    """Unload the reranker when it is loaded. Do not call release before that."""
    try:
        loaded = client.ps()
    except RagError as exc:
        _note_ollama(outcome, exc)
        raise
    names = [item.name for item in loaded]
    if model not in names:
        if names:
            raise RagError(EXIT_RESOURCES, t("ask.models_still_loaded"))
        return
    try:
        client.release_stage(model)
    except RagError as exc:
        logger.error("%s", t("log.unload_failed", model=model))
        _note_ollama(outcome, exc)
        raise


def _append_neighbors(store: SqliteStore, direct: Sequence[Excerpt]) -> list[Excerpt]:
    """Place same-document ordinal ±1 chunks after every direct hit.

    A chunk already used by a direct hit or an earlier neighbor is skipped.
    A missing ordinal is not an error. Neighbors are not expanded again.
    ``anchor_chunk_id`` is not read.
    """
    seen = {item.chunk_id for item in direct}
    neighbors: list[Excerpt] = []
    for item in direct:
        if item.doc_id == "" or item.ordinal is None:
            continue
        wanted: list[int] = []
        if item.ordinal > 0:
            wanted.append(item.ordinal - 1)
        wanted.append(item.ordinal + 1)
        rows = {
            row.ordinal: row for row in store.chunks_by_ordinals(item.doc_id, wanted)
        }
        for ordinal in wanted:
            row = rows.get(ordinal)
            if row is None or row.chunk_id in seen:
                continue
            seen.add(row.chunk_id)
            neighbors.append(_neighbor_excerpt(store, row, item))
    logger.info("%s", t("log.neighbors", count=len(neighbors)))
    return [*direct, *neighbors]


def _neighbor_excerpt(store: SqliteStore, chunk: Chunk, parent: Excerpt) -> Excerpt:
    """One adjacent row. Distance is the parent's. Text and page are this row's."""
    document = store.get_document(chunk.doc_id)
    return Excerpt(
        chunk_id=chunk.chunk_id,
        filename="" if document is None else document.filename,
        page=-1 if chunk.page_start is None else chunk.page_start,
        kind=chunk.kind,
        text=chunk.text,
        distance=parent.distance,
        doc_id=chunk.doc_id,
        ordinal=chunk.ordinal,
        origin="neighbor",
    )


def _excerpt(hit: QueryHit, *, origin: str = "vector") -> Excerpt:
    return Excerpt(
        chunk_id=hit.chunk_id,
        filename=hit.filename,
        page=hit.page_start,
        kind=hit.kind,
        text=hit.text,
        distance=hit.distance,
        doc_id=hit.doc_id,
        ordinal=hit.ordinal,
        origin=origin,
    )


def _fts_excerpt(store: SqliteStore, item: MergedHit) -> Excerpt | None:
    chunk = store.get_chunk(item.chunk_id)
    if chunk is None:
        return None
    document = store.get_document(chunk.doc_id)
    page = -1 if chunk.page_start is None else chunk.page_start
    return Excerpt(
        chunk_id=chunk.chunk_id,
        filename="" if document is None else document.filename,
        page=page,
        kind=chunk.kind,
        text=chunk.text,
        distance=None,
        doc_id=chunk.doc_id,
        ordinal=chunk.ordinal,
        origin=item.origin,
    )


def _print_hits(items: Sequence[Excerpt]) -> None:
    for item in items:
        if item.distance is None:
            distance = "-"
        else:
            distance = f"{item.distance:.6f}"
        page = t("status.hit.page", page=item.page) if item.page >= 1 else ""
        print(
            t(
                "status.hit",
                chunk_id=item.chunk_id,
                filename=item.filename,
                page=page,
                distance=distance,
                kind=item.kind,
            ),
            t("status.hit.origin", origin=item.origin),
            sep="",
            file=sys.stderr,
        )


def _emit(
    answer: str,
    *,
    refused: bool,
    model: str | None,
    sources: Sequence[Excerpt],
    as_json: bool,
) -> None:
    if as_json:
        payload = {
            "answer": answer,
            "refused": refused,
            "model": model,
            "sources": [_source(item) for item in sources],
        }
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return
    sys.stdout.write(answer)
    if not answer.endswith("\n"):
        sys.stdout.write("\n")


def _source(item: Excerpt) -> dict[str, object]:
    return {
        "chunk_id": item.chunk_id,
        "filename": item.filename,
        "page": None if item.page < 1 else item.page,
        "distance": item.distance,
        "kind": item.kind,
        "origin": item.origin,
    }


def _expected_native(config: Config, model: str) -> int | None:
    if model == config.embed_model:
        return config.embed_native_dimensions
    return _KNOWN_NATIVE.get(model)


def _clean_question(question: str) -> str:
    if not isinstance(question, str):
        raise RagError(EXIT_USAGE, t("ask.question_missing"))
    text = question.strip()
    if text == "":
        raise RagError(EXIT_USAGE, t("ask.question_missing"))
    return text


def _resolve_hybrid(config: Config, hybrid: bool | None) -> bool:
    if hybrid is None:
        return config.hybrid
    if not isinstance(hybrid, bool):
        raise RagError(EXIT_USAGE, t("config.retrieve_bool", label="retrieve.hybrid"))
    return hybrid


def _resolve_neighbors(config: Config, neighbors: bool | None) -> bool:
    if neighbors is None:
        return config.neighbors
    if not isinstance(neighbors, bool):
        raise RagError(EXIT_USAGE, t("config.retrieve_bool", label="retrieve.neighbors"))
    return neighbors


def _resolve_rerank(config: Config, rerank: bool | None) -> bool:
    if rerank is None:
        return config.rerank
    if not isinstance(rerank, bool):
        raise RagError(EXIT_USAGE, t("config.retrieve_bool", label="retrieve.rerank"))
    return rerank


def _resolve_verify(config: Config, verify: bool | None) -> bool:
    if verify is None:
        return config.verify
    if not isinstance(verify, bool):
        raise RagError(EXIT_USAGE, t("config.retrieve_bool", label="retrieve.verify"))
    return verify


def _resolve_k(config: Config, k: int | None) -> int:
    value = config.k if k is None else k
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 8:
        raise RagError(EXIT_USAGE, t("ask.k_range"))
    return value


def _require_disk(config: Config, free_bytes: Callable[[], int] | None) -> None:
    if free_bytes is None:
        probe = config.data_dir if config.data_dir.exists() else config.data_dir.parent
        try:
            free = shutil.disk_usage(probe).free
        except OSError as exc:
            raise RagError(EXIT_DISK, t("disk.free_unreadable")) from exc
    else:
        free = free_bytes()
    if isinstance(free, bool) or not isinstance(free, int) or free < 0:
        raise RagError(EXIT_DISK, t("disk.free_unreadable"))
    if free < config.min_free_bytes:
        raise RagError(
            EXIT_DISK,
            t("disk.free_below", free=free, minimum=config.min_free_bytes),
        )


def _now(clock: Callable[[], str] | None) -> str:
    if clock is not None:
        return clock()
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _session(
    config: Config,
    clock: Callable[[], str] | None,
    *,
    client_factory: Callable | None,
):
    store = SqliteStore.open(config.data_dir)
    try:
        index = ChromaIndex.open(config.data_dir, client_factory=client_factory)
    except BaseException:
        store.close()
        raise
    handler = _attach_log(store.data_dir)
    outcome = _Outcome()
    started = time.monotonic()
    run_id: str | None = None
    try:
        new_id = uuid.uuid4().hex
        store.start_run(Run(run_id=new_id, command="ask", started_at=_now(clock)))
        run_id = new_id
        yield store, index, outcome
    except RagError as exc:
        outcome.code = exc.code
        raise
    finally:
        try:
            if run_id is not None:
                if outcome.ask_latency_ms is None:
                    elapsed = int((time.monotonic() - started) * 1000)
                    outcome.ask_latency_ms = max(0, elapsed)
                logger.info("%s", t("log.exit_code", code=outcome.code))
                store.finish_run(
                    run_id,
                    finished_at=_now(clock),
                    exit_code=outcome.code,
                    chunk_count=outcome.chunk_count,
                    model_name=outcome.model_name,
                    refused_count=outcome.refused_count,
                    ollama_error_count=outcome.ollama_error_count,
                    ask_latency_ms=outcome.ask_latency_ms,
                )
        finally:
            _detach_log(handler)
            index.close()
            store.close()


def _attach_log(data_dir: Path) -> logging.Handler:
    path = data_dir / "logs" / "rag.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logging.getLogger("ragcli").addHandler(handler)
    os.chmod(path, _PRIVATE_FILE)
    return handler


def _detach_log(handler: logging.Handler) -> None:
    logging.getLogger("ragcli").removeHandler(handler)
    handler.close()
