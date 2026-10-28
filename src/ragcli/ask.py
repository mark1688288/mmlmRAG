"""Answer one question from the active collection.

The query string uses the Qwen3-Embedding instruction. Document text does
not. ``Query:`` has no space before the question. Hits whose best cosine
distance is above the configured gate never load a generator. Excerpts are
packed nearest-first into the context budget; a chunk that does not fit is
left out whole, and if the first chunk does not fit the prompt is not sent.
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
)
from ragcli.store.chroma import ChromaIndex, QueryHit
from ragcli.store.sqlite import Run, SqliteStore

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
    """Keep a nearest-first prefix that fits. Do not cut a chunk or skip ahead.

    The first excerpt that does not fit ends the pack. If that is also the
    first excerpt, raise exit 9 and do not build a partial prompt.
    """
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise RagError(EXIT_UNCLASSIFIED, t("ask.budget_type"))
    ordered = sorted(excerpts, key=lambda item: item.distance)
    packed: list[Excerpt] = []
    used = 0
    for item in ordered:
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
            n_results=k,
        )
        ordered = sorted(hits, key=lambda hit: hit.distance)
        if show_chunks:
            for hit in ordered:
                print(_hit_line(hit), file=sys.stderr)
        if not ordered or ordered[0].distance > config.max_cosine_distance:
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
        prompt = system_prompt()
        budget = retrieval_budget(
            num_ctx=config.generate_num_ctx,
            system=prompt,
            question=question,
            num_predict=config.num_predict,
        )
        packed = pack_excerpts([_excerpt(hit) for hit in ordered], budget)
        content = _generate(
            config,
            ollama,
            outcome,
            format_user_message(question, packed),
            system=prompt,
            preferred=config.generate_model if model is None else model,
            allow_fallback=allow_fallback,
            available=available,
        )
    refused = is_refusal(content)
    answer = refusal_sentence() if refused else content
    if refused:
        logger.info("%s", t("log.model_refused"))
        outcome.refused_count = 1
    outcome.chunk_count = len(packed)
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
        _release_if_loaded(client, chosen, body_error, outcome)
    if result is None or result.thinking_only or result.content.strip() == "":
        raise RagError(EXIT_UNCLASSIFIED, t("ask.empty_generation"))
    return result.content


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


def _excerpt(hit: QueryHit) -> Excerpt:
    return Excerpt(
        chunk_id=hit.chunk_id,
        filename=hit.filename,
        page=hit.page_start,
        kind=hit.kind,
        text=hit.text,
        distance=hit.distance,
    )


def _hit_line(hit: QueryHit) -> str:
    distance = f"{hit.distance:.6f}"
    page = t("status.hit.page", page=hit.page_start) if hit.page_start >= 1 else ""
    return t(
        "status.hit",
        chunk_id=hit.chunk_id,
        filename=hit.filename,
        page=page,
        distance=distance,
        kind=hit.kind,
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
