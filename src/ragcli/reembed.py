"""Rebuild the vector index, switch collections, delete one document, and report status.

``rag reembed`` reads chunk text and stored vectors. It does not open source
files, does not parse, and does not run OCR. The same model at its stored
width copies those floats into Chroma with no L2 norm. A shorter width uses
``truncate_normalize`` and does not call Ollama. A different model is embedded
from plain chunk text at that model's native width; the old vectors and the
old collection stay unless ``--drop-old`` is set.

``rag use`` only moves the active pointer, and only after the collection
metadata matches its name. ``rag delete`` removes Chroma ids first. If that
fails, SQLite and image files stay so the delete can be repeated.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import stat
import sys
import uuid
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ragcli.config import Config
from ragcli.errors import (
    EXIT_DISK,
    EXIT_MISMATCH,
    EXIT_MODEL_MISSING,
    EXIT_NOT_FOUND,
    EXIT_OK,
    EXIT_OLLAMA_UNREACHABLE,
    EXIT_UNCLASSIFIED,
    EXIT_USAGE,
    RagError,
)
from ragcli.i18n import t
from ragcli.models_lock import ollama_lock
from ragcli.ollama import OllamaClient
from ragcli.store.chroma import (
    ChromaIndex,
    IndexRecord,
    collection_name,
)
from ragcli.store.sqlite import (
    ActiveIndex,
    Chunk,
    Document,
    Run,
    SqliteStore,
    VectorRecord,
)

logger = logging.getLogger("ragcli.reembed")

KNOWN_NATIVE_DIM = {
    "qwen3-embedding:0.6b": 1024,
    "qwen3-embedding:4b": 2560,
    "qwen3-embedding:8b": 4096,
}

_DOC_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_FILE = 0o600


@dataclass(frozen=True, slots=True)
class ReembedPlan:
    """What one reembed would do. ``reparse`` is always false."""

    calls_ollama: bool
    reparse: bool
    model: str
    dimensions: int
    native_dim: int


@dataclass
class _Outcome:
    code: int = EXIT_OK
    chunk_count: int = 0
    model_name: str | None = None
    ollama_error_count: int = 0


def plan_reembed(
    *,
    stored_model: str,
    stored_native_dim: int,
    target_model: str,
    target_dimensions: int,
    target_native_dim: int,
) -> ReembedPlan:
    """Decide a rebuild without reading the corpus or calling Ollama.

    The same model cannot grow past the stored native width. A different
    model's own native width is not that error, including 2560 for
    ``qwen3-embedding:4b``. Neither path reparses.
    """
    _require_width(stored_native_dim, "stored_native_dim")
    _require_width(target_dimensions, "dimensions")
    _require_width(target_native_dim, "native_dim")
    if not isinstance(stored_model, str) or stored_model.strip() == "":
        raise RagError(EXIT_MISMATCH, t("reembed.native_missing"))
    if not isinstance(target_model, str) or target_model.strip() == "":
        raise RagError(EXIT_USAGE, t("reembed.model_empty"))
    if target_model == stored_model:
        if target_dimensions > stored_native_dim:
            raise RagError(EXIT_MISMATCH, t("reembed.same_model_wider"))
        return ReembedPlan(
            calls_ollama=False,
            reparse=False,
            model=target_model,
            dimensions=target_dimensions,
            native_dim=stored_native_dim,
        )
    if target_dimensions > target_native_dim:
        raise RagError(EXIT_MISMATCH, t("reembed.dimensions_exceed_native"))
    return ReembedPlan(
        calls_ollama=True,
        reparse=False,
        model=target_model,
        dimensions=target_dimensions,
        native_dim=target_native_dim,
    )


def native_width(config: Config, model: str, explicit: int | None) -> int:
    """Native width for ``model``. Unknown tags require ``explicit``."""
    if explicit is not None:
        _require_width(explicit, "--native-dimensions")
    if model == config.embed_model:
        expected = config.embed_native_dimensions
    elif model in KNOWN_NATIVE_DIM:
        expected = KNOWN_NATIVE_DIM[model]
    elif explicit is None:
        raise RagError(EXIT_USAGE, t("reembed.native_dimensions_required"))
    else:
        return explicit
    if explicit is not None and explicit != expected:
        raise RagError(EXIT_MISMATCH, t("reembed.native_dimensions_mismatch"))
    return expected


def run_reembed(
    config: Config,
    *,
    model: str | None = None,
    dimensions: int | None = None,
    native_dimensions: int | None = None,
    activate: bool = True,
    drop_old: bool = False,
    lock_timeout: float | None = None,
    client: OllamaClient | None = None,
    free_bytes: Callable[[], int] | None = None,
    clock: Callable[[], str] | None = None,
    client_factory: Callable | None = None,
) -> int:
    """Rebuild or switch the index. Returns a process exit code."""
    _require_disk(config, free_bytes)
    target = _target_model(config, model)
    native = native_width(config, target, native_dimensions)
    width = _target_dimensions(config, target, native, dimensions)
    if not target.startswith("qwen3-embedding"):
        print(t("ask.prefix_warning"), file=sys.stderr)
    with _session(config, "reembed", clock, client_factory=client_factory) as (store, index, outcome):
        outcome.chunk_count = _reembed(
            config,
            store,
            index,
            outcome,
            model=target,
            dimensions=width,
            native=native,
            activate=activate,
            drop_old=drop_old,
            lock_timeout=lock_timeout,
            client=client,
            clock=clock,
        )
        return outcome.code


def run_use(
    config: Config,
    name: str,
    *,
    clock: Callable[[], str] | None = None,
    client_factory: Callable | None = None,
) -> int:
    """Point the active index at an existing collection."""
    if not isinstance(name, str) or name.strip() == "":
        raise RagError(EXIT_USAGE, t("usage.use_needs_collection"))
    with _session(config, None, clock, client_factory=client_factory) as (store, index, _outcome):
        info = index.describe(name)
        store.set_active(
            ActiveIndex(
                collection=info.name,
                embed_model=info.embed_model,
                dimensions=info.dimensions,
                embed_num_ctx=info.embed_num_ctx,
            )
        )
        logger.info("%s", t("status.active_collection", name=info.name))
        print(t("status.active_collection", name=info.name))
        return EXIT_OK


def run_delete(
    config: Config,
    *,
    path: Path | None = None,
    doc_id: str | None = None,
    clock: Callable[[], str] | None = None,
    client_factory: Callable | None = None,
) -> int:
    """Delete one document from every collection, then from SQLite."""
    if (path is None) == (doc_id is None):
        raise RagError(EXIT_USAGE, t("usage.delete_needs_one"))
    with _session(config, "delete", clock, client_factory=client_factory) as (store, index, outcome):
        if doc_id is not None:
            document = _document_for_id(store, doc_id)
        else:
            document = _document_for_path(store, path)
        ids = store.chunk_ids_for(document.doc_id)
        if ids:
            try:
                index.delete(ids)
            except Exception as exc:
                raise RagError(EXIT_UNCLASSIFIED, t("reembed.index_delete_failed")) from exc
        store.delete_document(document.doc_id)
        outcome.chunk_count = len(ids)
        logger.info("%s", t("log.document_deleted", doc_id=document.doc_id))
        print(t("status.deleted", doc_id=document.doc_id))
        return EXIT_OK


def run_status(
    config: Config,
    *,
    client: OllamaClient | None = None,
    free_bytes: Callable[[], int] | None = None,
    clock: Callable[[], str] | None = None,
    client_factory: Callable | None = None,
) -> int:
    """Print the local corpus and whether Ollama answers. Exit 6 when it does not."""
    with _session(config, "status", clock, client_factory=client_factory) as (store, index, outcome):
        outcome.code = _report_status(config, store, index, client, free_bytes)
        return outcome.code


def _reembed(
    config: Config,
    store: SqliteStore,
    index: ChromaIndex,
    outcome: _Outcome,
    *,
    model: str,
    dimensions: int,
    native: int,
    activate: bool,
    drop_old: bool,
    lock_timeout: float | None,
    client: OllamaClient | None,
    clock: Callable[[], str] | None,
) -> int:
    previous = store.get_active()
    chunks = _working_chunks(store)
    logger.info("%s", t("log.reembed", model=model))
    if not chunks:
        plan_calls_ollama = False
        embed_num_ctx = config.embed_num_ctx
        records: list[IndexRecord] = []
        embedded: list[Sequence[float]] | None = None
    else:
        stored, missing, others = _classify(store, chunks, model)
        if not missing:
            plan, embed_num_ctx = _stored_plan(chunks, stored, model, dimensions, native)
            if plan.calls_ollama or plan.reparse:
                raise RagError(EXIT_UNCLASSIFIED, t("reembed.plan_invalid"))
            embedded = None
            records = _records(store, chunks, [stored[chunk.chunk_id].vector for chunk in chunks])
            plan_calls_ollama = False
        elif not others:
            raise RagError(EXIT_MISMATCH, t("reembed.native_missing"))
        else:
            plan = plan_reembed(
                stored_model=sorted(others)[0],
                stored_native_dim=1,
                target_model=model,
                target_dimensions=dimensions,
                target_native_dim=native,
            )
            if not plan.calls_ollama or plan.reparse:
                raise RagError(EXIT_UNCLASSIFIED, t("reembed.plan_invalid"))
            ollama = client if client is not None else OllamaClient(config.ollama_base_url)
            vectors = _embed_texts(
                config,
                ollama,
                model,
                chunks,
                native,
                lock_timeout,
                outcome,
            )
            embedded = vectors
            embed_num_ctx = config.embed_num_ctx
            records = _records(store, chunks, vectors)
            plan_calls_ollama = True
    name = index.upsert(
        records,
        model=model,
        dimensions=dimensions,
        embed_num_ctx=embed_num_ctx,
    )
    expected = collection_name(model, dimensions)
    if name != expected:
        raise RagError(EXIT_MISMATCH, t("reembed.collection_name_mismatch"))
    logger.info("%s", t("log.collection", name=name))
    _persist(
        store,
        chunks,
        embedded,
        model=model,
        native=native,
        dimensions=dimensions,
        embed_num_ctx=embed_num_ctx,
        activate=activate,
        clock=clock,
        called_ollama=plan_calls_ollama,
    )
    if activate:
        print(t("status.rebuilt", name=name, count=len(chunks)))
    else:
        print(t("status.created", name=name))
    if drop_old and previous is not None and previous.collection != name:
        index.drop(previous.collection)
        logger.info("%s", t("log.collection_dropped", name=previous.collection))
        print(t("status.collection_dropped", name=previous.collection))
        if not activate:
            print(t("status.pointer_dangling"), file=sys.stderr)
    return len(chunks)


def _stored_plan(
    chunks: Sequence[Chunk],
    stored: dict[str, VectorRecord],
    model: str,
    dimensions: int,
    native: int,
) -> tuple[ReembedPlan, int]:
    widths = {vector.native_dim for vector in stored.values()}
    contexts = {vector.num_ctx for vector in stored.values()}
    if len(widths) != 1 or len(contexts) != 1:
        raise RagError(EXIT_MISMATCH, t("reembed.stored_width_mismatch"))
    stored_native = next(iter(widths))
    plan = plan_reembed(
        stored_model=model,
        stored_native_dim=stored_native,
        target_model=model,
        target_dimensions=dimensions,
        target_native_dim=native,
    )
    for chunk in chunks:
        if chunk.token_estimate > stored[chunk.chunk_id].num_ctx:
            raise RagError(EXIT_MISMATCH, t("reembed.context_truncated"))
    return plan, next(iter(contexts))


def _persist(
    store: SqliteStore,
    chunks: Sequence[Chunk],
    embedded: Sequence[Sequence[float]] | None,
    *,
    model: str,
    native: int,
    dimensions: int,
    embed_num_ctx: int,
    activate: bool,
    clock: Callable[[], str] | None,
    called_ollama: bool,
) -> None:
    stamped = _now(clock)
    with store.transaction():
        if called_ollama:
            if embedded is None or len(embedded) != len(chunks):
                raise RagError(EXIT_UNCLASSIFIED, t("reembed.embed_count"))
            for chunk, vector in zip(chunks, embedded, strict=True):
                store.upsert_vector(
                    VectorRecord(
                        chunk_id=chunk.chunk_id,
                        embed_model=model,
                        native_dim=native,
                        vector=tuple(float(value) for value in vector),
                        num_ctx=embed_num_ctx,
                        embedded_at=stamped,
                    )
                )
                if chunk.embed_state != "done":
                    store.set_embed_state(chunk.chunk_id, "done")
            for doc_id in {chunk.doc_id for chunk in chunks}:
                store.refresh_document_status(doc_id)
        if activate:
            store.set_active(
                ActiveIndex(
                    collection=collection_name(model, dimensions),
                    embed_model=model,
                    dimensions=dimensions,
                    embed_num_ctx=embed_num_ctx,
                )
            )


def _embed_texts(
    config: Config,
    client: OllamaClient,
    model: str,
    chunks: Sequence[Chunk],
    native: int,
    lock_timeout: float | None,
    outcome: _Outcome,
) -> list[list[float]]:
    embedded: list[list[float]] = []
    body_error: BaseException | None = None
    with ollama_lock(config.data_dir, timeout=lock_timeout):
        try:
            if model not in set(client.tags()):
                raise RagError(
                    EXIT_MODEL_MISSING,
                    t("ollama.model_missing", names=model),
                )
            outcome.model_name = model
            for batch in _batches(chunks, config.embed_batch_size):
                texts = [chunk.text for chunk in batch]
                try:
                    vectors = client.embed(model, texts, num_ctx=config.embed_num_ctx)
                except RagError:
                    outcome.ollama_error_count += 1
                    raise
                if len(vectors) != len(batch) or any(len(vector) != native for vector in vectors):
                    outcome.ollama_error_count += 1
                    raise RagError(EXIT_MISMATCH, t("reembed.width_mismatch"))
                embedded.extend(vectors)
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            _release(client, model, body_error)
    return embedded


def _release(client: OllamaClient, model: str, body_error: BaseException | None) -> None:
    try:
        client.release_stage(model)
    except RagError:
        logger.error("%s", t("log.unload_failed", model=model))
        if body_error is None:
            raise


def _report_status(
    config: Config,
    store: SqliteStore,
    index: ChromaIndex,
    client: OllamaClient | None,
    free_bytes: Callable[[], int] | None,
) -> int:
    mode = stat.S_IMODE(store.data_dir.stat().st_mode)
    if mode & 0o077:
        print(t("status.data_dir_readable"), file=sys.stderr)
    done = store.count_chunks("done")
    pending = store.count_chunks("pending")
    blocked = store.count_chunks("blocked")
    failed = store.count_images("caption_failed")
    free = _read_free(config, free_bytes)
    probe = client if client is not None else OllamaClient(config.ollama_base_url)
    version, loaded, failure = _probe(probe)
    code = EXIT_OK
    if failure is not None:
        if failure.code == EXIT_OLLAMA_UNREACHABLE:
            print(t("ollama.unreachable"))
        else:
            print(failure.message)
        print(t("status.ps_unreadable"))
        code = failure.code
    else:
        print(t("status.ollama_version", version=version))
        if not loaded:
            print(t("status.ps_empty"))
        for item in loaded or ():
            if item.context_length is None:
                print(t("status.ps", name=item.name))
            else:
                print(
                    t(
                        "status.ps_context",
                        name=item.name,
                        context_length=item.context_length,
                    )
                )
    _print_index(store, index)
    print(t("status.embedded", count=done))
    print(t("status.pending", count=pending))
    print(t("status.blocked", count=blocked))
    print(t("status.failed_images", count=failed))
    print(t("status.free_disk", free=free))
    for run in _finished_runs(store):
        print(
            t(
                "status.run",
                command=run.command,
                exit_code=run.exit_code,
                chunk_count=run.chunk_count,
            )
        )
    return code


def _print_index(store: SqliteStore, index: ChromaIndex) -> None:
    active = store.get_active()
    if active is None:
        print(t("status.no_active_collection"))
        return
    print(t("status.active_collection", name=active.collection))
    try:
        count = index.count(
            model=active.embed_model,
            dimensions=active.dimensions,
            embed_num_ctx=active.embed_num_ctx,
        )
        present = index.chunk_ids(active.collection)
    except RagError as exc:
        print(exc.message)
        return
    print(t("status.collection_count", count=count))
    sqlite_ids = _all_chunk_ids(store)
    done_ids = {
        chunk.chunk_id
        for document in store.list_documents()
        for chunk in store.chunks_for_document(document.doc_id)
        if chunk.embed_state == "done"
    }
    extra = sorted(set(present) - sqlite_ids)
    missing = sorted(done_ids - set(present))
    if extra:
        print(t("status.index_extra", ids=" ".join(extra)))
    if missing:
        print(t("status.index_missing", ids=" ".join(missing)))
    if count != store.count_chunks("done"):
        print(t("status.count_mismatch"))


def _probe(
    client: OllamaClient,
) -> tuple[str | None, tuple | None, RagError | None]:
    try:
        version = client.version()
        loaded = client.ps()
    except RagError as exc:
        if exc.code == EXIT_OLLAMA_UNREACHABLE:
            logger.error("%s", t("ollama.unreachable"))
        return None, None, exc
    return version, loaded, None


def _finished_runs(store: SqliteStore) -> list[Run]:
    finished = [run for run in store.recent_runs(limit=20) if run.finished_at is not None]
    return finished[:5]


def _working_chunks(store: SqliteStore) -> list[Chunk]:
    rows: list[Chunk] = []
    for document in store.list_documents():
        for chunk in store.chunks_for_document(document.doc_id):
            if chunk.embed_state == "blocked" or chunk.text == "":
                continue
            rows.append(chunk)
    return rows


def _classify(
    store: SqliteStore, chunks: Sequence[Chunk], model: str
) -> tuple[dict[str, VectorRecord], list[Chunk], set[str]]:
    stored: dict[str, VectorRecord] = {}
    missing: list[Chunk] = []
    others: set[str] = set()
    for chunk in chunks:
        vector = store.get_vector(chunk.chunk_id, model)
        if vector is None:
            missing.append(chunk)
        else:
            stored[chunk.chunk_id] = vector
        for row in store.vectors_for_chunk(chunk.chunk_id):
            if row.embed_model != model:
                others.add(row.embed_model)
    return stored, missing, others


def _records(
    store: SqliteStore, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
) -> list[IndexRecord]:
    documents: dict[str, Document] = {}
    records: list[IndexRecord] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        document = documents.get(chunk.doc_id)
        if document is None:
            document = store.get_document(chunk.doc_id)
            if document is None:
                raise RagError(EXIT_NOT_FOUND, t("store.document_missing"))
            documents[chunk.doc_id] = document
        records.append(
            IndexRecord(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                embedding=tuple(float(value) for value in vector),
                source_path=document.source_path,
                filename=document.filename,
                kind=chunk.kind,
                ordinal=chunk.ordinal,
                page_start=-1 if chunk.page_start is None else chunk.page_start,
                page_end=-1 if chunk.page_end is None else chunk.page_end,
                image_id="" if chunk.image_id is None else chunk.image_id,
            )
        )
    return records


def _document_for_id(store: SqliteStore, doc_id: str) -> Document:
    if not isinstance(doc_id, str) or _DOC_ID_RE.fullmatch(doc_id) is None:
        raise RagError(EXIT_USAGE, t("reembed.doc_id_invalid"))
    found = store.get_document(doc_id)
    if found is None:
        raise RagError(EXIT_NOT_FOUND, t("store.document_missing"))
    return found


def _document_for_path(store: SqliteStore, raw: Path | None) -> Document:
    if raw is None:
        raise RagError(EXIT_USAGE, t("usage.delete_needs_one"))
    path = raw.expanduser()
    candidates: list[str] = [str(path)]
    try:
        candidates.append(str(path.resolve(strict=False)))
    except OSError:
        pass
    if path.exists():
        try:
            candidates.append(str(path.resolve(strict=True)))
        except OSError:
            pass
    seen: set[str] = set()
    for candidate in candidates:
        if candidate == "" or candidate in seen:
            continue
        seen.add(candidate)
        found = store.find_document_by_path(candidate)
        if found is not None:
            return found
    raise RagError(EXIT_NOT_FOUND, t("store.document_missing"))


def _all_chunk_ids(store: SqliteStore) -> set[str]:
    found: set[str] = set()
    for document in store.list_documents():
        found.update(store.chunk_ids_for(document.doc_id))
    return found


def _target_model(config: Config, model: str | None) -> str:
    chosen = config.embed_model if model is None else model.strip()
    if chosen == "" or "\x00" in chosen:
        raise RagError(EXIT_USAGE, t("reembed.model_empty"))
    return chosen


def _target_dimensions(config: Config, model: str, native: int, requested: int | None) -> int:
    if requested is None:
        if model == config.embed_model:
            return config.embed_dimensions
        return native
    _require_width(requested, "dimensions")
    return requested


def _require_width(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        if label.startswith("--"):
            raise RagError(EXIT_USAGE, t("usage.named_positive_int", name=label))
        raise RagError(EXIT_MISMATCH, t("usage.named_positive_int", name=label))
    return value


def _batches(chunks: Sequence[Chunk], size: int) -> list[list[Chunk]]:
    _require_width(size, "batch_size")
    return [list(chunks[start : start + size]) for start in range(0, len(chunks), size)]


def _require_disk(config: Config, free_bytes: Callable[[], int] | None) -> None:
    free = _read_free(config, free_bytes)
    if free < config.min_free_bytes:
        raise RagError(
            EXIT_DISK,
            t("disk.free_below", free=free, minimum=config.min_free_bytes),
        )


def _read_free(config: Config, free_bytes: Callable[[], int] | None) -> int:
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
    return free


def _now(clock: Callable[[], str] | None) -> str:
    if clock is not None:
        return clock()
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _session(
    config: Config,
    command: str | None,
    clock: Callable[[], str] | None,
    *,
    client_factory: Callable | None = None,
):
    store = SqliteStore.open(config.data_dir)
    try:
        index = ChromaIndex.open(config.data_dir, client_factory=client_factory)
    except BaseException:
        store.close()
        raise
    handler = _attach_log(store.data_dir)
    outcome = _Outcome()
    run_id: str | None = None
    try:
        if command is not None:
            new_id = uuid.uuid4().hex
            store.start_run(Run(run_id=new_id, command=command, started_at=_now(clock)))
            run_id = new_id
        yield store, index, outcome
    except RagError as exc:
        outcome.code = exc.code
        raise
    finally:
        try:
            logger.info("%s", t("log.exit_code", code=outcome.code))
            if run_id is not None:
                store.finish_run(
                    run_id,
                    finished_at=_now(clock),
                    exit_code=outcome.code,
                    chunk_count=outcome.chunk_count,
                    model_name=outcome.model_name,
                    ollama_error_count=outcome.ollama_error_count,
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
