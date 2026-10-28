"""Seven-stage ingest for one file.

The order is fixed: preflight, identity, parse, OCR, image chunks, vision,
then embed. Parsers only return records. This module writes SQLite and
``data/images``. OCR finishes before any generator is loaded, and the
generator is unloaded before the embed model is loaded. A document error
before the parse transaction commits rolls that attempt back and deletes
image files written for it. Later failures keep the committed parse.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from ragcli.chunking import estimate_tokens
from ragcli.config import Config
from ragcli.errors import (
    EXIT_DISK,
    EXIT_DOCUMENT,
    EXIT_MISMATCH,
    EXIT_MODEL_MISSING,
    EXIT_NOT_FOUND,
    EXIT_NO_CHUNKS,
    EXIT_OK,
    EXIT_OLLAMA_UNREACHABLE,
    EXIT_RESOURCES,
    EXIT_UNCLASSIFIED,
    EXIT_UNSUPPORTED_TYPE,
    EXIT_USAGE,
    RagError,
)
from ragcli.i18n import t
from ragcli.images.caption import (
    CAPTION_DONE,
    VisionImage,
    VisionSettings,
    describe_batch,
)
from ragcli.images.route import (
    ROUTE_CAPTION_ONLY,
    ROUTE_LABEL_AND_CAPTION,
    ROUTE_TEXT_RICH,
    add_caption_line,
    compose,
    needs_vision,
    route_for,
)
from ragcli.models_lock import ollama_lock
from ragcli.ollama import (
    FOUR_B_MODEL,
    NINE_B_MODEL,
    OllamaClient,
)
from ragcli.parse.model import (
    PARSER_VERSION,
    ParsedChunk,
    ParsedDocument,
    ParsedImage,
    ParseLimits,
    ParseResult,
)
from ragcli.store.chroma import (
    ChromaIndex,
    IndexRecord,
    collection_name,
)
from ragcli.store.sqlite import (
    ActiveIndex,
    Chunk,
    Document,
    ImageRecord,
    Run,
    SqliteStore,
    VectorRecord,
    chunk_id_for,
)

logger = logging.getLogger("ragcli.ingest")

NO_OCR_IMAGE_LIMIT = 20
_SUFFIXES = frozenset(
    {
        ".pdf",
        ".docx",
        ".pptx",
        ".txt",
        ".md",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".tif",
        ".tiff",
        ".gif",
        ".bmp",
        ".heic",
        ".heif",
    }
)
_KNOWN_NATIVE = {
    "qwen3-embedding:0.6b": 1024,
    "qwen3-embedding:4b": 2560,
    "qwen3-embedding:8b": 4096,
}
_STRICT_IMAGE_STATUSES = frozenset({"caption_failed", "empty_no_vision"})
_PRIVATE_FILE = 0o600


@dataclass
class _Stats:
    chunk_count: int = 0
    ocr_count: int = 0
    vision_count: int = 0
    ollama_error_count: int = 0
    model_name: str | None = None
    failures: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class _FileDone:
    doc_id: str
    reused: bool
    filename: str


@dataclass
class _Job:
    config: Config
    store: SqliteStore
    index: ChromaIndex
    stats: _Stats
    client: OllamaClient | None
    ocr_engine: Callable | None
    free_bytes: Callable[[], int] | None
    available: int | None
    clock: Callable[[], str] | None
    force: bool
    no_ocr: bool
    no_vision: bool
    no_embed: bool
    strict: bool
    max_pages: int
    max_bytes: int
    lock_timeout: float | None
    probed_embed: bool = False
    probed_vision: bool = False


def run_ingest(
    config: Config,
    *,
    path: Path | None = None,
    force: bool = False,
    resume: bool = False,
    no_ocr: bool = False,
    no_vision: bool = False,
    no_embed: bool = False,
    strict: bool = False,
    max_pages: int | None = None,
    max_bytes: int | None = None,
    lock_timeout: float | None = None,
    client: OllamaClient | None = None,
    ocr_engine: Callable | None = None,
    free_bytes: Callable[[], int] | None = None,
    available: int | None = None,
    clock: Callable[[], str] | None = None,
) -> int:
    """Ingest one file, or resume unfinished current-parser work.

    ``client`` and ``ocr_engine`` are for tests. The engine matches the OCR
    callable: it receives a raster plus ``languages``, ``recognition_level``,
    and ``confidence_threshold``. ``free_bytes`` replaces the disk check.
    ``available`` replaces the vision memory reading.
    """
    if path is None and not resume:
        raise RagError(EXIT_USAGE, t("usage.file_not_directory"))
    page_cap = config.max_pages if max_pages is None else max_pages
    byte_cap = config.max_bytes if max_bytes is None else max_bytes
    if path is not None:
        _check_file(path, byte_cap)
    _require_disk(config, free_bytes)

    store = SqliteStore.open(config.data_dir)
    try:
        index = ChromaIndex.open(config.data_dir)
    except BaseException:
        store.close()
        raise
    handler = _attach_log(store.data_dir)
    run_id = _begin_run(store, clock)
    job = _Job(
        config=config,
        store=store,
        index=index,
        stats=_Stats(),
        client=client,
        ocr_engine=ocr_engine,
        free_bytes=free_bytes,
        available=available,
        clock=clock,
        force=force,
        no_ocr=no_ocr,
        no_vision=no_vision,
        no_embed=no_embed,
        strict=strict,
        max_pages=page_cap,
        max_bytes=byte_cap,
        lock_timeout=lock_timeout,
    )
    code = EXIT_OK
    error: RagError | None = None
    recorded = False
    try:
        try:
            code = _execute(job, path=path, resume_only=path is None)
        except RagError as exc:
            code = exc.code
            error = exc
            if code == EXIT_OLLAMA_UNREACHABLE:
                job.stats.ollama_error_count += 1
        recorded = True
        _finish_run(job, run_id, code)
        if error is not None:
            raise error
        return code
    finally:
        _detach_log(handler)
        index.close()
        store.close()


def _execute(job: _Job, *, path: Path | None, resume_only: bool) -> int:
    if resume_only:
        doc_ids, missing = _resume_all(job)
        filename = None
        reused = False
    else:
        assert path is not None
        outcome = _ingest_file(job, path)
        doc_ids = [outcome.doc_id]
        missing = []
        filename = outcome.filename
        reused = outcome.reused
    if missing:
        raise RagError(EXIT_NOT_FOUND, t("io.file_unreadable", path=missing[0]))
    code = _result_code(job, doc_ids)
    if code == EXIT_OK:
        if resume_only:
            print(t("ingest.resumed"))
        elif reused:
            print(t("ingest.already", filename=filename))
        else:
            print(t("ingest.done", filename=filename))
    elif job.stats.failures or _stored_strict_failure(job, doc_ids):
        print(t("ingest.partial_failure"), file=_stderr())
    else:
        print(t("ingest.no_chunks"), file=_stderr())
    return code


def _resume_all(job: _Job) -> tuple[list[str], list[str]]:
    """Reparse stale documents whose files remain, then finish current ones.

    A stale row whose file is gone is not embedded. The caller turns that
    into exit 4 after current-parser work has continued.
    """
    processed: list[str] = []
    seen: set[str] = set()
    missing: list[str] = []
    for document in job.store.list_documents():
        if document.parser_version == PARSER_VERSION:
            continue
        source = Path(document.source_path)
        if not _readable_file(source):
            missing.append(document.source_path)
            continue
        _purge(job, document.doc_id)
        outcome = _ingest_file(job, source)
        seen.add(outcome.doc_id)
        processed.append(outcome.doc_id)
    for document in job.store.list_documents():
        if document.parser_version != PARSER_VERSION or document.doc_id in seen:
            continue
        _continue_document(job, document.doc_id)
        seen.add(document.doc_id)
        processed.append(document.doc_id)
    return processed, missing


def _ingest_file(job: _Job, path: Path) -> _FileDone:
    resolved = _check_file(path, job.max_bytes)
    digest = _sha256(resolved)
    by_path = job.store.find_document_by_path(str(resolved))
    if by_path is not None and by_path.doc_id != digest:
        _purge(job, by_path.doc_id)
    by_hash = job.store.get_document(digest)
    if by_hash is not None and (job.force or by_hash.parser_version != PARSER_VERSION):
        _purge(job, by_hash.doc_id)
        by_hash = None
    if by_hash is not None:
        if by_hash.source_path != str(resolved) or by_hash.filename != resolved.name:
            _retarget(job, by_hash, resolved)
            job.store.update_source_path(by_hash.doc_id, str(resolved), resolved.name)
        logger.info("%s", t("log.stage.reuse"))
        _continue_document(job, by_hash.doc_id)
        return _FileDone(by_hash.doc_id, True, resolved.name)

    logger.info("%s", t("log.stage.parse"))
    _probe_if_needed(job, will_parse=True, doc_ids=set())
    result = _parse(resolved, _limits(job))
    if result.document.doc_id != digest:
        raise RagError(EXIT_UNCLASSIFIED, t("ingest.hash_mismatch"))
    _guard_no_ocr(job, result.images)
    _require_disk(job.config, job.free_bytes)
    _commit_parse(job, result)
    contexts = {image.image_id: image.preceding_chunk_count for image in result.images}
    _ocr_vision_embed(job, digest, contexts)
    return _FileDone(digest, False, resolved.name)


def _continue_document(job: _Job, doc_id: str) -> None:
    _probe_if_needed(job, will_parse=False, doc_ids={doc_id})
    _ocr_vision_embed(job, doc_id, {})


def _ocr_vision_embed(job: _Job, doc_id: str, contexts: dict[str, int]) -> None:
    images = job.store.images_for_document(doc_id)
    _guard_no_ocr(job, images)
    if any(image.status == "pending_ocr" and not _is_skipped(image) for image in images):
        logger.info("%s", t("log.stage.ocr"))
        _run_ocr(job, doc_id)
    _insert_image_chunks(job, doc_id, contexts)
    if job.no_vision:
        _apply_no_vision(job, doc_id)
    elif _vision_images(job, doc_id):
        logger.info("%s", t("log.stage.vision"))
        _run_vision(job, doc_id)
    if not job.no_embed and _pending_for(job, {doc_id}):
        logger.info("%s", t("log.stage.embed"))
        _embed(job, {doc_id})
    job.store.refresh_document_status(doc_id)


def _probe_if_needed(job: _Job, *, will_parse: bool, doc_ids: set[str]) -> None:
    need_embed, need_vision = _ollama_needed(job, will_parse=will_parse, doc_ids=doc_ids)
    check_embed = need_embed and not job.probed_embed
    check_vision = need_vision and not job.probed_vision
    if not check_embed and not check_vision:
        return
    logger.info("%s", t("log.stage.preflight"))
    client = _client(job)
    client.version()
    installed = set(client.tags())
    missing: list[str] = []
    if check_embed:
        model = _active_embed_model(job)
        if model not in installed:
            missing.append(model)
    if check_vision:
        names = (job.config.generate_model, job.config.generate_fallback_model)
        if not any(name in installed for name in names):
            for name in names:
                if name not in missing:
                    missing.append(name)
    if missing:
        names = t("ollama.list_joiner").join(missing)
        raise RagError(EXIT_MODEL_MISSING, t("ollama.model_missing", names=names))
    if check_embed:
        job.probed_embed = True
    if check_vision:
        job.probed_vision = True


def _ollama_needed(job: _Job, *, will_parse: bool, doc_ids: set[str]) -> tuple[bool, bool]:
    if will_parse:
        return (not job.no_embed, not job.no_vision)
    need_embed = not job.no_embed and bool(_pending_for(job, doc_ids))
    need_vision = False
    if job.no_vision:
        return need_embed, need_vision
    for doc_id in doc_ids:
        for image in job.store.images_for_document(doc_id):
            if _is_skipped(image):
                continue
            if image.status == "pending_ocr":
                need_vision = True
            elif (
                image.status == "pending_caption"
                and image.route is not None
                and needs_vision(image.route)
            ):
                need_vision = True
        for chunk in job.store.chunks_for_document(doc_id):
            if chunk.embed_state == "blocked" and chunk.image_id:
                need_vision = True
    return need_embed, need_vision


def _run_ocr(job: _Job, doc_id: str) -> None:
    from ragcli.images.ocr import recognize

    for image in job.store.images_for_document(doc_id):
        if image.status != "pending_ocr" or _is_skipped(image):
            continue
        if job.no_ocr:
            job.store.update_image(
                replace(
                    image,
                    route=ROUTE_CAPTION_ONLY,
                    status="pending_caption",
                    ocr_text=None,
                    ocr_confidence=None,
                )
            )
            continue
        payload = _read_image(job, doc_id, image.image_id)
        result = recognize(
            payload,
            languages=job.config.ocr_languages,
            recognition_level=job.config.ocr_recognition_level,
            confidence_threshold=job.config.ocr_confidence_threshold,
            max_pixels=job.config.max_pixels,
            engine=job.ocr_engine,
        )
        body = result.text.strip()
        route = route_for(body, rich_tokens=job.config.ocr_rich_tokens)
        status = "ocr_done" if route == ROUTE_TEXT_RICH else "pending_caption"
        job.store.update_image(
            replace(
                image,
                ocr_text=body or None,
                ocr_confidence=result.confidence,
                route=route,
                status=status,
            )
        )
        job.stats.ocr_count += 1


def _insert_image_chunks(job: _Job, doc_id: str, contexts: dict[str, int]) -> None:
    document = job.store.get_document(doc_id)
    if document is None:
        raise RagError(EXIT_NOT_FOUND, t("store.document_missing"))
    chunks = job.store.chunks_for_document(doc_id)
    occupied = {chunk.image_id for chunk in chunks if chunk.image_id}
    ordinal = max((chunk.ordinal for chunk in chunks), default=-1) + 1
    text_chunks = [chunk for chunk in chunks if chunk.kind == "text"]
    created: list[Chunk] = []
    for image in job.store.images_for_document(doc_id):
        if _is_skipped(image) or image.image_id in occupied or image.route is None:
            continue
        if image.status in _STRICT_IMAGE_STATUSES:
            continue
        before, after, anchor = _context(text_chunks, image, contexts.get(image.image_id))
        routed = compose(
            image.route,
            image.ocr_text or "",
            filename=document.filename,
            page=image.page,
            image_id=image.image_id,
            before=before,
            after=after,
            anchor_chunk_id=anchor,
            caption=None,
            target_tokens=job.config.chunk_target_tokens,
            max_tokens=job.config.chunk_max_tokens,
            overlap_tokens=job.config.chunk_overlap_tokens,
        )
        for synthetic in routed.chunks:
            created.append(
                Chunk(
                    chunk_id=chunk_id_for(doc_id, ordinal),
                    doc_id=doc_id,
                    ordinal=ordinal,
                    text=synthetic.text,
                    kind=synthetic.kind,
                    token_estimate=synthetic.token_estimate,
                    embed_state=synthetic.embed_state,
                    page_start=image.page,
                    page_end=image.page,
                    image_id=image.image_id,
                    anchor_chunk_id=synthetic.anchor_chunk_id,
                )
            )
            ordinal += 1
    if created:
        logger.info("%s", t("log.stage.image_chunks"))
        job.store.insert_chunks(created)


def _apply_no_vision(job: _Job, doc_id: str) -> None:
    chunks = job.store.chunks_for_document(doc_id)
    for image in job.store.images_for_document(doc_id):
        if image.route not in {ROUTE_LABEL_AND_CAPTION, ROUTE_CAPTION_ONLY}:
            continue
        if image.status in _STRICT_IMAGE_STATUSES or image.status in {
            "caption_done",
            "ocr_done",
        }:
            continue
        owned = [chunk for chunk in chunks if chunk.image_id == image.image_id]
        if image.route == ROUTE_LABEL_AND_CAPTION:
            for chunk in owned:
                if chunk.embed_state == "blocked":
                    job.store.set_embed_state(chunk.chunk_id, "pending")
            if image.status == "pending_caption":
                job.store.update_image(replace(image, status="ocr_done"))
            continue
        for chunk in owned:
            if chunk.embed_state != "done":
                job.store.delete_chunk(chunk.chunk_id)
        job.store.update_image(replace(image, status="empty_no_vision"))
        _note_failure(job, image.image_id, "empty_no_vision")


def _run_vision(job: _Job, doc_id: str) -> None:
    batch = _vision_images(job, doc_id)
    if not batch:
        return
    client = _client(job)
    with ollama_lock(job.config.data_dir, timeout=job.lock_timeout):
        results = describe_batch(
            batch,
            client,
            VisionSettings.from_config(job.config),
            available=job.available,
        )
    for result in results:
        _apply_caption(job, result.image_id, result.status, result.text, result.model)


def _vision_images(job: _Job, doc_id: str) -> list[VisionImage]:
    chunks = job.store.chunks_for_document(doc_id)
    blocked = {chunk.image_id for chunk in chunks if chunk.embed_state == "blocked" and chunk.image_id}
    chosen: list[VisionImage] = []
    for image in job.store.images_for_document(doc_id):
        if image.image_id not in blocked or image.route is None:
            continue
        if not needs_vision(image.route):
            continue
        if image.status in _STRICT_IMAGE_STATUSES or image.status == "caption_done":
            continue
        chosen.append(VisionImage(image.image_id, _read_image(job, doc_id, image.image_id)))
    return chosen


def _apply_caption(
    job: _Job, image_id: str, status: str, text: str | None, model: str | None
) -> None:
    image = job.store.get_image(image_id)
    if image is None:
        raise RagError(EXIT_NOT_FOUND, t("store.image_missing"))
    blocked = [
        chunk
        for chunk in job.store.chunks_for_document(image.doc_id)
        if chunk.image_id == image.image_id and chunk.embed_state == "blocked"
    ]
    paragraph = " ".join(text.split()) if isinstance(text, str) else ""
    if status == CAPTION_DONE and paragraph:
        for chunk in blocked:
            updated = add_caption_line(chunk.text, paragraph)
            estimate = estimate_tokens(updated)
            if estimate > job.config.chunk_max_tokens:
                _fail_caption(job, image, blocked)
                return
            job.store.update_chunk_text(chunk.chunk_id, updated, estimate)
            job.store.set_embed_state(chunk.chunk_id, "pending")
        job.store.update_image(replace(image, caption=paragraph, status="caption_done", error=None))
        job.stats.vision_count += 1
        if model and job.stats.model_name is None:
            job.stats.model_name = model
        return
    _fail_caption(job, image, blocked)


def _fail_caption(job: _Job, image: ImageRecord, blocked: Sequence[Chunk]) -> None:
    if image.route == ROUTE_LABEL_AND_CAPTION:
        for chunk in blocked:
            job.store.set_embed_state(chunk.chunk_id, "pending")
    else:
        for chunk in blocked:
            job.store.delete_chunk(chunk.chunk_id)
    job.store.update_image(replace(image, status="caption_failed", error="描述失敗。"))
    _note_failure(job, image.image_id, "caption_failed")


def _embed(job: _Job, doc_ids: set[str]) -> None:
    chunks = _pending_for(job, doc_ids)
    if not chunks or job.no_embed:
        return
    client = _client(job)
    model, dimensions, num_ctx, native = _embed_target(job)
    if dimensions > native:
        raise RagError(EXIT_MISMATCH, t("ingest.dimensions_exceed_native"))
    job.stats.model_name = model
    body_error: BaseException | None = None
    with ollama_lock(job.config.data_dir, timeout=job.lock_timeout):
        try:
            _reject_generators(job, client)
            _require_disk(job.config, job.free_bytes)
            for batch in _batches(chunks, job.config.embed_batch_size):
                _embed_batch(job, client, batch, model, dimensions, num_ctx, native)
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            _release_embed(client, model, body_error)
    for doc_id in doc_ids:
        job.store.refresh_document_status(doc_id)


def _embed_batch(
    job: _Job,
    client: OllamaClient,
    batch: Sequence[Chunk],
    model: str,
    dimensions: int,
    num_ctx: int,
    native: int,
) -> None:
    try:
        vectors = client.embed(
            model,
            [chunk.text for chunk in batch],
            num_ctx=num_ctx,
        )
    except RagError as exc:
        job.stats.ollama_error_count += 1
        for chunk in batch:
            _note_embed_failure(job, chunk.chunk_id)
        if exc.code in {EXIT_OLLAMA_UNREACHABLE, EXIT_RESOURCES, EXIT_MODEL_MISSING, EXIT_MISMATCH}:
            raise
        return
    if len(vectors) != len(batch) or any(len(vector) != native for vector in vectors):
        for chunk in batch:
            _note_embed_failure(job, chunk.chunk_id)
        return
    _require_disk(job.config, job.free_bytes)
    active = job.store.get_active()
    expected = collection_name(model, dimensions)
    if active is not None and active.collection != expected:
        raise RagError(EXIT_MISMATCH, t("ingest.collection_mismatch"))
    records: list[IndexRecord] = []
    for chunk, vector in zip(batch, vectors, strict=True):
        document = job.store.get_document(chunk.doc_id)
        if document is None:
            raise RagError(EXIT_NOT_FOUND, t("store.document_missing"))
        records.append(_index_record(chunk, document, vector))
    name = job.index.upsert(
        records,
        model=model,
        dimensions=dimensions,
        embed_num_ctx=num_ctx,
    )
    logger.info("%s", t("log.collection", name=name))
    if name != expected:
        raise RagError(EXIT_MISMATCH, t("ingest.collection_mismatch"))
    if active is None:
        job.store.set_active(
            ActiveIndex(
                collection=name,
                embed_model=model,
                dimensions=dimensions,
                embed_num_ctx=num_ctx,
            )
        )
    stamped = _now(job.clock)
    for chunk, vector in zip(batch, vectors, strict=True):
        job.store.upsert_vector(
            _vector_record(chunk.chunk_id, model, native, vector, num_ctx, stamped)
        )
        job.store.set_embed_state(chunk.chunk_id, "done")
        job.stats.chunk_count += 1


def _embed_target(job: _Job) -> tuple[str, int, int, int]:
    active = job.store.get_active()
    if active is None:
        model = job.config.embed_model
        dimensions = job.config.embed_dimensions
        num_ctx = job.config.embed_num_ctx
    else:
        model = active.embed_model
        dimensions = active.dimensions
        num_ctx = active.embed_num_ctx
    return model, dimensions, num_ctx, _native_dim(job.config, model)


def _native_dim(config: Config, model: str) -> int:
    if model == config.embed_model:
        return config.embed_native_dimensions
    if model in _KNOWN_NATIVE:
        return _KNOWN_NATIVE[model]
    raise RagError(EXIT_MISMATCH, t("ingest.native_width_required"))


def _active_embed_model(job: _Job) -> str:
    active = job.store.get_active()
    if active is None:
        return job.config.embed_model
    return active.embed_model


def _reject_generators(job: _Job, client: OllamaClient) -> None:
    names = {
        NINE_B_MODEL,
        FOUR_B_MODEL,
        job.config.generate_model,
        job.config.generate_fallback_model,
    }
    for _attempt in range(2):
        loaded = [item for item in client.ps() if item.name in names]
        if not loaded:
            return
        for item in loaded:
            client.unload_model(item.name)
    if any(item.name in names for item in client.ps()):
        raise RagError(EXIT_RESOURCES, t("ingest.generator_still_loaded"))


def _release_embed(client: OllamaClient, model: str, body_error: BaseException | None) -> None:
    try:
        client.release_stage(model)
    except RagError:
        logger.error("%s", t("log.unload_failed", model=model))
        if body_error is None:
            raise


def _retarget(job: _Job, document: Document, path: Path) -> None:
    active = job.store.get_active()
    if active is None:
        return
    done = [
        chunk
        for chunk in job.store.chunks_for_document(document.doc_id)
        if chunk.embed_state == "done"
    ]
    if not done:
        return
    found = job.index.get_embeddings(
        [chunk.chunk_id for chunk in done],
        model=active.embed_model,
        dimensions=active.dimensions,
        embed_num_ctx=active.embed_num_ctx,
    )
    moved = replace(document, source_path=str(path), filename=path.name)
    records = [
        _index_record(chunk, moved, found[chunk.chunk_id])
        for chunk in done
        if chunk.chunk_id in found
    ]
    if not records:
        return
    job.index.upsert(
        records,
        model=active.embed_model,
        dimensions=active.dimensions,
        embed_num_ctx=active.embed_num_ctx,
    )


def _purge(job: _Job, doc_id: str) -> None:
    if job.store.get_document(doc_id) is None:
        return
    ids = job.store.chunk_ids_for(doc_id)
    if ids:
        job.index.delete(ids)
    job.store.delete_document(doc_id)


def _commit_parse(job: _Job, result: ParseResult) -> None:
    doc_id = result.document.doc_id
    try:
        for image in result.images:
            if image.png_bytes is None:
                continue
            job.store.write_image(doc_id, image.image_id, image.png_bytes)
        job.store.commit_parsed(
            _stored_document(result.document, _now(job.clock)),
            [_stored_chunk(chunk) for chunk in result.chunks],
            [_stored_image(image) for image in result.images],
        )
    except BaseException:
        job.store.remove_image_dir(doc_id)
        raise


def _parse(path: Path, limits: ParseLimits) -> ParseResult:
    from ragcli.parse import parse_file

    return parse_file(path, limits)


def _pending_for(job: _Job, doc_ids: set[str]) -> list[Chunk]:
    current = {
        document.doc_id
        for document in job.store.list_documents()
        if document.parser_version == PARSER_VERSION
    }
    allowed = set(doc_ids) & current
    return [chunk for chunk in job.store.pending_embed_chunks() if chunk.doc_id in allowed]


def _result_code(job: _Job, doc_ids: Sequence[str]) -> int:
    if not doc_ids:
        return EXIT_OK
    total = sum(len(job.store.chunks_for_document(doc_id)) for doc_id in doc_ids)
    if total == 0 or (job.strict and _has_strict_failure(job, doc_ids)):
        return EXIT_NO_CHUNKS
    return EXIT_OK


def _has_strict_failure(job: _Job, doc_ids: Sequence[str]) -> bool:
    if job.stats.failures or _stored_strict_failure(job, doc_ids):
        return True
    return False


def _stored_strict_failure(job: _Job, doc_ids: Sequence[str]) -> bool:
    for doc_id in doc_ids:
        for image in job.store.images_for_document(doc_id):
            if image.status in _STRICT_IMAGE_STATUSES:
                return True
    return False


def _guard_no_ocr(job: _Job, images: Sequence[ParsedImage | ImageRecord]) -> None:
    if not job.no_ocr:
        return
    live = [image for image in images if not _is_skipped(image)]
    if len(live) > NO_OCR_IMAGE_LIMIT:
        raise RagError(EXIT_USAGE, t("ingest.no_ocr_image_limit"))


def _is_skipped(image: ParsedImage | ImageRecord) -> bool:
    route = image.route or ""
    return route.startswith("skipped_") or image.status in {"skipped", "skipped_bomb"}


def _context(
    text_chunks: Sequence[Chunk], image: ImageRecord, preceding: int | None
) -> tuple[str, str, str | None]:
    """Pick the text before and after an image.

    ``preceding`` is the parser's text-chunk count and exists only for a
    parse that just committed. Resume does not have that count stored.
    """
    if preceding is None:
        return "", "", None
    ordered = sorted(text_chunks, key=lambda chunk: chunk.ordinal)
    before = ordered[preceding - 1] if 0 < preceding <= len(ordered) else None
    after = ordered[preceding] if 0 <= preceding < len(ordered) else None
    return (
        before.text if before is not None else "",
        after.text if after is not None else "",
        before.chunk_id if before is not None else None,
    )


def _index_record(chunk: Chunk, document: Document, vector: Sequence[float]) -> IndexRecord:
    return IndexRecord(
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


def _vector_record(
    chunk_id: str,
    model: str,
    native: int,
    vector: Sequence[float],
    num_ctx: int,
    stamped: str,
) -> VectorRecord:
    return VectorRecord(
        chunk_id=chunk_id,
        embed_model=model,
        native_dim=native,
        vector=tuple(float(value) for value in vector),
        num_ctx=num_ctx,
        embedded_at=stamped,
    )


def _stored_document(document: ParsedDocument, ingested_at: str) -> Document:
    return Document(
        doc_id=document.doc_id,
        source_path=document.source_path,
        filename=document.filename,
        media_type=document.media_type,
        byte_size=document.byte_size,
        sha256=document.sha256,
        parser=document.parser,
        parser_version=document.parser_version,
        ingested_at=ingested_at,
        status="parsed",
        title=document.title,
        page_count=document.page_count,
    )


def _stored_chunk(chunk: ParsedChunk) -> Chunk:
    return Chunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        ordinal=chunk.ordinal,
        text=chunk.text,
        kind=chunk.kind,
        token_estimate=chunk.token_estimate,
        embed_state=chunk.embed_state,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section=chunk.section,
    )


def _stored_image(image: ParsedImage) -> ImageRecord:
    return ImageRecord(
        image_id=image.image_id,
        doc_id=image.doc_id,
        storage_relpath=image.storage_relpath,
        width=image.width,
        height=image.height,
        byte_size=image.byte_size,
        sha256=image.sha256,
        kind=image.kind,
        status=image.status,
        page=image.page,
        route=image.route,
        duplicate_count=image.duplicate_count,
        error=image.error,
    )


def _limits(job: _Job) -> ParseLimits:
    return ParseLimits(
        max_pages=job.max_pages,
        max_bytes=job.max_bytes,
        max_pixels=job.config.max_pixels,
        max_images=job.config.max_images,
    )


def _check_file(path: Path, max_bytes: int) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise RagError(EXIT_NOT_FOUND, t("io.file_unreadable", path=path)) from exc
    except OSError as exc:
        raise RagError(EXIT_NOT_FOUND, t("io.file_unreadable", path=path)) from exc
    if not _readable_file(resolved):
        if resolved.is_dir():
            raise RagError(EXIT_USAGE, t("usage.file_not_directory"))
        raise RagError(EXIT_NOT_FOUND, t("io.file_unreadable", path=resolved))
    suffix = resolved.suffix.lower()
    if suffix not in _SUFFIXES:
        label = suffix if suffix else t("usage.no_suffix")
        raise RagError(EXIT_UNSUPPORTED_TYPE, t("usage.suffix_unsupported", label=label))
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise RagError(EXIT_NOT_FOUND, t("io.file_unreadable", path=resolved)) from exc
    if size > max_bytes:
        raise RagError(EXIT_DOCUMENT, t("parse.bytes_exceeded"))
    return resolved


def _readable_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and os.access(path, os.R_OK)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_disk(config: Config, free_bytes: Callable[[], int] | None) -> None:
    if free_bytes is None:
        probe = config.data_dir if config.data_dir.exists() else config.data_dir.parent
        free = shutil.disk_usage(probe).free
    else:
        free = free_bytes()
    if isinstance(free, bool) or not isinstance(free, int):
        raise RagError(EXIT_DISK, t("disk.free_unreadable"))
    if free < config.min_free_bytes:
        raise RagError(
            EXIT_DISK,
            t("disk.free_below", free=free, minimum=config.min_free_bytes),
        )


def _read_image(job: _Job, doc_id: str, image_id: str) -> bytes:
    path = job.store.image_path(doc_id, image_id)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise RagError(EXIT_UNCLASSIFIED, t("image.unreadable")) from exc


def _note_failure(job: _Job, image_id: str, status: str) -> None:
    job.stats.failures.append((image_id, status))
    logger.warning("%s", t("image.status", image_id=image_id, status=status))
    print(t("image.status", image_id=image_id, status=status), file=_stderr())


def _note_embed_failure(job: _Job, chunk_id: str) -> None:
    job.stats.failures.append((chunk_id, "embed"))
    logger.warning("%s", t("ingest.chunk_embed_failed", chunk_id=chunk_id))
    print(t("ingest.chunk_embed_failed", chunk_id=chunk_id), file=_stderr())


def _client(job: _Job) -> OllamaClient:
    if job.client is None:
        job.client = OllamaClient(job.config.ollama_base_url)
    return job.client


def _batches(chunks: Sequence[Chunk], size: int) -> list[list[Chunk]]:
    return [list(chunks[start : start + size]) for start in range(0, len(chunks), size)]


def _now(clock: Callable[[], str] | None) -> str:
    if clock is not None:
        return clock()
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _begin_run(store: SqliteStore, clock: Callable[[], str] | None) -> str:
    run_id = uuid.uuid4().hex
    store.start_run(Run(run_id=run_id, command="ingest", started_at=_now(clock)))
    return run_id


def _finish_run(job: _Job, run_id: str, code: int) -> None:
    logger.info("%s", t("log.exit_code", code=code))
    job.store.finish_run(
        run_id,
        finished_at=_now(job.clock),
        exit_code=code,
        chunk_count=job.stats.chunk_count,
        ocr_count=job.stats.ocr_count,
        vision_count=job.stats.vision_count,
        model_name=job.stats.model_name,
        ollama_error_count=job.stats.ollama_error_count,
    )


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


def _stderr():
    import sys

    return sys.stderr
