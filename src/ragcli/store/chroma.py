"""Chroma index for vectors the caller already computed.

SQLite remains the authority for chunk text. This module only writes and
queries vectors. Native width is stored as given, with no L2 norm. A shorter
collection width uses ``truncate_normalize`` for both documents and queries.
Queries pass ``query_embeddings`` only. The default MiniLM function is never
attached, and the client is opened with telemetry off.

Collection names follow the check still shipped in Chroma 1.5.9: 3 to 63
characters, ``^[a-z0-9][a-z0-9._-]*[a-z0-9]$``, and no ``..``. A colon in a
model tag never reaches Chroma.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragcli.errors import (
    EXIT_MISMATCH,
    EXIT_NO_COLLECTION,
    EXIT_UNCLASSIFIED,
    EXIT_USAGE,
    RagError,
)
from ragcli.i18n import t
from ragcli.store.sqlite import CHUNK_KINDS, chunk_id_for

CHROMA_DIRNAME = "chroma"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*[a-z0-9]$")
_DOC_ID = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID = re.compile(r"^[0-9a-f]{12}:img:[0-9]{4}$")
_MAX_NAME = 63
_MAX_DIMENSIONS = 100_000
_MAX_NUM_CTX = 1_000_000
_MAX_PAGE = 1_000_000
_PRIVATE_DIR = 0o700
_LIST_PAGE = 100

_METADATA_KEYS = (
    "doc_id",
    "source_path",
    "filename",
    "page_start",
    "page_end",
    "kind",
    "image_id",
    "ordinal",
)

ClientFactory = Callable[[Path, Any], Any]


@dataclass(frozen=True, slots=True)
class ClientSettings:
    """Settings the index passes to a client factory. Telemetry stays off."""

    anonymized_telemetry: bool = False


@dataclass(frozen=True, slots=True)
class IndexRecord:
    """One chunk to upsert. Pages use -1 when the source has none."""

    chunk_id: str
    doc_id: str
    text: str
    embedding: tuple[float, ...]
    source_path: str
    filename: str
    kind: str
    ordinal: int
    page_start: int = -1
    page_end: int = -1
    image_id: str = ""


@dataclass(frozen=True, slots=True)
class CollectionInfo:
    """Metadata of one existing collection. The name already matched it."""

    name: str
    embed_model: str
    dimensions: int
    embed_num_ctx: int


@dataclass(frozen=True, slots=True)
class QueryHit:
    """One neighbor. ``distance`` is cosine distance, smaller is closer."""

    chunk_id: str
    text: str
    distance: float
    doc_id: str
    source_path: str
    filename: str
    page_start: int
    page_end: int
    kind: str
    image_id: str
    ordinal: int


def collection_name(model: str, dimensions: int) -> str:
    """Return ``emb-{slug}-d{dimensions}`` for this model and width.

    The slug is the stripped tag, lowercased, with ``:`` and ``/`` turned
    into hyphens. Characters outside ``[a-z0-9._-]`` are removed, repeated
    hyphens collapse, and repeated dots collapse. A name that would pass 63
    characters, or a tag that slugs to nothing, keeps a head and a tail of
    the slug with the first 8 hex digits of SHA-256(the stripped tag)
    between them.
    """
    raw = _model_tag(model)
    width = _dimensions(dimensions)
    slug = _model_slug(raw)
    suffix = f"-d{width}"
    direct = f"emb-{slug}{suffix}" if slug else ""
    if slug and _acceptable_name(direct):
        return direct
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    room = _MAX_NAME - len("emb-") - len(suffix)
    if room < len(digest):
        raise ValueError("集合名稱放不下維度。")
    name = f"emb-{_hashed_body(slug, digest, room)}{suffix}"
    if not _acceptable_name(name):
        raise ValueError("集合名稱不合法。")
    return name


def truncate_normalize(vec: Sequence[float], dim: int) -> list[float]:
    """Keep the first ``dim`` components and scale them to L2 norm 1."""
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        raise ValueError("dim 必須是正整數。")
    numbers = _finite_floats(vec)
    if dim > len(numbers):
        raise ValueError("dim 大於向量長度。")
    head = numbers[:dim]
    norm = math.sqrt(sum(value * value for value in head))
    if norm == 0.0:
        raise ValueError("zero vector")
    return [value / norm for value in head]


def embedding_for_width(vec: Sequence[float], dimensions: int) -> list[float]:
    """Fit a native vector to a collection width.

    The same width is returned unchanged, with no L2 norm. A shorter width
    uses ``truncate_normalize``. A wider request cannot invent components.
    """
    try:
        numbers = _finite_floats(vec)
        width = _dimensions(dimensions)
    except ValueError as exc:
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.embedding_invalid")) from exc
    if len(numbers) == width:
        return numbers
    if len(numbers) < width:
        raise RagError(EXIT_MISMATCH, t("chroma.vector_shorter"))
    try:
        return truncate_normalize(numbers, width)
    except ValueError as exc:
        raise RagError(EXIT_MISMATCH, t("chroma.vector_zero")) from exc


class ChromaIndex:
    """Persistent Chroma directory under ``data/chroma``."""

    def __init__(self, data_dir: Path, chroma_dir: Path, client: Any) -> None:
        self.data_dir = data_dir
        self.chroma_dir = chroma_dir
        self._client: Any = client
        self._collections: dict[str, Any] = {}

    @classmethod
    def open(cls, data_dir: Path, *, client_factory: ClientFactory | None = None) -> ChromaIndex:
        """Create ``data/chroma`` as 0700 when it is missing, then open it.

        An existing directory keeps the mode the user left it with. Symlinks
        are refused. The default factory is Chroma's persistent client with
        telemetry disabled. Tests can pass a factory and never import Chroma.
        """
        root = _private_root(data_dir)
        _ensure_private_dir(root)
        chroma_dir = root / CHROMA_DIRNAME
        _ensure_private_dir(chroma_dir)
        settings = ClientSettings(anonymized_telemetry=False)
        factory = client_factory or _default_client_factory
        client = factory(chroma_dir, settings)
        return cls(root, chroma_dir, client)

    def close(self) -> None:
        client = self._client
        self._client = None
        self._collections.clear()
        if client is not None and hasattr(client, "close"):
            client.close()

    def __enter__(self) -> ChromaIndex:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def upsert(
        self,
        records: Sequence[IndexRecord],
        *,
        model: str,
        dimensions: int,
        embed_num_ctx: int,
    ) -> str:
        """Insert or replace rows. Returns the collection name.

        Embeddings are prepared with ``embedding_for_width``. An empty
        sequence still ensures the collection exists.
        """
        prepared = _prepare_records(records, dimensions)
        collection = self._collection(model, dimensions, embed_num_ctx)
        if not prepared:
            return collection.name
        size = _batch_size(self._client_open())
        for start in range(0, len(prepared), size):
            batch = prepared[start : start + size]
            collection.upsert(
                ids=[item[0] for item in batch],
                embeddings=[item[1] for item in batch],
                documents=[item[2] for item in batch],
                metadatas=[item[3] for item in batch],
            )
        return collection.name

    def query(
        self,
        embedding: Sequence[float],
        *,
        model: str,
        dimensions: int,
        embed_num_ctx: int,
        n_results: int,
    ) -> list[QueryHit]:
        """Return nearest chunks. The query vector is fitted like a document."""
        if isinstance(n_results, bool) or not isinstance(n_results, int) or n_results < 1:
            raise RagError(EXIT_USAGE, t("chroma.n_results"))
        vector = embedding_for_width(embedding, dimensions)
        collection = self._existing_collection(model, dimensions, embed_num_ctx)
        if collection.count() == 0:
            return []
        result = collection.query(
            query_embeddings=[vector],
            n_results=n_results,
            include=["metadatas", "documents", "distances"],
        )
        return _hits(result)

    def delete(
        self,
        ids: Sequence[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        embed_num_ctx: int | None = None,
    ) -> None:
        """Delete chunk ids.

        A model, width, and ``embed_num_ctx`` together select one collection.
        Omitting all three deletes those ids from every collection. Missing
        ids are ignored. An unknown single collection is exit 11.
        """
        targets = _id_list(ids)
        if not targets:
            return
        if model is None and dimensions is None and embed_num_ctx is None:
            for name in self.collection_names():
                collection = self._client_open().get_collection(
                    name=name,
                    embedding_function=None,
                )
                collection.delete(ids=targets)
            return
        if model is None or dimensions is None or embed_num_ctx is None:
            raise RagError(EXIT_USAGE, t("chroma.delete_needs_all"))
        collection = self._existing_collection(model, dimensions, embed_num_ctx)
        collection.delete(ids=targets)

    def count(self, *, model: str, dimensions: int, embed_num_ctx: int) -> int:
        collection = self._existing_collection(model, dimensions, embed_num_ctx)
        return int(collection.count())

    def get_embeddings(
        self,
        ids: Sequence[str],
        *,
        model: str,
        dimensions: int,
        embed_num_ctx: int,
    ) -> dict[str, tuple[float, ...]]:
        """Return stored vectors for ids that are present. Missing ids are absent."""
        targets = _id_list(ids)
        if not targets:
            return {}
        collection = self._existing_collection(model, dimensions, embed_num_ctx)
        result = collection.get(ids=targets, include=["embeddings"])
        found_ids = _as_rows(result.get("ids")) if isinstance(result, dict) else None
        found_vectors = _as_rows(result.get("embeddings")) if isinstance(result, dict) else None
        if found_ids is None or found_vectors is None or len(found_ids) != len(found_vectors):
            raise RagError(EXIT_UNCLASSIFIED, t("chroma.vector_shape"))
        stored: dict[str, tuple[float, ...]] = {}
        for chunk_id, vector in zip(found_ids, found_vectors, strict=True):
            if vector is None:
                raise RagError(EXIT_MISMATCH, t("chroma.vector_missing"))
            stored[str(chunk_id)] = tuple(float(value) for value in vector)
        return stored

    def describe(self, name: str) -> CollectionInfo:
        """Return metadata for one collection.

        A missing collection is exit 11. Metadata that does not produce this
        name, or a collection that breaks the cosine and no-embedding-function
        contract, is exit 12.
        """
        collection = self._open_named(name)
        metadata = getattr(collection, "metadata", None)
        if not isinstance(metadata, dict):
            raise RagError(EXIT_MISMATCH, t("chroma.metadata_incomplete"))
        model = metadata.get("embed_model")
        dimensions = _optional_int(metadata.get("dimensions"))
        num_ctx = _optional_int(metadata.get("embed_num_ctx"))
        if not isinstance(model, str) or model.strip() == "" or dimensions is None or num_ctx is None:
            raise RagError(EXIT_MISMATCH, t("chroma.metadata_incomplete"))
        try:
            expected = collection_name(model, dimensions)
        except ValueError as exc:
            raise RagError(EXIT_MISMATCH, t("chroma.name_model")) from exc
        if expected != name or getattr(collection, "name", None) != name:
            raise RagError(EXIT_MISMATCH, t("chroma.name_metadata"))
        spec = _collection_spec(model, dimensions, num_ctx)
        _require_contract(collection, spec)
        return CollectionInfo(
            name=name,
            embed_model=spec.model,
            dimensions=spec.dimensions,
            embed_num_ctx=spec.embed_num_ctx,
        )

    def drop(self, name: str) -> None:
        """Delete one collection by name. A missing collection is already gone."""
        client = self._client_open()
        try:
            client.delete_collection(name)
        except Exception as exc:
            if _is_missing_collection(exc):
                self._collections.pop(name, None)
                return
            raise RagError(EXIT_UNCLASSIFIED, t("chroma.drop_failed")) from exc
        self._collections.pop(name, None)

    def chunk_ids(self, name: str) -> list[str]:
        """Return every id in one collection. A missing collection is exit 11."""
        collection = self._open_named(name)
        found: list[str] = []
        seen: set[str] = set()
        offset = 0
        while True:
            result = collection.get(include=[], limit=_LIST_PAGE, offset=offset)
            page = _ids_of(result)
            fresh = [item for item in page if item not in seen]
            if not fresh:
                break
            seen.update(fresh)
            found.extend(fresh)
            if len(page) < _LIST_PAGE:
                break
            offset += len(page)
        return found

    def collection_names(self) -> list[str]:
        client = self._client_open()
        names: list[str] = []
        offset = 0
        while True:
            page = list(client.list_collections(limit=_LIST_PAGE, offset=offset))
            if not page:
                break
            names.extend(str(collection.name) for collection in page)
            if len(page) < _LIST_PAGE:
                break
            offset += _LIST_PAGE
        return sorted(names)

    def _open_named(self, name: str) -> Any:
        if not isinstance(name, str) or name.strip() == "" or "\x00" in name:
            raise RagError(EXIT_MISMATCH, t("chroma.name_metadata"))
        cached = self._collections.get(name)
        if cached is not None:
            return cached
        try:
            collection = self._client_open().get_collection(
                name=name,
                embedding_function=None,
            )
        except Exception as exc:
            if _is_missing_collection(exc):
                raise RagError(EXIT_NO_COLLECTION, t("chroma.collection_missing")) from exc
            raise RagError(EXIT_MISMATCH, t("chroma.read_failed")) from exc
        self._collections[name] = collection
        return collection

    def _collection(self, model: str, dimensions: int, embed_num_ctx: int) -> Any:
        spec = _collection_spec(model, dimensions, embed_num_ctx)
        cached = self._collections.get(spec.name)
        if cached is not None:
            return cached
        collection = self._client_open().get_or_create_collection(
            name=spec.name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata=spec.metadata,
        )
        _require_contract(collection, spec)
        self._collections[spec.name] = collection
        return collection

    def _existing_collection(self, model: str, dimensions: int, embed_num_ctx: int) -> Any:
        spec = _collection_spec(model, dimensions, embed_num_ctx)
        cached = self._collections.get(spec.name)
        if cached is not None:
            return cached
        try:
            collection = self._client_open().get_collection(
                name=spec.name,
                embedding_function=None,
            )
        except Exception as exc:
            if _is_missing_collection(exc):
                raise RagError(EXIT_NO_COLLECTION, t("chroma.collection_missing")) from exc
            raise
        _require_contract(collection, spec)
        self._collections[spec.name] = collection
        return collection

    def _client_open(self) -> Any:
        if self._client is None:
            raise RagError(EXIT_UNCLASSIFIED, t("chroma.closed"))
        return self._client


@dataclass(frozen=True, slots=True)
class _CollectionSpec:
    name: str
    model: str
    dimensions: int
    embed_num_ctx: int
    metadata: dict[str, Any]


def _default_client_factory(path: Path, settings: Any) -> Any:
    if getattr(settings, "anonymized_telemetry", True) is not False:
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.telemetry"))
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )


def _collection_spec(model: str, dimensions: int, embed_num_ctx: int) -> _CollectionSpec:
    try:
        raw = _model_tag(model)
        width = _dimensions(dimensions)
        name = collection_name(raw, width)
    except ValueError as exc:
        raise RagError(EXIT_MISMATCH, t("chroma.collection_spec_invalid")) from exc
    num_ctx = _num_ctx(embed_num_ctx)
    return _CollectionSpec(
        name=name,
        model=raw,
        dimensions=width,
        embed_num_ctx=num_ctx,
        metadata={
            "embed_model": raw,
            "dimensions": width,
            "embed_num_ctx": num_ctx,
        },
    )


def _index_configuration(collection: Any) -> dict[str, Any] | None:
    """Return the raw collection configuration.

    ``configuration`` reloads the JSON and warns when Chroma 1.5.9 stored
    ``embedding_function=None`` as ``{"type": "legacy"}``. The raw JSON has
    the same HNSW space and does not emit that warning.
    """
    raw = getattr(collection, "configuration_json", None)
    if isinstance(raw, dict):
        return raw
    configuration = getattr(collection, "configuration", None)
    if isinstance(configuration, dict):
        return configuration
    return None


def _embedding_function_attached(value: object) -> bool:
    if value is None:
        return False
    # 1.5.9 persists a missing embedding function as this legacy marker.
    if isinstance(value, dict):
        return value.get("type") not in {None, "legacy"}
    return True


def _require_contract(collection: Any, spec: _CollectionSpec) -> None:
    if getattr(collection, "name", None) != spec.name or ":" in spec.name:
        raise RagError(EXIT_MISMATCH, t("chroma.name_model"))
    configuration = _index_configuration(collection)
    hnsw = configuration.get("hnsw") if isinstance(configuration, dict) else None
    space = hnsw.get("space") if isinstance(hnsw, dict) else None
    if space != "cosine":
        raise RagError(EXIT_MISMATCH, t("chroma.space_cosine"))
    stored_function = None if configuration is None else configuration.get("embedding_function")
    if _embedding_function_attached(stored_function):
        raise RagError(EXIT_MISMATCH, t("chroma.embedding_function"))
    if getattr(collection, "_embedding_function", None) is not None:
        raise RagError(EXIT_MISMATCH, t("chroma.embedding_function"))
    metadata = getattr(collection, "metadata", None)
    if not isinstance(metadata, dict):
        raise RagError(EXIT_MISMATCH, t("chroma.metadata_incomplete"))
    if metadata.get("embed_model") != spec.model:
        raise RagError(EXIT_MISMATCH, t("chroma.model_mismatch"))
    if _optional_int(metadata.get("dimensions")) != spec.dimensions:
        raise RagError(EXIT_MISMATCH, t("chroma.dimensions_mismatch"))
    if _optional_int(metadata.get("embed_num_ctx")) != spec.embed_num_ctx:
        raise RagError(EXIT_MISMATCH, t("chroma.num_ctx_mismatch"))


def _prepare_records(
    records: Sequence[IndexRecord], dimensions: int
) -> list[tuple[str, list[float], str, dict[str, Any]]]:
    if isinstance(records, (str, bytes, IndexRecord)) or not isinstance(records, Sequence):
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.upsert_records"))
    try:
        width = _dimensions(dimensions)
    except ValueError as exc:
        raise RagError(EXIT_MISMATCH, t("chroma.upsert_dimensions_invalid")) from exc
    prepared: list[tuple[str, list[float], str, dict[str, Any]]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, IndexRecord):
            raise RagError(EXIT_UNCLASSIFIED, t("chroma.upsert_record_type"))
        item = _prepare_record(record, width)
        if item[0] in seen:
            raise RagError(EXIT_UNCLASSIFIED, t("chroma.chunk_id_duplicate"))
        seen.add(item[0])
        prepared.append(item)
    return prepared


def _prepare_record(
    record: IndexRecord, dimensions: int
) -> tuple[str, list[float], str, dict[str, Any]]:
    doc_id = _require_doc_id(record.doc_id)
    ordinal = _require_ordinal(record.ordinal)
    chunk_id = _require_text(record.chunk_id, "chunk_id")
    if chunk_id != chunk_id_for(doc_id, ordinal):
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.chunk_id_mismatch"))
    text = _require_chunk_text(record.text)
    if record.kind not in CHUNK_KINDS:
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.kind"))
    source_path = _require_text(record.source_path, "source_path")
    filename = _require_filename(record.filename)
    page_start, page_end = _pages(record.page_start, record.page_end)
    image_id = _image_id(record.image_id, doc_id)
    vector = embedding_for_width(record.embedding, dimensions)
    metadata = {
        "doc_id": doc_id,
        "source_path": source_path,
        "filename": filename,
        "page_start": page_start,
        "page_end": page_end,
        "kind": record.kind,
        "image_id": image_id,
        "ordinal": ordinal,
    }
    if any(value is None for value in metadata.values()):
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.metadata_none"))
    return chunk_id, vector, text, metadata


def _hits(result: Any) -> list[QueryHit]:
    if not isinstance(result, dict):
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.query_shape"))
    if result.get("query_texts") not in (None, []):
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.query_texts"))
    ids = _query_column(result, "ids")
    documents = _query_column(result, "documents")
    metadatas = _query_column(result, "metadatas")
    distances = _query_column(result, "distances")
    width = len(ids)
    if len(documents) != width or len(metadatas) != width or len(distances) != width:
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.query_shape"))
    hits: list[QueryHit] = []
    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances, strict=True):
        hits.append(_hit(chunk_id, text, metadata, distance))
    return hits


def _hit(chunk_id: object, text: object, metadata: object, distance: object) -> QueryHit:
    if not isinstance(metadata, dict) or not set(_METADATA_KEYS) <= set(metadata):
        raise RagError(EXIT_MISMATCH, t("chroma.hit_metadata_incomplete"))
    if any(metadata[key] is None for key in _METADATA_KEYS):
        raise RagError(EXIT_MISMATCH, t("chroma.hit_metadata_none"))
    page_start = _optional_int(metadata["page_start"])
    page_end = _optional_int(metadata["page_end"])
    ordinal = _optional_int(metadata["ordinal"])
    if page_start is None or page_end is None or ordinal is None:
        raise RagError(EXIT_MISMATCH, t("chroma.hit_metadata_type"))
    number = _finite_distance(distance)
    kind = metadata["kind"]
    if kind not in CHUNK_KINDS:
        raise RagError(EXIT_MISMATCH, t("chroma.hit_kind"))
    return QueryHit(
        chunk_id=str(chunk_id),
        text=_require_chunk_text(text),
        distance=number,
        doc_id=_require_doc_id(metadata["doc_id"]),
        source_path=_require_text(metadata["source_path"], "source_path"),
        filename=_require_filename(metadata["filename"]),
        page_start=page_start,
        page_end=page_end,
        kind=str(kind),
        image_id=_image_id(metadata["image_id"], _require_doc_id(metadata["doc_id"])),
        ordinal=ordinal,
    )


def _query_column(result: dict[str, Any], key: str) -> list[Any]:
    rows = _as_rows(result.get(key))
    if rows is None or len(rows) != 1:
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.query_shape"))
    inner = _as_rows(rows[0])
    if inner is None:
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.query_shape"))
    return inner


def _ids_of(result: object) -> list[str]:
    if not isinstance(result, dict):
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.id_shape"))
    rows = _as_rows(result.get("ids"))
    if rows is None:
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.id_shape"))
    return [str(item) for item in rows]


def _as_rows(value: object) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    to_list = getattr(value, "tolist", None)
    if not callable(to_list):
        return None
    converted = to_list()
    if isinstance(converted, list):
        return converted
    return None


def _batch_size(client: Any) -> int:
    size = client.get_max_batch_size()
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.batch_size"))
    return size


def _id_list(ids: Sequence[str]) -> list[str]:
    if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.delete_ids"))
    cleaned: list[str] = []
    seen: set[str] = set()
    for chunk_id in ids:
        text = _require_text(chunk_id, "chunk_id")
        if text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _is_missing_collection(exc: BaseException) -> bool:
    if type(exc).__name__ in {"NotFoundError", "NotFound"}:
        return True
    text = str(exc).lower()
    return "does not exist" in text or "not found" in text


def _model_tag(model: str) -> str:
    if not isinstance(model, str):
        raise ValueError("嵌入模型標籤必須是字串。")
    raw = model.strip()
    if raw == "" or "\x00" in raw:
        raise ValueError("嵌入模型標籤不可為空。")
    return raw


def _dimensions(dimensions: int) -> int:
    if (
        isinstance(dimensions, bool)
        or not isinstance(dimensions, int)
        or dimensions < 1
        or dimensions > _MAX_DIMENSIONS
    ):
        raise ValueError("dimensions 必須是 1 到 100000 的整數。")
    return dimensions


def _num_ctx(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > _MAX_NUM_CTX:
        raise RagError(EXIT_MISMATCH, t("chroma.num_ctx_range"))
    return value


def _model_slug(raw: str) -> str:
    text = raw.lower().replace(":", "-").replace("/", "-")
    text = re.sub(r"[^a-z0-9._-]", "", text)
    text = re.sub(r"-{2,}", "-", text)
    text = re.sub(r"\.{2,}", ".", text)
    return text.strip(".-")


def _hashed_body(slug: str, digest: str, room: int) -> str:
    extra = room - len(digest)
    if extra < 2 or not slug:
        return digest
    side = extra - 2
    left = _trim_edge(slug[: side // 2])
    right_len = side - (side // 2)
    right = _trim_edge(slug[-right_len:] if right_len else "")
    parts = [part for part in (left, digest, right) if part]
    body = "-".join(parts)
    body = re.sub(r"-{2,}", "-", body)
    body = re.sub(r"\.{2,}", ".", body).strip(".-")
    if not body or len(body) > room or ".." in body or not _NAME_RE.fullmatch(body):
        return digest
    return body


def _trim_edge(text: str) -> str:
    return text.strip(".-")


def _acceptable_name(name: str) -> bool:
    return (
        3 <= len(name) <= _MAX_NAME
        and ".." not in name
        and ":" not in name
        and _NAME_RE.fullmatch(name) is not None
    )


def _finite_floats(values: Sequence[float]) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("向量必須是數字序列。")
    numbers: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("向量必須是數字序列。")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("向量不能包含非有限數字。")
        numbers.append(number)
    if not numbers:
        raise ValueError("向量不能是空的。")
    return numbers


def _finite_distance(value: object) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.distance_finite"))
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.distance_finite")) from None
    if not math.isfinite(number):
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.distance_finite"))
    return number


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or isinstance(value, (str, bytes, float)):
        return None
    if isinstance(value, int):
        return value
    # Chroma may hand back a NumPy integer. Those are not ``int`` subclasses.
    if type(value).__name__.startswith(("int", "uint")):
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    return None


def _require_doc_id(value: object) -> str:
    if not isinstance(value, str) or _DOC_ID.fullmatch(value) is None:
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.doc_id"))
    return value


def _require_ordinal(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 999_999:
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.ordinal_range"))
    return value


def _require_chunk_text(value: object) -> str:
    if not isinstance(value, str) or value == "" or "\x00" in value:
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.chunk_text_nonempty"))
    return value


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or value == "" or "\x00" in value:
        raise RagError(EXIT_UNCLASSIFIED, t("config.nonempty_string", label=label))
    return value


def _require_filename(value: object) -> str:
    filename = _require_text(value, "filename")
    if "/" in filename or "\\" in filename or filename in {".", ".."}:
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.filename"))
    return filename


def _pages(page_start: object, page_end: object) -> tuple[int, int]:
    start = _optional_int(page_start)
    end = _optional_int(page_end)
    if start is None or end is None:
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.page_int"))
    absent = (start == -1, end == -1)
    if absent == (True, True):
        return -1, -1
    if any(absent):
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.page_pair"))
    if start < 1 or end < 1 or start > _MAX_PAGE or end > _MAX_PAGE or end < start:
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.page_range"))
    return start, end


def _image_id(value: object, doc_id: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise RagError(EXIT_UNCLASSIFIED, t("chroma.image_id_type"))
    if value == "":
        return ""
    if _IMAGE_ID.fullmatch(value) is None or not value.startswith(doc_id[:12] + ":"):
        raise RagError(EXIT_UNCLASSIFIED, t("store.image_id_document"))
    return value


def _private_root(data_dir: Path) -> Path:
    if not isinstance(data_dir, Path):
        raise RagError(EXIT_UNCLASSIFIED, t("lock.data_dir_type"))
    if data_dir.is_symlink():
        raise RagError(EXIT_UNCLASSIFIED, t("io.data_dir_symlink"))
    return data_dir.expanduser().resolve()


def _ensure_private_dir(path: Path) -> None:
    """Create ``path`` as 0700. An existing directory keeps its mode."""
    if path.is_symlink():
        raise RagError(EXIT_UNCLASSIFIED, t("io.data_dir_symlink"))
    if path.exists():
        if not path.is_dir():
            raise RagError(EXIT_UNCLASSIFIED, t("io.data_path_not_directory", name=path.name))
        return
    try:
        path.mkdir(mode=_PRIVATE_DIR)
    except FileExistsError:
        if path.is_symlink() or not path.is_dir():
            raise RagError(EXIT_UNCLASSIFIED, t("io.data_dir_symlink")) from None
        return
    os.chmod(path, _PRIVATE_DIR)


__all__ = [
    "CHROMA_DIRNAME",
    "ChromaIndex",
    "ClientSettings",
    "CollectionInfo",
    "IndexRecord",
    "QueryHit",
    "collection_name",
    "embedding_for_width",
    "truncate_normalize",
]
