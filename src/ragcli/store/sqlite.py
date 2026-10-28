"""SQLite corpus store.

Page numbers and image ids are NULL when absent. The Chroma convention of -1
and an empty image id belongs to the index layer, not these tables. This
module does not parse documents and does not call Ollama.

New data directories are mode 0700. A database this process creates is mode
0600. Existing paths keep the mode the user left them with. WAL sidecars are
tightened to 0600 because SQLite recreates them. A schema newer than
``schema_version`` 1 is refused and is not rewritten.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import sqlite3
import struct
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ragcli.errors import EXIT_MISMATCH, EXIT_NOT_FOUND, EXIT_UNCLASSIFIED, RagError
from ragcli.i18n import t

SCHEMA_VERSION = 1
DB_FILENAME = "corpus.sqlite"

DOCUMENT_STATUSES = frozenset({"parsed", "complete"})
EMBED_STATES = frozenset({"pending", "blocked", "done"})
CHUNK_KINDS = frozenset({"text", "ocr", "figure", "caption"})
MEDIA_TYPES = frozenset({"pdf", "docx", "pptx", "text", "image"})
PARSERS = frozenset({"pymupdf", "python-docx", "python-pptx", "text", "image"})
IMAGE_KINDS = frozenset({"embedded", "page_render", "vector_render", "standalone"})
IMAGE_ROUTES = frozenset(
    {
        "text_rich",
        "label_and_caption",
        "caption_only",
        "skipped_small",
        "skipped_duplicate",
        "skipped_bomb",
    }
)
IMAGE_STATUSES = frozenset(
    {
        "pending_ocr",
        "ocr_done",
        "pending_caption",
        "caption_done",
        "empty",
        "caption_failed",
        "skipped",
        "empty_no_vision",
        "skipped_bomb",
    }
)
RUN_COMMANDS = frozenset({"ingest", "ask", "reembed", "delete", "status", "models"})
SETTING_KEYS = (
    "active_collection",
    "active_embed_model",
    "active_dimensions",
    "active_embed_num_ctx",
)

_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "schema_migrations": ("version",),
    "documents": (
        "doc_id",
        "source_path",
        "filename",
        "media_type",
        "byte_size",
        "sha256",
        "title",
        "parser",
        "parser_version",
        "page_count",
        "ingested_at",
        "status",
    ),
    "chunks": (
        "chunk_id",
        "doc_id",
        "ordinal",
        "text",
        "kind",
        "page_start",
        "page_end",
        "section",
        "image_id",
        "anchor_chunk_id",
        "token_estimate",
        "embed_state",
    ),
    "images": (
        "image_id",
        "doc_id",
        "page",
        "storage_relpath",
        "width",
        "height",
        "byte_size",
        "sha256",
        "kind",
        "ocr_text",
        "ocr_confidence",
        "caption",
        "route",
        "status",
        "duplicate_count",
        "error",
    ),
    "vectors": (
        "chunk_id",
        "embed_model",
        "native_dim",
        "vector",
        "num_ctx",
        "embedded_at",
    ),
    "settings": ("key", "value"),
    "runs": (
        "run_id",
        "command",
        "started_at",
        "finished_at",
        "exit_code",
        "chunk_count",
        "ocr_count",
        "vision_count",
        "model_name",
        "refused_count",
        "ollama_error_count",
        "ask_latency_ms",
    ),
}

_DOC_ID = re.compile(r"^[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID = re.compile(r"^[0-9a-f]{12}:img:[0-9]{4}$")
_PRIVATE_DIR = 0o700
_PRIVATE_FILE = 0o600

_SCHEMA = """
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY CHECK (version >= 1)
);
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    media_type TEXT NOT NULL CHECK (media_type IN ('pdf', 'docx', 'pptx', 'text', 'image')),
    byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
    sha256 TEXT NOT NULL UNIQUE,
    title TEXT,
    parser TEXT NOT NULL CHECK (parser IN ('pymupdf', 'python-docx', 'python-pptx', 'text', 'image')),
    parser_version TEXT NOT NULL,
    page_count INTEGER CHECK (page_count IS NULL OR page_count >= 0),
    ingested_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('parsed', 'complete')),
    CHECK (sha256 = doc_id)
);
CREATE TABLE images (
    image_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    page INTEGER CHECK (page IS NULL OR page >= 1),
    storage_relpath TEXT NOT NULL,
    width INTEGER NOT NULL CHECK (width >= 1),
    height INTEGER NOT NULL CHECK (height >= 1),
    byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
    sha256 TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('embedded', 'page_render', 'vector_render', 'standalone')),
    ocr_text TEXT,
    ocr_confidence REAL CHECK (
        ocr_confidence IS NULL OR (ocr_confidence >= 0 AND ocr_confidence <= 1)
    ),
    caption TEXT,
    route TEXT CHECK (
        route IS NULL OR route IN (
            'text_rich', 'label_and_caption', 'caption_only',
            'skipped_small', 'skipped_duplicate', 'skipped_bomb'
        )
    ),
    status TEXT NOT NULL CHECK (
        status IN (
            'pending_ocr', 'ocr_done', 'pending_caption', 'caption_done',
            'empty', 'caption_failed', 'skipped', 'empty_no_vision', 'skipped_bomb'
        )
    ),
    duplicate_count INTEGER NOT NULL DEFAULT 0 CHECK (duplicate_count >= 0),
    error TEXT,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) DEFERRABLE INITIALLY DEFERRED
);
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0 AND ordinal <= 999999),
    text TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('text', 'ocr', 'figure', 'caption')),
    page_start INTEGER CHECK (page_start IS NULL OR page_start >= 1),
    page_end INTEGER CHECK (page_end IS NULL OR page_end >= 1),
    section TEXT,
    image_id TEXT,
    anchor_chunk_id TEXT,
    token_estimate INTEGER NOT NULL CHECK (token_estimate >= 1),
    embed_state TEXT NOT NULL CHECK (embed_state IN ('pending', 'blocked', 'done')),
    UNIQUE (doc_id, ordinal),
    CHECK (page_end IS NULL OR page_start IS NULL OR page_end >= page_start),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY (image_id) REFERENCES images(image_id) DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY (anchor_chunk_id) REFERENCES chunks(chunk_id) DEFERRABLE INITIALLY DEFERRED
);
CREATE TABLE vectors (
    chunk_id TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    native_dim INTEGER NOT NULL CHECK (native_dim >= 1),
    vector BLOB NOT NULL,
    num_ctx INTEGER NOT NULL CHECK (num_ctx >= 1),
    embedded_at TEXT NOT NULL,
    PRIMARY KEY (chunk_id, embed_model),
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) DEFERRABLE INITIALLY DEFERRED
);
CREATE TABLE settings (
    key TEXT PRIMARY KEY CHECK (
        key IN (
            'active_collection',
            'active_embed_model',
            'active_dimensions',
            'active_embed_num_ctx'
        )
    ),
    value TEXT NOT NULL CHECK (length(value) > 0)
);
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    command TEXT NOT NULL CHECK (
        command IN ('ingest', 'ask', 'reembed', 'delete', 'status', 'models')
    ),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    exit_code INTEGER CHECK (exit_code IS NULL OR (exit_code >= 0 AND exit_code <= 255)),
    chunk_count INTEGER NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
    ocr_count INTEGER NOT NULL DEFAULT 0 CHECK (ocr_count >= 0),
    vision_count INTEGER NOT NULL DEFAULT 0 CHECK (vision_count >= 0),
    model_name TEXT,
    refused_count INTEGER NOT NULL DEFAULT 0 CHECK (refused_count >= 0),
    ollama_error_count INTEGER NOT NULL DEFAULT 0 CHECK (ollama_error_count >= 0),
    ask_latency_ms INTEGER CHECK (ask_latency_ms IS NULL OR ask_latency_ms >= 0),
    CHECK ((finished_at IS NULL) = (exit_code IS NULL)),
    CHECK (command = 'ask' OR ask_latency_ms IS NULL),
    CHECK (command = 'ask' OR refused_count = 0)
);
CREATE INDEX chunks_doc_id_idx ON chunks (doc_id);
CREATE INDEX chunks_embed_state_idx ON chunks (embed_state);
CREATE INDEX images_doc_id_idx ON images (doc_id);
CREATE INDEX images_status_idx ON images (status);
CREATE INDEX documents_source_path_idx ON documents (source_path);
CREATE INDEX vectors_embed_model_idx ON vectors (embed_model);
"""


@dataclass(frozen=True, slots=True)
class Document:
    """One source file. ``doc_id`` and ``sha256`` are the same digest."""

    doc_id: str
    source_path: str
    filename: str
    media_type: str
    byte_size: int
    sha256: str
    parser: str
    parser_version: str
    ingested_at: str
    status: str
    title: str | None = None
    page_count: int | None = None


@dataclass(frozen=True, slots=True)
class Chunk:
    """Text that can be embedded. ``blocked`` waits for a caption."""

    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    kind: str
    token_estimate: int
    embed_state: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    image_id: str | None = None
    anchor_chunk_id: str | None = None


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """One stored image derivative. ``duplicate_count`` starts at 0."""

    image_id: str
    doc_id: str
    storage_relpath: str
    width: int
    height: int
    byte_size: int
    sha256: str
    kind: str
    status: str
    page: int | None = None
    ocr_text: str | None = None
    ocr_confidence: float | None = None
    caption: str | None = None
    route: str | None = None
    duplicate_count: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """Native-width float32 vector. The blob is little-endian."""

    chunk_id: str
    embed_model: str
    native_dim: int
    vector: tuple[float, ...]
    num_ctx: int
    embedded_at: str


@dataclass(frozen=True, slots=True)
class ActiveIndex:
    """The collection ingest and ask follow. All four fields are one record."""

    collection: str
    embed_model: str
    dimensions: int
    embed_num_ctx: int


@dataclass(frozen=True, slots=True)
class Run:
    """One command. Rows never store chunk text, OCR, or captions."""

    run_id: str
    command: str
    started_at: str
    chunk_count: int = 0
    ocr_count: int = 0
    vision_count: int = 0
    refused_count: int = 0
    ollama_error_count: int = 0
    finished_at: str | None = None
    exit_code: int | None = None
    model_name: str | None = None
    ask_latency_ms: int | None = None


def chunk_id_for(doc_id: str, ordinal: int) -> str:
    """Return ``{first 12 hex}:{ordinal:06d}``."""
    _require_doc_id(doc_id)
    _require_small_int(ordinal, "ordinal", maximum=999_999)
    return f"{doc_id[:12]}:{ordinal:06d}"


def image_id_for(doc_id: str, ordinal: int) -> str:
    """Return ``{first 12 hex}:img:{ordinal:04d}``."""
    _require_doc_id(doc_id)
    _require_small_int(ordinal, "image ordinal", maximum=9_999)
    return f"{doc_id[:12]}:img:{ordinal:04d}"


def storage_relpath(doc_id: str, image_id: str) -> str:
    """Return the path relative to the data directory."""
    _require_image_identity(doc_id, image_id)
    return f"images/{doc_id}/{image_id}.png"


def encode_vector(values: Sequence[float]) -> bytes:
    """Pack floats as little-endian float32."""
    if isinstance(values, (str, bytes)):
        raise RagError(EXIT_UNCLASSIFIED, t("store.vector_not_numbers"))
    numbers: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RagError(EXIT_UNCLASSIFIED, t("store.vector_not_numbers"))
        number = float(value)
        if not math.isfinite(number):
            raise RagError(EXIT_UNCLASSIFIED, t("store.vector_nonfinite"))
        numbers.append(number)
    if not numbers:
        raise RagError(EXIT_UNCLASSIFIED, t("store.vector_empty"))
    return struct.pack("<" + "f" * len(numbers), *numbers)


def decode_vector(blob: bytes) -> tuple[float, ...]:
    """Unpack a little-endian float32 blob."""
    if not isinstance(blob, bytes) or len(blob) % 4 != 0 or len(blob) == 0:
        raise RagError(EXIT_MISMATCH, t("store.blob_width"))
    count = len(blob) // 4
    return struct.unpack("<" + "f" * count, blob)


class SqliteStore:
    """Open ``data/corpus.sqlite`` and read or write corpus rows."""

    def __init__(self, data_dir: Path, db_path: Path, connection: sqlite3.Connection) -> None:
        self.data_dir = data_dir
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = connection
        self._depth = 0
        self._created_db = False

    @classmethod
    def open(cls, data_dir: Path) -> SqliteStore:
        """Create the data layout if needed and open the corpus database.

        Orphan image directories, those with no ``documents`` row, are removed
        on open. Symlinks are not followed.
        """
        root = _private_root(data_dir)
        created_root = _ensure_private_dir(root)
        for name in ("images", "locks", "logs"):
            _ensure_private_dir(root / name)
        if created_root:
            os.chmod(root, _PRIVATE_DIR)
        db_path = root / DB_FILENAME
        if db_path.is_symlink():
            raise RagError(EXIT_UNCLASSIFIED, t("store.db_symlink"))
        created_db = not db_path.exists()
        if created_db:
            fd = os.open(db_path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, _PRIVATE_FILE)
            os.close(fd)
            os.chmod(db_path, _PRIVATE_FILE)
        try:
            connection = sqlite3.connect(db_path, isolation_level=None)
        except sqlite3.Error as exc:
            raise RagError(EXIT_UNCLASSIFIED, t("store.open_failed")) from exc
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        store = cls(root, db_path, connection)
        store._created_db = created_db
        try:
            store._prepare()
        except Exception:
            connection.close()
            raise
        return store

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> SqliteStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Commit on success. Nested calls use savepoints."""
        conn = self._connection()
        depth = self._depth
        if depth == 0:
            conn.execute("BEGIN IMMEDIATE")
        else:
            conn.execute(f"SAVEPOINT ragcli_{depth}")
        self._depth = depth + 1
        try:
            yield
            if depth == 0:
                conn.commit()
                self._chmod_sidecars()
            else:
                conn.execute(f"RELEASE ragcli_{depth}")
        except sqlite3.IntegrityError as exc:
            self._depth = depth
            _abort_transaction(conn, depth)
            raise RagError(EXIT_UNCLASSIFIED, t("store.write_refused")) from exc
        except BaseException:
            self._depth = depth
            _abort_transaction(conn, depth)
            raise
        self._depth = depth

    def schema_version(self) -> int:
        row = self._connection().execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        if row is None or row[0] != SCHEMA_VERSION:
            raise RagError(EXIT_MISMATCH, t("store.schema_refused"))
        return SCHEMA_VERSION

    def journal_mode(self) -> str:
        row = self._connection().execute("PRAGMA journal_mode").fetchone()
        return str(row[0])

    def foreign_keys_enabled(self) -> bool:
        row = self._connection().execute("PRAGMA foreign_keys").fetchone()
        return int(row[0]) == 1

    def insert_document(self, document: Document) -> None:
        doc = _validate_document(document)
        with self.transaction():
            if self.get_document(doc.doc_id) is not None:
                raise RagError(EXIT_UNCLASSIFIED, t("store.document_exists"))
            other = self.find_document_by_path(doc.source_path)
            if other is not None:
                raise RagError(EXIT_UNCLASSIFIED, t("store.path_owned"))
            self._execute(
                """
                INSERT INTO documents (
                    doc_id, source_path, filename, media_type, byte_size, sha256,
                    title, parser, parser_version, page_count, ingested_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.doc_id,
                    doc.source_path,
                    doc.filename,
                    doc.media_type,
                    doc.byte_size,
                    doc.sha256,
                    doc.title,
                    doc.parser,
                    doc.parser_version,
                    doc.page_count,
                    doc.ingested_at,
                    doc.status,
                ),
            )

    def get_document(self, doc_id: str) -> Document | None:
        _require_doc_id(doc_id)
        row = self._connection().execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return None if row is None else _document_from_row(row)

    def find_document_by_path(self, source_path: str) -> Document | None:
        _require_text(source_path, "source_path")
        row = self._connection().execute(
            "SELECT * FROM documents WHERE source_path = ?", (source_path,)
        ).fetchone()
        return None if row is None else _document_from_row(row)

    def list_documents(self) -> list[Document]:
        """Return every document. Ingest uses this to resume unfinished work."""
        rows = self._connection().execute(
            "SELECT * FROM documents ORDER BY ingested_at, doc_id"
        ).fetchall()
        return [_document_from_row(row) for row in rows]

    def update_source_path(self, doc_id: str, source_path: str, filename: str) -> None:
        _require_doc_id(doc_id)
        _require_text(source_path, "source_path")
        _require_text(filename, "filename")
        with self.transaction():
            current = self.get_document(doc_id)
            if current is None:
                raise RagError(EXIT_NOT_FOUND, t("store.document_missing"))
            other = self.find_document_by_path(source_path)
            if other is not None and other.doc_id != doc_id:
                raise RagError(EXIT_UNCLASSIFIED, t("store.path_owned"))
            self._execute(
                "UPDATE documents SET source_path = ?, filename = ? WHERE doc_id = ?",
                (source_path, filename, doc_id),
            )

    def set_document_status(self, doc_id: str, status: str) -> None:
        _require_doc_id(doc_id)
        _choice(status, DOCUMENT_STATUSES, "documents.status")
        with self.transaction():
            self._require_document(doc_id)
            self._execute(
                "UPDATE documents SET status = ? WHERE doc_id = ?", (status, doc_id)
            )

    def refresh_document_status(self, doc_id: str) -> str:
        """Set ``complete`` when the document has no pending or blocked chunk."""
        with self.transaction():
            self._require_document(doc_id)
            row = self._connection().execute(
                """
                SELECT COUNT(*) FROM chunks
                WHERE doc_id = ? AND embed_state IN ('pending', 'blocked')
                """,
                (doc_id,),
            ).fetchone()
            status = "parsed" if int(row[0]) else "complete"
            self._execute(
                "UPDATE documents SET status = ? WHERE doc_id = ?", (status, doc_id)
            )
            return status

    def commit_parsed(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        images: Sequence[ImageRecord],
    ) -> None:
        """Insert one parsed file. The caller writes image bytes first."""
        if document.status != "parsed":
            raise RagError(EXIT_UNCLASSIFIED, t("store.status_parsed"))
        with self.transaction():
            self.insert_document(document)
            for image in images:
                self.insert_image(image)
            self.insert_chunks(chunks)

    def insert_chunk(self, chunk: Chunk) -> None:
        self.insert_chunks((chunk,))

    def insert_chunks(self, chunks: Sequence[Chunk]) -> None:
        with self.transaction():
            for chunk in chunks:
                self._insert_chunk(chunk)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self._connection().execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return None if row is None else _chunk_from_row(row)

    def chunks_for_document(self, doc_id: str) -> list[Chunk]:
        _require_doc_id(doc_id)
        rows = self._connection().execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY ordinal", (doc_id,)
        ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def chunk_ids_for(self, doc_id: str) -> list[str]:
        """Return chunk ids in reading order. Does not delete anything."""
        self._require_document(doc_id)
        rows = self._connection().execute(
            "SELECT chunk_id FROM chunks WHERE doc_id = ? ORDER BY ordinal", (doc_id,)
        ).fetchall()
        return [str(row[0]) for row in rows]

    def pending_embed_chunks(self) -> list[Chunk]:
        """Pending chunks with text. ``blocked`` rows are excluded."""
        rows = self._connection().execute(
            """
            SELECT * FROM chunks
            WHERE embed_state = 'pending' AND text <> ''
            ORDER BY doc_id, ordinal
            """
        ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def has_unfinished_chunks(self) -> bool:
        row = self._connection().execute(
            """
            SELECT 1 FROM chunks
            WHERE embed_state IN ('pending', 'blocked') LIMIT 1
            """
        ).fetchone()
        return row is not None

    def count_chunks(self, embed_state: str | None = None) -> int:
        if embed_state is None:
            row = self._connection().execute("SELECT COUNT(*) FROM chunks").fetchone()
        else:
            _choice(embed_state, EMBED_STATES, "embed_state")
            row = self._connection().execute(
                "SELECT COUNT(*) FROM chunks WHERE embed_state = ?", (embed_state,)
            ).fetchone()
        return int(row[0])

    def update_chunk_text(self, chunk_id: str, text: str, token_estimate: int) -> None:
        if not isinstance(text, str):
            raise RagError(EXIT_UNCLASSIFIED, t("store.text_type"))
        _require_small_int(token_estimate, "token_estimate", maximum=10_000_000, minimum=1)
        with self.transaction():
            if self.get_chunk(chunk_id) is None:
                raise RagError(EXIT_NOT_FOUND, t("store.chunk_missing"))
            self._execute(
                "UPDATE chunks SET text = ?, token_estimate = ? WHERE chunk_id = ?",
                (text, token_estimate, chunk_id),
            )

    def set_embed_state(self, chunk_id: str, embed_state: str) -> None:
        _choice(embed_state, EMBED_STATES, "embed_state")
        with self.transaction():
            if self.get_chunk(chunk_id) is None:
                raise RagError(EXIT_NOT_FOUND, t("store.chunk_missing"))
            self._execute(
                "UPDATE chunks SET embed_state = ? WHERE chunk_id = ?",
                (embed_state, chunk_id),
            )

    def delete_chunk(self, chunk_id: str) -> None:
        with self.transaction():
            if self.get_chunk(chunk_id) is None:
                raise RagError(EXIT_NOT_FOUND, t("store.chunk_missing"))
            self._execute("DELETE FROM vectors WHERE chunk_id = ?", (chunk_id,))
            self._execute("DELETE FROM chunks WHERE chunk_id = ?", (chunk_id,))

    def insert_image(self, image: ImageRecord) -> None:
        record = _validate_image(image)
        with self.transaction():
            if self.get_image(record.image_id) is not None:
                raise RagError(EXIT_UNCLASSIFIED, t("store.image_exists"))
            self._require_document(record.doc_id)
            self._execute(
                """
                INSERT INTO images (
                    image_id, doc_id, page, storage_relpath, width, height, byte_size,
                    sha256, kind, ocr_text, ocr_confidence, caption, route, status,
                    duplicate_count, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _image_values(record),
            )

    def get_image(self, image_id: str) -> ImageRecord | None:
        row = self._connection().execute(
            "SELECT * FROM images WHERE image_id = ?", (image_id,)
        ).fetchone()
        return None if row is None else _image_from_row(row)

    def images_for_document(self, doc_id: str) -> list[ImageRecord]:
        _require_doc_id(doc_id)
        rows = self._connection().execute(
            "SELECT * FROM images WHERE doc_id = ? ORDER BY image_id", (doc_id,)
        ).fetchall()
        return [_image_from_row(row) for row in rows]

    def update_image(self, image: ImageRecord) -> None:
        record = _validate_image(image)
        with self.transaction():
            current = self.get_image(record.image_id)
            if current is None:
                raise RagError(EXIT_NOT_FOUND, t("store.image_missing"))
            if current.doc_id != record.doc_id or current.storage_relpath != record.storage_relpath:
                raise RagError(EXIT_UNCLASSIFIED, t("store.image_identity"))
            self._execute(
                """
                UPDATE images SET
                    page = ?, width = ?, height = ?, byte_size = ?, sha256 = ?, kind = ?,
                    ocr_text = ?, ocr_confidence = ?, caption = ?, route = ?, status = ?,
                    duplicate_count = ?, error = ?
                WHERE image_id = ?
                """,
                (
                    record.page,
                    record.width,
                    record.height,
                    record.byte_size,
                    record.sha256,
                    record.kind,
                    record.ocr_text,
                    record.ocr_confidence,
                    record.caption,
                    record.route,
                    record.status,
                    record.duplicate_count,
                    record.error,
                    record.image_id,
                ),
            )

    def increment_duplicate_count(self, image_id: str) -> int:
        with self.transaction():
            current = self.get_image(image_id)
            if current is None:
                raise RagError(EXIT_NOT_FOUND, t("store.image_missing"))
            updated = current.duplicate_count + 1
            self._execute(
                "UPDATE images SET duplicate_count = ? WHERE image_id = ?",
                (updated, image_id),
            )
            return updated

    def count_images(self, status: str | None = None) -> int:
        if status is None:
            row = self._connection().execute("SELECT COUNT(*) FROM images").fetchone()
        else:
            _choice(status, IMAGE_STATUSES, "images.status")
            row = self._connection().execute(
                "SELECT COUNT(*) FROM images WHERE status = ?", (status,)
            ).fetchone()
        return int(row[0])

    def upsert_vector(self, record: VectorRecord) -> None:
        vector = _validate_vector(record)
        blob = encode_vector(vector.vector)
        if len(blob) != vector.native_dim * 4:
            raise RagError(EXIT_MISMATCH, t("store.vector_width_mismatch"))
        with self.transaction():
            if self.get_chunk(vector.chunk_id) is None:
                raise RagError(EXIT_NOT_FOUND, t("store.chunk_missing"))
            self._execute(
                """
                INSERT INTO vectors (
                    chunk_id, embed_model, native_dim, vector, num_ctx, embedded_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (chunk_id, embed_model) DO UPDATE SET
                    native_dim = excluded.native_dim,
                    vector = excluded.vector,
                    num_ctx = excluded.num_ctx,
                    embedded_at = excluded.embedded_at
                """,
                (
                    vector.chunk_id,
                    vector.embed_model,
                    vector.native_dim,
                    blob,
                    vector.num_ctx,
                    vector.embedded_at,
                ),
            )

    def get_vector(self, chunk_id: str, embed_model: str) -> VectorRecord | None:
        _require_text(embed_model, "embed_model")
        row = self._connection().execute(
            "SELECT * FROM vectors WHERE chunk_id = ? AND embed_model = ?",
            (chunk_id, embed_model),
        ).fetchone()
        return None if row is None else _vector_from_row(row)

    def vectors_for_chunk(self, chunk_id: str) -> list[VectorRecord]:
        """Return every stored model for one chunk, ordered by model tag."""
        _require_text(chunk_id, "chunk_id")
        rows = self._connection().execute(
            "SELECT * FROM vectors WHERE chunk_id = ? ORDER BY embed_model",
            (chunk_id,),
        ).fetchall()
        return [_vector_from_row(row) for row in rows]

    def get_active(self) -> ActiveIndex | None:
        rows = self._connection().execute("SELECT key, value FROM settings").fetchall()
        found = {str(row["key"]): str(row["value"]) for row in rows}
        if not found:
            return None
        unknown = set(found) - set(SETTING_KEYS)
        if unknown or set(found) != set(SETTING_KEYS):
            raise RagError(EXIT_MISMATCH, t("store.active_pointer_incomplete"))
        return ActiveIndex(
            collection=found["active_collection"],
            embed_model=found["active_embed_model"],
            dimensions=_setting_int(found["active_dimensions"]),
            embed_num_ctx=_setting_int(found["active_embed_num_ctx"]),
        )

    def set_active(self, active: ActiveIndex) -> None:
        collection = _require_text(active.collection, "active_collection")
        model = _require_text(active.embed_model, "active_embed_model")
        dimensions = _require_small_int(active.dimensions, "active_dimensions", maximum=100_000, minimum=1)
        num_ctx = _require_small_int(
            active.embed_num_ctx, "active_embed_num_ctx", maximum=1_000_000, minimum=1
        )
        pairs = (
            ("active_collection", collection),
            ("active_embed_model", model),
            ("active_dimensions", str(dimensions)),
            ("active_embed_num_ctx", str(num_ctx)),
        )
        with self.transaction():
            self._execute("DELETE FROM settings")
            self._connection().executemany(
                "INSERT INTO settings (key, value) VALUES (?, ?)", pairs
            )

    def start_run(self, run: Run) -> None:
        if run.finished_at is not None or run.exit_code is not None:
            raise RagError(EXIT_UNCLASSIFIED, t("store.run_open"))
        record = _validate_run(run)
        with self.transaction():
            if self.get_run(record.run_id) is not None:
                raise RagError(EXIT_UNCLASSIFIED, t("store.run_exists"))
            self._execute(
                """
                INSERT INTO runs (
                    run_id, command, started_at, finished_at, exit_code, chunk_count,
                    ocr_count, vision_count, model_name, refused_count,
                    ollama_error_count, ask_latency_ms
                ) VALUES (?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.command,
                    record.started_at,
                    record.chunk_count,
                    record.ocr_count,
                    record.vision_count,
                    record.model_name,
                    record.refused_count,
                    record.ollama_error_count,
                    record.ask_latency_ms,
                ),
            )

    def finish_run(
        self,
        run_id: str,
        *,
        finished_at: str,
        exit_code: int,
        chunk_count: int = 0,
        ocr_count: int = 0,
        vision_count: int = 0,
        model_name: str | None = None,
        refused_count: int = 0,
        ollama_error_count: int = 0,
        ask_latency_ms: int | None = None,
    ) -> None:
        _require_text(finished_at, "finished_at")
        _require_small_int(exit_code, "exit_code", maximum=255)
        with self.transaction():
            current = self.get_run(run_id)
            if current is None:
                raise RagError(EXIT_NOT_FOUND, t("store.run_missing"))
            if current.finished_at is not None:
                raise RagError(EXIT_UNCLASSIFIED, t("store.run_finished"))
            finished = _validate_run(
                Run(
                    run_id=current.run_id,
                    command=current.command,
                    started_at=current.started_at,
                    chunk_count=chunk_count,
                    ocr_count=ocr_count,
                    vision_count=vision_count,
                    refused_count=refused_count,
                    ollama_error_count=ollama_error_count,
                    finished_at=finished_at,
                    exit_code=exit_code,
                    model_name=model_name,
                    ask_latency_ms=ask_latency_ms,
                )
            )
            self._execute(
                """
                UPDATE runs SET
                    finished_at = ?, exit_code = ?, chunk_count = ?, ocr_count = ?,
                    vision_count = ?, model_name = ?, refused_count = ?,
                    ollama_error_count = ?, ask_latency_ms = ?
                WHERE run_id = ?
                """,
                (
                    finished.finished_at,
                    finished.exit_code,
                    finished.chunk_count,
                    finished.ocr_count,
                    finished.vision_count,
                    finished.model_name,
                    finished.refused_count,
                    finished.ollama_error_count,
                    finished.ask_latency_ms,
                    finished.run_id,
                ),
            )

    def get_run(self, run_id: str) -> Run | None:
        _require_text(run_id, "run_id")
        row = self._connection().execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return None if row is None else _run_from_row(row)

    def recent_runs(self, limit: int = 20) -> list[Run]:
        _require_small_int(limit, "limit", maximum=1000, minimum=1)
        rows = self._connection().execute(
            "SELECT * FROM runs ORDER BY started_at DESC, run_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_run_from_row(row) for row in rows]

    def delete_derivatives(self, doc_id: str) -> list[str]:
        """Drop chunks, images, vectors, and image files. Keep the document row.

        Callers that also delete Chroma ids should read ``chunk_ids_for`` and
        delete those ids before calling this. Image files go before the rows.
        Status returns to ``parsed``.
        """
        with self.transaction():
            ids = self.chunk_ids_for(doc_id)
            self.remove_image_dir(doc_id)
            self._delete_child_rows(doc_id)
            self._execute(
                "UPDATE documents SET status = 'parsed' WHERE doc_id = ?", (doc_id,)
            )
            return ids

    def delete_document(self, doc_id: str) -> list[str]:
        """Drop the document, its rows, and its image files."""
        with self.transaction():
            ids = self.chunk_ids_for(doc_id)
            self.remove_image_dir(doc_id)
            self._delete_child_rows(doc_id)
            self._execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            return ids

    def image_path(self, doc_id: str, image_id: str) -> Path:
        """Return the PNG path. It is lexical and stays under ``data/images``."""
        relative = storage_relpath(doc_id, image_id)
        path = self.data_dir / relative
        _assert_under(path, self.data_dir / "images")
        return path

    def write_image(self, doc_id: str, image_id: str, payload: bytes) -> str:
        """Write PNG bytes mode 0600. Return the relative storage path."""
        if not isinstance(payload, bytes):
            raise RagError(EXIT_UNCLASSIFIED, t("store.image_bytes"))
        path = self.image_path(doc_id, image_id)
        _ensure_private_dir(path.parent)
        _assert_under(path, self.data_dir / "images")
        try:
            fd = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
                _PRIVATE_FILE,
            )
        except OSError as exc:
            raise RagError(EXIT_UNCLASSIFIED, t("store.image_write_failed")) from exc
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        os.chmod(path, _PRIVATE_FILE)
        return storage_relpath(doc_id, image_id)

    def remove_image_dir(self, doc_id: str) -> None:
        """Remove ``data/images/{doc_id}`` when it is a real directory."""
        _require_doc_id(doc_id)
        root = self.data_dir / "images"
        path = root / doc_id
        if path.is_symlink():
            raise RagError(EXIT_UNCLASSIFIED, t("store.image_dir_symlink"))
        if not path.exists():
            return
        if not path.is_dir():
            raise RagError(EXIT_UNCLASSIFIED, t("store.image_path_directory"))
        _assert_under(path, root)
        shutil.rmtree(path)

    def _prepare(self) -> None:
        conn = self._connection()
        tables = _table_names(conn)
        if not tables:
            self._create_schema()
        else:
            _require_known_schema(conn)
        mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()
        if str(mode[0]).lower() != "wal":
            raise RagError(EXIT_UNCLASSIFIED, t("store.wal_failed"))
        if self._created_db:
            os.chmod(self.db_path, _PRIVATE_FILE)
        self._chmod_sidecars()
        self._remove_orphan_image_dirs()

    def _create_schema(self) -> None:
        conn = self._connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in _sql_statements(_SCHEMA):
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations (version) VALUES (?)", (SCHEMA_VERSION,)
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        _require_known_schema(conn)

    def _remove_orphan_image_dirs(self) -> None:
        root = self.data_dir / "images"
        if not root.is_dir() or root.is_symlink():
            return
        known = {
            str(row[0])
            for row in self._connection().execute("SELECT doc_id FROM documents")
        }
        for child in list(root.iterdir()):
            if child.is_symlink() or not child.is_dir():
                continue
            if child.name in known:
                continue
            resolved = child.resolve()
            if not resolved.is_relative_to(root.resolve()) or resolved == root.resolve():
                continue
            shutil.rmtree(child)

    def _insert_chunk(self, chunk: Chunk) -> None:
        record = _validate_chunk(chunk)
        if self.get_chunk(record.chunk_id) is not None:
            raise RagError(EXIT_UNCLASSIFIED, t("store.chunk_exists"))
        self._require_document(record.doc_id)
        if record.image_id is not None and self.get_image(record.image_id) is None:
            raise RagError(EXIT_NOT_FOUND, t("store.chunk_image_missing"))
        self._execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, ordinal, text, kind, page_start, page_end, section,
                image_id, anchor_chunk_id, token_estimate, embed_state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.chunk_id,
                record.doc_id,
                record.ordinal,
                record.text,
                record.kind,
                record.page_start,
                record.page_end,
                record.section,
                record.image_id,
                record.anchor_chunk_id,
                record.token_estimate,
                record.embed_state,
            ),
        )

    def _delete_child_rows(self, doc_id: str) -> None:
        self._execute(
            """
            DELETE FROM vectors WHERE chunk_id IN (
                SELECT chunk_id FROM chunks WHERE doc_id = ?
            )
            """,
            (doc_id,),
        )
        self._execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._execute("DELETE FROM images WHERE doc_id = ?", (doc_id,))

    def _require_document(self, doc_id: str) -> Document:
        document = self.get_document(doc_id)
        if document is None:
            raise RagError(EXIT_NOT_FOUND, t("store.document_missing"))
        return document

    def _chmod_sidecars(self) -> None:
        for suffix in ("-wal", "-shm"):
            path = Path(str(self.db_path) + suffix)
            if path.exists() and not path.is_symlink():
                os.chmod(path, _PRIVATE_FILE)

    def _execute(self, sql: str, params: tuple[object, ...] = ()) -> None:
        try:
            self._connection().execute(sql, params)
        except sqlite3.IntegrityError as exc:
            raise RagError(EXIT_UNCLASSIFIED, t("store.write_refused")) from exc

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RagError(EXIT_UNCLASSIFIED, t("store.closed"))
        return self._conn


def _abort_transaction(conn: sqlite3.Connection, depth: int) -> None:
    try:
        if depth == 0:
            conn.rollback()
        else:
            conn.execute(f"ROLLBACK TO ragcli_{depth}")
            conn.execute(f"RELEASE ragcli_{depth}")
    except sqlite3.Error:
        if depth == 0:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {str(row[0]) for row in rows}


def _require_known_schema(conn: sqlite3.Connection) -> None:
    tables = _table_names(conn)
    expected = set(_TABLE_COLUMNS)
    if tables != expected or "schema_migrations" not in tables:
        raise RagError(EXIT_MISMATCH, t("store.schema_refused"))
    versions = [
        int(row[0])
        for row in conn.execute("SELECT version FROM schema_migrations ORDER BY version")
    ]
    if versions != [SCHEMA_VERSION]:
        raise RagError(EXIT_MISMATCH, t("store.schema_refused"))
    for table, columns in _TABLE_COLUMNS.items():
        found = tuple(
            str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")
        )
        if found != columns:
            raise RagError(EXIT_MISMATCH, t("store.schema_refused"))


def _private_root(data_dir: Path) -> Path:
    if not isinstance(data_dir, Path):
        raise RagError(EXIT_UNCLASSIFIED, t("lock.data_dir_type"))
    if data_dir.is_symlink():
        raise RagError(EXIT_UNCLASSIFIED, t("io.data_dir_symlink"))
    return data_dir.expanduser().resolve()


def _ensure_private_dir(path: Path) -> bool:
    """Create ``path`` as 0700. Return whether this call created it.

    An existing directory is left alone, including one the user chmod'd later.
    """
    if path.is_symlink():
        raise RagError(EXIT_UNCLASSIFIED, t("io.data_dir_symlink"))
    if path.exists():
        if not path.is_dir():
            raise RagError(EXIT_UNCLASSIFIED, t("io.data_path_not_directory", name=path.name))
        return False
    try:
        path.mkdir(mode=_PRIVATE_DIR)
    except FileExistsError:
        if path.is_symlink() or not path.is_dir():
            raise RagError(EXIT_UNCLASSIFIED, t("io.data_dir_symlink")) from None
        return False
    os.chmod(path, _PRIVATE_DIR)
    return True


def _sql_statements(script: str) -> list[str]:
    return [piece.strip() for piece in script.split(";") if piece.strip()]


def _assert_under(path: Path, root: Path) -> None:
    root_abs = root if root.is_absolute() else root.absolute()
    path_abs = path if path.is_absolute() else path.absolute()
    if root_abs.is_symlink() or path_abs == root_abs or not path_abs.is_relative_to(root_abs):
        raise RagError(EXIT_UNCLASSIFIED, t("store.image_path_outside"))
    cursor = path_abs
    while cursor != root_abs:
        if cursor.is_symlink():
            raise RagError(EXIT_UNCLASSIFIED, t("store.image_path_symlink"))
        parent = cursor.parent
        if parent == cursor:
            raise RagError(EXIT_UNCLASSIFIED, t("store.image_path_outside"))
        cursor = parent


def _validate_document(document: Document) -> Document:
    _require_doc_id(document.doc_id)
    if document.sha256 != document.doc_id:
        raise RagError(EXIT_UNCLASSIFIED, t("store.sha256_doc_id"))
    _choice(document.media_type, MEDIA_TYPES, "media_type")
    _choice(document.parser, PARSERS, "parser")
    _choice(document.status, DOCUMENT_STATUSES, "documents.status")
    _require_text(document.source_path, "source_path")
    _require_text(document.filename, "filename")
    _require_text(document.parser_version, "parser_version")
    _require_text(document.ingested_at, "ingested_at")
    _require_small_int(document.byte_size, "byte_size", maximum=2**62)
    if document.title is not None:
        _require_optional_text(document.title, "title")
    if document.page_count is not None:
        _require_small_int(document.page_count, "page_count", maximum=1_000_000)
    return document


def _validate_chunk(chunk: Chunk) -> Chunk:
    _require_doc_id(chunk.doc_id)
    expected = chunk_id_for(chunk.doc_id, chunk.ordinal)
    if chunk.chunk_id != expected:
        raise RagError(EXIT_UNCLASSIFIED, t("store.chunk_id_expected", expected=expected))
    if not isinstance(chunk.text, str):
        raise RagError(EXIT_UNCLASSIFIED, t("store.text_type"))
    _choice(chunk.kind, CHUNK_KINDS, "chunks.kind")
    _choice(chunk.embed_state, EMBED_STATES, "embed_state")
    _require_small_int(chunk.token_estimate, "token_estimate", maximum=10_000_000, minimum=1)
    _require_page_pair(chunk.page_start, chunk.page_end)
    if chunk.section is not None:
        _require_optional_text(chunk.section, "section")
    if chunk.image_id is not None:
        _require_image_identity(chunk.doc_id, chunk.image_id)
    if chunk.anchor_chunk_id is not None:
        if not chunk.anchor_chunk_id.startswith(chunk.doc_id[:12] + ":"):
            raise RagError(EXIT_UNCLASSIFIED, t("store.anchor_chunk"))
    return chunk


def _validate_image(image: ImageRecord) -> ImageRecord:
    _require_image_identity(image.doc_id, image.image_id)
    if image.storage_relpath != storage_relpath(image.doc_id, image.image_id):
        raise RagError(EXIT_UNCLASSIFIED, t("store.storage_relpath"))
    if not _SHA256.fullmatch(image.sha256):
        raise RagError(EXIT_UNCLASSIFIED, t("store.image_sha256"))
    _choice(image.kind, IMAGE_KINDS, "images.kind")
    _choice(image.status, IMAGE_STATUSES, "images.status")
    if image.route is not None:
        _choice(image.route, IMAGE_ROUTES, "images.route")
    _require_small_int(image.width, "width", maximum=1_000_000, minimum=1)
    _require_small_int(image.height, "height", maximum=1_000_000, minimum=1)
    _require_small_int(image.byte_size, "byte_size", maximum=2**62)
    _require_small_int(image.duplicate_count, "duplicate_count", maximum=1_000_000)
    if image.page is not None:
        _require_small_int(image.page, "page", maximum=1_000_000, minimum=1)
    if image.ocr_text is not None:
        _require_optional_text(image.ocr_text, "ocr_text")
    if image.caption is not None:
        _require_optional_text(image.caption, "caption")
    if image.error is not None:
        _require_optional_text(image.error, "error")
    confidence = image.ocr_confidence
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise RagError(EXIT_UNCLASSIFIED, t("store.confidence_range"))
        if not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
            raise RagError(EXIT_UNCLASSIFIED, t("store.confidence_range"))
    return image


def _validate_vector(record: VectorRecord) -> VectorRecord:
    _require_text(record.chunk_id, "chunk_id")
    _require_text(record.embed_model, "embed_model")
    _require_text(record.embedded_at, "embedded_at")
    _require_small_int(record.native_dim, "native_dim", maximum=100_000, minimum=1)
    _require_small_int(record.num_ctx, "num_ctx", maximum=1_000_000, minimum=1)
    if len(record.vector) != record.native_dim:
        raise RagError(EXIT_MISMATCH, t("store.vector_width_mismatch"))
    encode_vector(record.vector)
    return record


def _validate_run(run: Run) -> Run:
    _require_text(run.run_id, "run_id")
    _choice(run.command, RUN_COMMANDS, "runs.command")
    _require_text(run.started_at, "started_at")
    _require_small_int(run.chunk_count, "chunk_count", maximum=10_000_000)
    _require_small_int(run.ocr_count, "ocr_count", maximum=10_000_000)
    _require_small_int(run.vision_count, "vision_count", maximum=10_000_000)
    _require_small_int(run.ollama_error_count, "ollama_error_count", maximum=10_000_000)
    _require_small_int(run.refused_count, "refused_count", maximum=10_000_000)
    if run.model_name is not None:
        _require_text(run.model_name, "model_name")
    if run.command != "ask":
        if run.ask_latency_ms is not None:
            raise RagError(EXIT_UNCLASSIFIED, t("store.ask_latency_null"))
        if run.refused_count != 0:
            raise RagError(EXIT_UNCLASSIFIED, t("store.refused_count_zero"))
    elif run.finished_at is not None and run.ask_latency_ms is None:
        raise RagError(EXIT_UNCLASSIFIED, t("store.ask_latency_required"))
    if run.ask_latency_ms is not None:
        _require_small_int(run.ask_latency_ms, "ask_latency_ms", maximum=3_600_000)
    if (run.finished_at is None) != (run.exit_code is None):
        raise RagError(EXIT_UNCLASSIFIED, t("store.run_finished_pair"))
    if run.finished_at is not None:
        _require_text(run.finished_at, "finished_at")
        _require_small_int(run.exit_code if run.exit_code is not None else -1, "exit_code", maximum=255)
    return run


def _vector_from_row(row: sqlite3.Row) -> VectorRecord:
    values = decode_vector(row["vector"])
    if len(values) != int(row["native_dim"]):
        raise RagError(EXIT_MISMATCH, t("store.vector_width_mismatch"))
    return VectorRecord(
        chunk_id=row["chunk_id"],
        embed_model=row["embed_model"],
        native_dim=int(row["native_dim"]),
        vector=values,
        num_ctx=int(row["num_ctx"]),
        embedded_at=row["embedded_at"],
    )


def _image_values(image: ImageRecord) -> tuple[object, ...]:
    return (
        image.image_id,
        image.doc_id,
        image.page,
        image.storage_relpath,
        image.width,
        image.height,
        image.byte_size,
        image.sha256,
        image.kind,
        image.ocr_text,
        image.ocr_confidence,
        image.caption,
        image.route,
        image.status,
        image.duplicate_count,
        image.error,
    )


def _document_from_row(row: sqlite3.Row) -> Document:
    return Document(
        doc_id=row["doc_id"],
        source_path=row["source_path"],
        filename=row["filename"],
        media_type=row["media_type"],
        byte_size=int(row["byte_size"]),
        sha256=row["sha256"],
        parser=row["parser"],
        parser_version=row["parser_version"],
        ingested_at=row["ingested_at"],
        status=row["status"],
        title=row["title"],
        page_count=None if row["page_count"] is None else int(row["page_count"]),
    )


def _chunk_from_row(row: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        ordinal=int(row["ordinal"]),
        text=row["text"],
        kind=row["kind"],
        token_estimate=int(row["token_estimate"]),
        embed_state=row["embed_state"],
        page_start=None if row["page_start"] is None else int(row["page_start"]),
        page_end=None if row["page_end"] is None else int(row["page_end"]),
        section=row["section"],
        image_id=row["image_id"],
        anchor_chunk_id=row["anchor_chunk_id"],
    )


def _image_from_row(row: sqlite3.Row) -> ImageRecord:
    confidence = row["ocr_confidence"]
    return ImageRecord(
        image_id=row["image_id"],
        doc_id=row["doc_id"],
        storage_relpath=row["storage_relpath"],
        width=int(row["width"]),
        height=int(row["height"]),
        byte_size=int(row["byte_size"]),
        sha256=row["sha256"],
        kind=row["kind"],
        status=row["status"],
        page=None if row["page"] is None else int(row["page"]),
        ocr_text=row["ocr_text"],
        ocr_confidence=None if confidence is None else float(confidence),
        caption=row["caption"],
        route=row["route"],
        duplicate_count=int(row["duplicate_count"]),
        error=row["error"],
    )


def _run_from_row(row: sqlite3.Row) -> Run:
    return Run(
        run_id=row["run_id"],
        command=row["command"],
        started_at=row["started_at"],
        chunk_count=int(row["chunk_count"]),
        ocr_count=int(row["ocr_count"]),
        vision_count=int(row["vision_count"]),
        refused_count=int(row["refused_count"]),
        ollama_error_count=int(row["ollama_error_count"]),
        finished_at=row["finished_at"],
        exit_code=None if row["exit_code"] is None else int(row["exit_code"]),
        model_name=row["model_name"],
        ask_latency_ms=None if row["ask_latency_ms"] is None else int(row["ask_latency_ms"]),
    )


def _choice(value: str, allowed: frozenset[str], label: str) -> str:
    if value not in allowed:
        names = t("config.list_joiner").join(sorted(allowed))
        raise RagError(EXIT_UNCLASSIFIED, t("store.choice", label=label, names=names))
    return value


def _require_doc_id(doc_id: str) -> str:
    if not isinstance(doc_id, str) or _DOC_ID.fullmatch(doc_id) is None:
        raise RagError(EXIT_UNCLASSIFIED, t("store.doc_id"))
    return doc_id


def _require_image_identity(doc_id: str, image_id: str) -> None:
    _require_doc_id(doc_id)
    if not isinstance(image_id, str) or _IMAGE_ID.fullmatch(image_id) is None:
        raise RagError(EXIT_UNCLASSIFIED, t("store.image_id_format"))
    if not image_id.startswith(doc_id[:12] + ":"):
        raise RagError(EXIT_UNCLASSIFIED, t("store.image_id_document"))


def _require_text(value: str, label: str) -> str:
    if not isinstance(value, str) or value == "" or "\x00" in value:
        raise RagError(EXIT_UNCLASSIFIED, t("config.nonempty_string", label=label))
    return value


def _require_optional_text(value: str, label: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise RagError(EXIT_UNCLASSIFIED, t("config.must_be_string", label=label))
    return value


def _require_small_int(
    value: int | None, label: str, *, maximum: int, minimum: int = 0
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        raise RagError(
            EXIT_UNCLASSIFIED,
            t("store.int_range", label=label, minimum=minimum, maximum=maximum),
        )
    return value


def _require_page_pair(page_start: int | None, page_end: int | None) -> None:
    if page_start is not None:
        _require_small_int(page_start, "page_start", maximum=1_000_000, minimum=1)
    if page_end is not None:
        _require_small_int(page_end, "page_end", maximum=1_000_000, minimum=1)
    if page_start is not None and page_end is not None and page_end < page_start:
        raise RagError(EXIT_UNCLASSIFIED, t("store.page_order"))


def _setting_int(value: str) -> int:
    try:
        number = int(value, 10)
    except ValueError as exc:
        raise RagError(EXIT_MISMATCH, t("store.active_pointer_incomplete")) from exc
    if str(number) != value or number < 1:
        raise RagError(EXIT_MISMATCH, t("store.active_pointer_incomplete"))
    return number
