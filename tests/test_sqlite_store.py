"""SQLite schema, permissions, and the corpus store. No parser and no Ollama."""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import stat
import struct
from pathlib import Path

import pytest

from ragcli.errors import EXIT_MISMATCH, EXIT_NOT_FOUND, EXIT_UNCLASSIFIED, RagError
from ragcli.i18n.en import MESSAGES as EN_MESSAGES
from ragcli.i18n.zh_Hant import MESSAGES as ZH_MESSAGES
from ragcli.store.sqlite import (
    SCHEMA_VERSION,
    _SCHEMA,
    ActiveIndex,
    Chunk,
    Document,
    ImageRecord,
    Run,
    SqliteStore,
    VectorRecord,
    _sql_statements,
    chunk_id_for,
    decode_vector,
    encode_vector,
    image_id_for,
    storage_relpath,
)

DOC = "a" * 64
DOC_B = "b" * 64
IMAGE_SHA = "c" * 64
SECRET = "僅存在資料庫的標記句"
WHEN = "2026-10-25T00:00:00+00:00"

DOCUMENT_COLUMNS = (
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
)
CHUNK_COLUMNS = (
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
)
IMAGE_COLUMNS = (
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
)
VECTOR_COLUMNS = (
    "chunk_id",
    "embed_model",
    "native_dim",
    "vector",
    "num_ctx",
    "embedded_at",
)
RUN_COLUMNS = (
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
)


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def make_document(doc_id: str = DOC, **overrides) -> Document:
    values = {
        "doc_id": doc_id,
        "source_path": f"/docs/{doc_id}.pdf",
        "filename": "退款規定.pdf",
        "media_type": "pdf",
        "byte_size": 32,
        "sha256": doc_id,
        "parser": "pymupdf",
        "parser_version": "1",
        "ingested_at": WHEN,
        "status": "parsed",
        "title": None,
        "page_count": 2,
    }
    values.update(overrides)
    if "doc_id" in overrides and "sha256" not in overrides:
        values["sha256"] = overrides["doc_id"]
    return Document(**values)


def make_chunk(doc_id: str = DOC, ordinal: int = 0, **overrides) -> Chunk:
    values = {
        "chunk_id": chunk_id_for(doc_id, ordinal),
        "doc_id": doc_id,
        "ordinal": ordinal,
        "text": "退款規定",
        "kind": "text",
        "token_estimate": 3,
        "embed_state": "pending",
        "page_start": 1,
        "page_end": 1,
        "section": None,
        "image_id": None,
        "anchor_chunk_id": None,
    }
    values.update(overrides)
    if "chunk_id" not in overrides:
        values["chunk_id"] = chunk_id_for(values["doc_id"], values["ordinal"])
    return Chunk(**values)


def make_image(doc_id: str = DOC, ordinal: int = 0, **overrides) -> ImageRecord:
    image_id = image_id_for(doc_id, ordinal)
    values = {
        "image_id": image_id,
        "doc_id": doc_id,
        "storage_relpath": storage_relpath(doc_id, image_id),
        "width": 120,
        "height": 80,
        "byte_size": 40,
        "sha256": IMAGE_SHA,
        "kind": "embedded",
        "status": "pending_ocr",
        "page": 1,
        "ocr_text": None,
        "ocr_confidence": None,
        "caption": None,
        "route": None,
        "duplicate_count": 0,
        "error": None,
    }
    values.update(overrides)
    if "image_id" not in overrides:
        values["image_id"] = image_id_for(values["doc_id"], ordinal)
        values["storage_relpath"] = storage_relpath(values["doc_id"], values["image_id"])
    return ImageRecord(**values)


@pytest.fixture
def store(tmp_path: Path):
    opened = SqliteStore.open(tmp_path / "data")
    try:
        yield opened
    finally:
        opened.close()


def test_schema_version_and_contract_sets():
    assert SCHEMA_VERSION == 2
    from ragcli.store import sqlite as module

    assert module.DOCUMENT_STATUSES == frozenset({"parsed", "complete"})
    assert module.EMBED_STATES == frozenset({"pending", "blocked", "done"})
    assert "empty_no_vision" in module.IMAGE_STATUSES
    assert "skipped_bomb" in module.IMAGE_STATUSES
    assert module.RUN_COMMANDS == frozenset(
        {"ingest", "ask", "reembed", "delete", "status", "models"}
    )


def test_new_data_dir_is_private_and_database_is_wal(tmp_path: Path):
    previous = os.umask(0)
    try:
        opened = SqliteStore.open(tmp_path / "data")
        try:
            data = opened.data_dir
            assert mode(data) == 0o700
            for name in ("images", "locks", "logs"):
                assert mode(data / name) == 0o700
            assert mode(opened.db_path) == 0o600
            assert opened.schema_version() == 2
            assert not (data / "corpus.sqlite.v1.bak").exists()
            assert opened.journal_mode() == "wal"
            assert opened.foreign_keys_enabled() is True
            assert not (data / "logs" / "rag.log").exists()
            opened.insert_document(make_document())
            for suffix in ("-wal", "-shm"):
                sidecar = Path(str(opened.db_path) + suffix)
                assert sidecar.exists()
                assert mode(sidecar) == 0o600
        finally:
            opened.close()
    finally:
        os.umask(previous)


def test_existing_directory_and_database_modes_stay(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    data.chmod(0o755)
    opened = SqliteStore.open(data)
    opened.close()
    assert mode(data) == 0o755
    assert mode(data / "locks") == 0o700
    opened.db_path.chmod(0o640)
    reopened = SqliteStore.open(data)
    try:
        assert mode(reopened.db_path) == 0o640
        assert reopened.journal_mode() == "wal"
    finally:
        reopened.close()


def test_table_columns_match_the_design(store: SqliteStore):
    conn = store._connection()
    expected = {
        "schema_migrations": ("version",),
        "documents": DOCUMENT_COLUMNS,
        "chunks": CHUNK_COLUMNS,
        "images": IMAGE_COLUMNS,
        "vectors": VECTOR_COLUMNS,
        "settings": ("key", "value"),
        "runs": RUN_COLUMNS,
    }
    found = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    fts_names = {
        "chunks_fts",
        "chunks_fts_config",
        "chunks_fts_content",
        "chunks_fts_data",
        "chunks_fts_docsize",
        "chunks_fts_idx",
    }
    assert found == set(expected) | fts_names
    for table, columns in expected.items():
        info = tuple(row[1] for row in conn.execute(f"PRAGMA table_info({table})"))
        assert info == columns
    versions = [row[0] for row in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]
    assert versions == [1, 2]
    chunk_fks = {row[2] for row in conn.execute("PRAGMA foreign_key_list(chunks)")}
    assert chunk_fks == {"documents", "images", "chunks"}
    image_fks = {row[2] for row in conn.execute("PRAGMA foreign_key_list(images)")}
    assert image_fks == {"documents"}
    vector_fks = {row[2] for row in conn.execute("PRAGMA foreign_key_list(vectors)")}
    assert vector_fks == {"chunks"}


def test_round_trip_document_chunk_and_null_page(store: SqliteStore):
    document = make_document(title="退款", page_count=None)
    chunk = make_chunk(text=SECRET, page_start=None, page_end=None, embed_state="blocked")
    store.insert_document(document)
    store.insert_chunk(chunk)
    assert store.get_document(DOC) == document
    assert store.find_document_by_path(document.source_path) == document
    stored = store.get_chunk(chunk.chunk_id)
    assert stored == chunk
    assert stored is not None and stored.page_start is None
    row = store._connection().execute(
        "SELECT page_start FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,)
    ).fetchone()
    assert row[0] is None
    assert store.has_unfinished_chunks() is True
    assert store.pending_embed_chunks() == []


def test_pending_embed_skips_blocked_and_empty_text(store: SqliteStore):
    store.insert_document(make_document())
    ready = make_chunk(ordinal=0, text="有字", embed_state="pending")
    empty = make_chunk(ordinal=1, text="", embed_state="pending", token_estimate=1)
    blocked = make_chunk(ordinal=2, text="等描述", embed_state="blocked", kind="figure")
    store.insert_chunks((ready, empty, blocked))
    assert store.pending_embed_chunks() == [ready]
    assert store.count_chunks("blocked") == 1
    assert store.refresh_document_status(DOC) == "parsed"
    store.set_embed_state(blocked.chunk_id, "pending")
    store.delete_chunk(empty.chunk_id)
    store.set_embed_state(ready.chunk_id, "done")
    store.set_embed_state(blocked.chunk_id, "done")
    assert store.refresh_document_status(DOC) == "complete"
    assert store.has_unfinished_chunks() is False


def test_image_statuses_duplicate_count_and_ocr_update(store: SqliteStore):
    store.insert_document(make_document())
    image = make_image(status="skipped_bomb", route="skipped_bomb")
    store.insert_image(image)
    assert store.get_image(image.image_id) == image
    assert store.increment_duplicate_count(image.image_id) == 1
    assert store.increment_duplicate_count(image.image_id) == 2
    updated = make_image(
        status="empty_no_vision",
        route="caption_only",
        ocr_text="",
        ocr_confidence=0.25,
        duplicate_count=2,
        error=None,
    )
    store.update_image(updated)
    assert store.get_image(image.image_id) == updated
    assert store.count_images("empty_no_vision") == 1


def test_vector_is_little_endian_and_keeps_native_width(store: SqliteStore):
    assert encode_vector([1.0, -2.5]) == struct.pack("<2f", 1.0, -2.5)
    assert decode_vector(encode_vector([1.0, -2.5])) == (1.0, -2.5)
    store.insert_document(make_document())
    chunk = make_chunk()
    store.insert_chunk(chunk)
    wide = (0.25,) * 2560
    store.upsert_vector(
        VectorRecord(
            chunk_id=chunk.chunk_id,
            embed_model="qwen3-embedding:4b",
            native_dim=2560,
            vector=wide,
            num_ctx=2048,
            embedded_at=WHEN,
        )
    )
    small = (1.0, 0.0, 0.0, 1.0)
    store.upsert_vector(
        VectorRecord(
            chunk_id=chunk.chunk_id,
            embed_model="qwen3-embedding:0.6b",
            native_dim=4,
            vector=small,
            num_ctx=2048,
            embedded_at=WHEN,
        )
    )
    store.upsert_vector(
        VectorRecord(
            chunk_id=chunk.chunk_id,
            embed_model="qwen3-embedding:0.6b",
            native_dim=4,
            vector=(2.0, 0.0, 0.0, 1.0),
            num_ctx=2048,
            embedded_at=WHEN,
        )
    )
    first = store.get_vector(chunk.chunk_id, "qwen3-embedding:0.6b")
    second = store.get_vector(chunk.chunk_id, "qwen3-embedding:4b")
    assert first is not None and first.vector == (2.0, 0.0, 0.0, 1.0)
    assert first.native_dim == 4
    assert second is not None and second.native_dim == 2560
    assert second.vector == wide
    blob = store._connection().execute(
        "SELECT vector, native_dim FROM vectors WHERE embed_model = ?",
        ("qwen3-embedding:0.6b",),
    ).fetchone()
    assert blob[0] == struct.pack("<4f", 2.0, 0.0, 0.0, 1.0)
    assert blob[1] == 4
    with pytest.raises(RagError) as exc:
        store.upsert_vector(
            VectorRecord(
                chunk_id=chunk.chunk_id,
                embed_model="qwen3-embedding:0.6b",
                native_dim=4,
                vector=(1.0, 2.0),
                num_ctx=2048,
                embedded_at=WHEN,
            )
        )
    assert exc.value.code == EXIT_MISMATCH


def test_active_index_is_one_record(store: SqliteStore):
    assert store.get_active() is None
    active = ActiveIndex(
        collection="emb-qwen3-embedding-0.6b-d1024",
        embed_model="qwen3-embedding:0.6b",
        dimensions=1024,
        embed_num_ctx=2048,
    )
    store.set_active(active)
    assert store.get_active() == active
    store._connection().execute("DELETE FROM settings WHERE key = 'active_dimensions'")
    with pytest.raises(RagError) as exc:
        store.get_active()
    assert exc.value.code == EXIT_MISMATCH


def test_runs_use_the_schema_fields_and_ask_rules(store: SqliteStore):
    store.start_run(Run(run_id="ingest-1", command="ingest", started_at=WHEN))
    with pytest.raises(RagError, match="ask_latency_ms"):
        store.finish_run(
            "ingest-1",
            finished_at="2026-10-25T00:00:01+00:00",
            exit_code=0,
            ask_latency_ms=15,
        )
    with pytest.raises(RagError, match="refused_count"):
        store.finish_run(
            "ingest-1",
            finished_at="2026-10-25T00:00:01+00:00",
            exit_code=0,
            refused_count=1,
        )
    assert store.get_run("ingest-1").finished_at is None
    store.finish_run(
        "ingest-1",
        finished_at="2026-10-25T00:00:01+00:00",
        exit_code=0,
        chunk_count=2,
        ocr_count=1,
        vision_count=0,
        model_name="qwen3-embedding:0.6b",
    )
    finished = store.get_run("ingest-1")
    assert finished is not None
    assert finished.ask_latency_ms is None
    assert finished.refused_count == 0
    assert finished.chunk_count == 2
    assert finished.model_name == "qwen3-embedding:0.6b"
    store.start_run(Run(run_id="ask-1", command="ask", started_at="2026-10-25T00:00:02+00:00"))
    with pytest.raises(RagError, match="ask_latency_ms"):
        store.finish_run("ask-1", finished_at="2026-10-25T00:00:03+00:00", exit_code=0)
    store.finish_run(
        "ask-1",
        finished_at="2026-10-25T00:00:03+00:00",
        exit_code=0,
        refused_count=1,
        ask_latency_ms=40,
        model_name="qwen3.5:9b",
    )
    ask = store.get_run("ask-1")
    assert ask is not None and ask.ask_latency_ms == 40 and ask.refused_count == 1
    assert [run.run_id for run in store.recent_runs()] == ["ask-1", "ingest-1"]
    names = {row[1] for row in store._connection().execute("PRAGMA table_info(runs)")}
    assert "text" not in names
    assert "ocr_text" not in names
    assert "caption" not in names


def test_commit_parsed_rolls_back_and_later_insert_works(store: SqliteStore):
    good = make_chunk(ordinal=0)
    bad = make_chunk(ordinal=1, embed_state="queued")
    with pytest.raises(RagError, match="embed_state"):
        store.commit_parsed(make_document(), [good, bad], [])
    assert store.get_document(DOC) is None
    store.commit_parsed(make_document(), [good], [])
    assert store.get_document(DOC) is not None
    assert store.chunks_for_document(DOC) == [good]


def test_missing_anchor_does_not_leave_a_chunk(store: SqliteStore):
    store.insert_document(make_document())
    chunk = make_chunk(anchor_chunk_id=chunk_id_for(DOC, 9))
    with pytest.raises(RagError, match="refused"):
        store.insert_chunk(chunk)
    assert store.get_chunk(chunk.chunk_id) is None
    store.insert_chunk(make_chunk())


def test_status_and_identity_rejections(store: SqliteStore):
    with pytest.raises(RagError, match="parsed"):
        store.insert_document(make_document(status="partial"))
    with pytest.raises(RagError, match="sha256"):
        store.insert_document(make_document(sha256="d" * 64))
    store.insert_document(make_document())
    with pytest.raises(RagError, match="already exists"):
        store.insert_document(make_document())
    with pytest.raises(RagError, match="path"):
        store.insert_document(make_document(DOC_B, source_path=f"/docs/{DOC}.pdf"))
    with pytest.raises(RagError) as missing:
        store.set_document_status("e" * 64, "complete")
    assert missing.value.code == EXIT_NOT_FOUND


def test_sql_checks_reject_values_the_api_also_rejects(tmp_path: Path):
    data = tmp_path / "data"
    with SqliteStore.open(data) as opened:
        opened.insert_document(make_document())
        db_path = opened.db_path
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO documents (
                doc_id, source_path, filename, media_type, byte_size, sha256,
                title, parser, parser_version, page_count, ingested_at, status
            ) VALUES (?, ?, ?, 'pdf', 1, ?, NULL, 'pymupdf', '1', NULL, ?, 'partial')
            """,
            (DOC_B, "/docs/other.pdf", "a.pdf", DOC_B, WHEN),
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, ordinal, text, kind, token_estimate, embed_state
            ) VALUES (?, ?, 0, 'x', 'text', 1, 'queued')
            """,
            (chunk_id_for(DOC, 0), DOC),
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO images (
                image_id, doc_id, storage_relpath, width, height, byte_size, sha256,
                kind, status
            ) VALUES (?, ?, 'images/x', 10, 10, 1, ?, 'embedded', 'partial')
            """,
            (image_id_for(DOC, 0), DOC, IMAGE_SHA),
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO runs (
                run_id, command, started_at, chunk_count, ocr_count, vision_count,
                refused_count, ollama_error_count, ask_latency_ms
            ) VALUES ('r', 'ingest', ?, 0, 0, 0, 0, 0, 5)
            """,
            (WHEN,),
        )
    conn.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO chunks (chunk_id, doc_id, ordinal, text, kind, token_estimate, embed_state) "
            "VALUES (?, ?, 0, 'x', 'text', 1, 'pending')",
            (chunk_id_for("f" * 64, 0), "f" * 64),
        )
        conn.commit()
    conn.close()


def test_unrecognized_schema_is_refused_without_rewrite(tmp_path: Path):
    english = EN_MESSAGES["store.schema_refused"]
    chinese = ZH_MESSAGES["store.schema_refused"]
    assert english == (
        "SQLite schema does not match this build. "
        "An unrecognized schema is not repaired. "
        "Restore from a data/ backup."
    )
    assert chinese == "SQLite 綱要不符合這個版本的程式。無法辨識的綱要不會就地修復。請從 data/ 備份還原。"
    for text in (english, chinese):
        assert "schema_version=1" not in text
        assert "不會就地改寫" not in text
    assert EN_MESSAGES["status.fts_mismatch"] == (
        "FTS row count {fts_count} does not match embedded chunks {done_count}."
    )
    assert ZH_MESSAGES["status.fts_mismatch"] == "FTS 列數 {fts_count} 與已嵌入 {done_count} 不一致。"

    data = tmp_path / "data"
    with SqliteStore.open(data) as opened:
        opened.insert_document(make_document(title=SECRET))
        orphan = data / "images" / ("d" * 64)
        orphan.mkdir()
        (orphan / "keep.txt").write_text(SECRET, encoding="utf-8")
        db_path = opened.db_path
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO schema_migrations (version) VALUES (3)")
    conn.commit()
    conn.close()
    with pytest.raises(RagError) as exc:
        SqliteStore.open(data)
    assert exc.value.code == EXIT_MISMATCH
    assert exc.value.message == english
    assert (orphan / "keep.txt").read_text(encoding="utf-8") == SECRET
    assert not (data / "corpus.sqlite.v1.bak").exists()
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT title FROM documents").fetchone()[0] == SECRET
    assert conn.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall() == [(1,), (2,), (3,)]
    conn.execute("DELETE FROM schema_migrations WHERE version = 3")
    conn.execute("ALTER TABLE runs ADD COLUMN body TEXT")
    conn.commit()
    conn.close()
    with pytest.raises(RagError) as drifted:
        SqliteStore.open(data)
    assert drifted.value.code == EXIT_MISMATCH
    assert not (data / "corpus.sqlite.v1.bak").exists()
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT title FROM documents").fetchone()[0] == SECRET
    columns = [row[1] for row in conn.execute("PRAGMA table_info(runs)")]
    assert "body" in columns
    conn.close()

    extra = tmp_path / "extra"
    with SqliteStore.open(extra) as opened:
        opened.insert_document(make_document(title=SECRET))
        extra_db = opened.db_path
    conn = sqlite3.connect(extra_db)
    conn.execute("CREATE TABLE extra (id INTEGER)")
    conn.commit()
    conn.close()
    with pytest.raises(RagError) as extra_error:
        SqliteStore.open(extra)
    assert extra_error.value.code == EXIT_MISMATCH
    assert not (extra / "corpus.sqlite.v1.bak").exists()
    conn = sqlite3.connect(extra_db)
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE name = 'extra'"
    ).fetchone()[0] == "extra"
    assert conn.execute("SELECT title FROM documents").fetchone()[0] == SECRET
    conn.close()

    missing = tmp_path / "missing"
    with SqliteStore.open(missing) as opened:
        missing_db = opened.db_path
    conn = sqlite3.connect(missing_db)
    conn.execute("DROP TABLE chunks_fts")
    conn.commit()
    conn.close()
    with pytest.raises(RagError) as missing_error:
        SqliteStore.open(missing)
    assert missing_error.value.code == EXIT_MISMATCH
    assert not (missing / "corpus.sqlite.v1.bak").exists()
    conn = sqlite3.connect(missing_db)
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    assert "chunks_fts" not in names
    assert conn.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall() == [(1,), (2,)]
    conn.close()


def test_orphan_image_dirs_are_removed_without_following_links(tmp_path: Path):
    data = tmp_path / "data"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    with SqliteStore.open(data) as opened:
        opened.insert_document(make_document())
        kept = data / "images" / DOC
        kept.mkdir()
        (kept / "page.png").write_bytes(b"png")
        orphan = data / "images" / ("9" * 64)
        orphan.mkdir()
        (orphan / "link").symlink_to(outside / "keep.txt")
        (data / "images" / "readme.txt").write_text("leave", encoding="utf-8")
        link = data / "images" / ("8" * 64)
        link.symlink_to(outside, target_is_directory=True)
    with SqliteStore.open(data) as reopened:
        assert (kept / "page.png").read_bytes() == b"png"
        assert reopened.get_document(DOC) is not None
    assert not orphan.exists()
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert (data / "images" / "readme.txt").read_text(encoding="utf-8") == "leave"
    assert link.is_symlink()


def test_image_bytes_stay_inside_and_are_private(tmp_path: Path):
    previous = os.umask(0)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"safe")
    try:
        opened = SqliteStore.open(tmp_path / "data")
        try:
            image_id = image_id_for(DOC, 0)
            with pytest.raises(RagError):
                opened.image_path(DOC, "../secret")
            with pytest.raises(RagError):
                opened.write_image("../" + DOC, image_id, b"nope")
            assert outside.read_bytes() == b"safe"
            assert not (tmp_path / "secret").exists()
            relative = opened.write_image(DOC, image_id, b"png-bytes")
            path = opened.data_dir / relative
            assert relative == f"images/{DOC}/{image_id}.png"
            assert path.read_bytes() == b"png-bytes"
            assert mode(path) == 0o600
            assert mode(path.parent) == 0o700
            assert path.resolve().is_relative_to((opened.data_dir / "images").resolve())
            linked = opened.image_path(DOC, image_id_for(DOC, 1))
            linked.parent.mkdir(exist_ok=True)
            linked.symlink_to(outside)
            with pytest.raises(RagError):
                opened.write_image(DOC, image_id_for(DOC, 1), b"pwn")
            assert outside.read_bytes() == b"safe"
            escaped = opened.data_dir / "images" / DOC
            assert escaped.is_dir() and not escaped.is_symlink()
        finally:
            opened.close()
    finally:
        os.umask(previous)


def test_image_directory_symlink_is_not_followed(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    with SqliteStore.open(tmp_path / "data") as opened:
        link = opened.data_dir / "images" / DOC
        link.symlink_to(outside, target_is_directory=True)
        with pytest.raises(RagError):
            opened.write_image(DOC, image_id_for(DOC, 0), b"nope")
        with pytest.raises(RagError):
            opened.remove_image_dir(DOC)
        assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"
        assert not (outside / f"{image_id_for(DOC, 0)}.png").exists()


def test_data_dir_symlink_is_refused(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(RagError, match="symbolic link"):
        SqliteStore.open(link)
    assert not (real / "corpus.sqlite").exists()


def test_delete_derivatives_keeps_the_document_and_delete_removes_it(store: SqliteStore):
    store.insert_document(make_document())
    store.insert_document(make_document(DOC_B, source_path="/docs/other.md", media_type="text", parser="text"))
    image = make_image()
    store.insert_image(image)
    chunk = make_chunk(image_id=image.image_id, kind="ocr", embed_state="done")
    store.insert_chunk(chunk)
    store.upsert_vector(
        VectorRecord(
            chunk_id=chunk.chunk_id,
            embed_model="qwen3-embedding:0.6b",
            native_dim=2,
            vector=(1.0, 0.0),
            num_ctx=2048,
            embedded_at=WHEN,
        )
    )
    relative = store.write_image(DOC, image.image_id, b"png")
    other = make_chunk(DOC_B, ordinal=0, text="另一份")
    store.insert_chunk(other)
    assert store.chunk_ids_for(DOC) == [chunk.chunk_id]
    removed = store.delete_derivatives(DOC)
    assert removed == [chunk.chunk_id]
    assert store.get_document(DOC) is not None
    assert store.get_document(DOC).status == "parsed"
    assert store.chunks_for_document(DOC) == []
    assert store.images_for_document(DOC) == []
    assert store.get_vector(chunk.chunk_id, "qwen3-embedding:0.6b") is None
    assert not (store.data_dir / relative).exists()
    assert store.get_chunk(other.chunk_id) == other
    store.insert_chunk(make_chunk())
    gone = store.delete_document(DOC)
    assert gone == [chunk_id_for(DOC, 0)]
    assert store.get_document(DOC) is None
    assert store.get_document(DOC_B) is not None


def test_source_path_update_and_chunk_text(store: SqliteStore):
    store.insert_document(make_document())
    chunk = make_chunk(text="舊")
    store.insert_chunk(chunk)
    store.update_source_path(DOC, "/docs/moved.pdf", "moved.pdf")
    stored = store.get_document(DOC)
    assert stored is not None
    assert stored.source_path == "/docs/moved.pdf"
    assert stored.filename == "moved.pdf"
    assert stored.doc_id == DOC
    store.update_chunk_text(chunk.chunk_id, "新描述", 4)
    updated = store.get_chunk(chunk.chunk_id)
    assert updated is not None
    assert updated.text == "新描述"
    assert updated.token_estimate == 4
    assert updated.embed_state == "pending"


def test_reopen_persists_rows(tmp_path: Path):
    data = tmp_path / "data"
    with SqliteStore.open(data) as opened:
        opened.insert_document(make_document(title=SECRET))
        opened.insert_chunk(make_chunk(text=SECRET))
    with SqliteStore.open(data) as reopened:
        stored = reopened.get_document(DOC)
        assert stored is not None and stored.title == SECRET
        assert reopened.get_chunk(chunk_id_for(DOC, 0)).text == SECRET
        assert reopened.journal_mode() == "wal"


def test_corpus_text_is_not_copied_outside_sqlite(store: SqliteStore, caplog):
    store.insert_document(make_document(title=SECRET))
    store.insert_chunk(make_chunk(text=SECRET, token_estimate=8))
    store.start_run(Run(run_id="status-1", command="status", started_at=WHEN))
    assert SECRET not in caplog.text
    for path in store.data_dir.rglob("*"):
        if not path.is_file() or path.name.startswith("corpus.sqlite"):
            continue
        assert SECRET.encode() not in path.read_bytes()


def test_zero_chunks_refresh_to_complete(store: SqliteStore):
    store.insert_document(make_document())
    assert store.refresh_document_status(DOC) == "complete"


def test_failed_parse_image_dir_is_an_orphan_on_next_open(tmp_path: Path):
    data = tmp_path / "data"
    with SqliteStore.open(data) as opened:
        relative = opened.write_image(DOC, image_id_for(DOC, 0), b"temp")
        with pytest.raises(RagError):
            opened.commit_parsed(make_document(status="complete"), [], [])
        assert (opened.data_dir / relative).is_file()
    with SqliteStore.open(data):
        pass
    assert not (data / "images" / DOC).exists()


FTS_NAMES = {
    "chunks_fts",
    "chunks_fts_config",
    "chunks_fts_content",
    "chunks_fts_data",
    "chunks_fts_docsize",
    "chunks_fts_idx",
}
CLAUSE = "根據第12條退款金額為100元。"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _master_names(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    finally:
        conn.close()


def _versions(db_path: Path) -> list[int]:
    conn = sqlite3.connect(db_path)
    try:
        return [
            int(row[0])
            for row in conn.execute("SELECT version FROM schema_migrations ORDER BY version")
        ]
    finally:
        conn.close()


def build_version_one(data: Path) -> Path:
    """Create a version 1 corpus with one done chunk and one pending chunk."""
    data.mkdir(parents=True)
    db_path = data / "corpus.sqlite"
    fd = os.open(db_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.close(fd)
    document = make_document(title=SECRET)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        for statement in _sql_statements(_SCHEMA):
            conn.execute(statement)
        conn.execute("INSERT INTO schema_migrations (version) VALUES (1)")
        conn.execute(
            """
            INSERT INTO documents (
                doc_id, source_path, filename, media_type, byte_size, sha256,
                title, parser, parser_version, page_count, ingested_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document.doc_id,
                document.source_path,
                document.filename,
                document.media_type,
                document.byte_size,
                document.sha256,
                document.title,
                document.parser,
                document.parser_version,
                document.page_count,
                document.ingested_at,
                document.status,
            ),
        )
        for ordinal, text, state in ((0, CLAUSE, "done"), (1, "waiting", "pending")):
            conn.execute(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, ordinal, text, kind, page_start, page_end,
                    section, image_id, anchor_chunk_id, token_estimate, embed_state
                ) VALUES (?, ?, ?, ?, 'text', 1, 1, NULL, NULL, NULL, 8, ?)
                """,
                (chunk_id_for(DOC, ordinal), DOC, ordinal, text, state),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def test_version_one_migrates_done_chunks_and_writes_a_private_backup(tmp_path: Path, caplog):
    caplog.set_level(logging.INFO)
    data = tmp_path / "data"
    db_path = build_version_one(data)
    with SqliteStore.open(data) as opened:
        assert opened.schema_version() == 2
        assert _versions(db_path) == [1, 2]
        assert FTS_NAMES <= _master_names(db_path)
        info = tuple(row[1] for row in opened._connection().execute("PRAGMA table_info(chunks)"))
        assert info == CHUNK_COLUMNS
        assert opened.count_fts_rows() == 1
        hits = opened.search_fts('"第12條"', 5)
        assert [item[0] for item in hits] == [chunk_id_for(DOC, 0)]
        assert opened.search_fts('"waiting"', 5) == []
        backup = data / "corpus.sqlite.v1.bak"
        assert backup.is_file()
        assert not backup.is_symlink()
        assert mode(backup) == 0o600
        assert _versions(backup) == [1]
        assert "corpus.sqlite.v1.bak" in caplog.text
        assert str(data) not in caplog.text
        assert SECRET not in caplog.text
        assert CLAUSE not in caplog.text
        digest = _digest(backup)
        inode = backup.stat().st_ino
    with SqliteStore.open(data) as again:
        assert again.schema_version() == 2
    assert _digest(backup) == digest
    assert backup.stat().st_ino == inode


def test_existing_version_one_backup_is_not_overwritten(tmp_path: Path):
    data = tmp_path / "data"
    db_path = build_version_one(data)
    backup = data / "corpus.sqlite.v1.bak"
    source = sqlite3.connect(db_path)
    try:
        dest = sqlite3.connect(backup)
        try:
            source.backup(dest)
        finally:
            dest.close()
    finally:
        source.close()
    os.chmod(backup, 0o600)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO chunks (
            chunk_id, doc_id, ordinal, text, kind, page_start, page_end,
            section, image_id, anchor_chunk_id, token_estimate, embed_state
        ) VALUES (?, ?, 2, 'later refunds', 'text', 1, 1, NULL, NULL, NULL, 4, 'done')
        """,
        (chunk_id_for(DOC, 2), DOC),
    )
    conn.commit()
    conn.close()
    digest = _digest(backup)
    with SqliteStore.open(data) as opened:
        found = {item[0] for item in opened.search_fts('"第12條" OR "refunds"', 5)}
        assert chunk_id_for(DOC, 0) in found
        assert chunk_id_for(DOC, 2) in found
        assert opened.count_fts_rows() == 2
    assert _digest(backup) == digest
    saved = sqlite3.connect(f"file:{backup.as_posix()}?mode=ro", uri=True)
    try:
        assert saved.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
        assert _versions(backup) == [1]
    finally:
        saved.close()
    with SqliteStore.open(data):
        pass
    assert _digest(backup) == digest


def _connect_proxy(*, fail_backup: bool = False, fail_backfill: bool = False):
    """Wrap connections so a test can fail backup or the FTS backfill."""
    real_connect = sqlite3.connect

    class Proxy:
        def __init__(self, inner: sqlite3.Connection) -> None:
            object.__setattr__(self, "_inner", inner)

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

        def __setattr__(self, name: str, value: object) -> None:
            setattr(self._inner, name, value)

        def execute(self, sql, parameters=(), /):
            if fail_backfill and isinstance(sql, str) and "INSERT INTO chunks_fts" in sql:
                raise sqlite3.OperationalError("injected")
            return self._inner.execute(sql, parameters)

        def backup(self, target, *args, **kwargs):
            if fail_backup:
                raise sqlite3.OperationalError("backup failed")
            inner = target._inner if isinstance(target, Proxy) else target
            return self._inner.backup(inner, *args, **kwargs)

    def connect(*args, **kwargs):
        return Proxy(real_connect(*args, **kwargs))

    return connect


@pytest.mark.parametrize("kind", ["backup", "chmod"])
def test_failed_new_backup_is_removed_and_database_stays_version_one(tmp_path, monkeypatch, kind):
    data = tmp_path / "data"
    db_path = build_version_one(data)
    backup = data / "corpus.sqlite.v1.bak"
    if kind == "backup":
        monkeypatch.setattr(
            "ragcli.store.sqlite.sqlite3.connect",
            _connect_proxy(fail_backup=True),
        )
    else:
        real_chmod = os.chmod

        def broken_chmod(path, file_mode):
            if str(path).endswith("corpus.sqlite.v1.bak"):
                raise OSError(13, "denied")
            return real_chmod(path, file_mode)

        monkeypatch.setattr("ragcli.store.sqlite.os.chmod", broken_chmod)
    with pytest.raises(RagError) as exc:
        SqliteStore.open(data)
    assert exc.value.code == EXIT_MISMATCH
    assert exc.value.message == EN_MESSAGES["store.backup_failed"]
    assert SECRET not in exc.value.message
    assert CLAUSE not in exc.value.message
    assert not backup.exists()
    assert not Path(str(backup) + "-wal").exists()
    assert not Path(str(backup) + "-shm").exists()
    assert _versions(db_path) == [1]
    assert "chunks_fts" not in _master_names(db_path)


def test_empty_existing_backup_refuses_migration(tmp_path: Path):
    data = tmp_path / "data"
    db_path = build_version_one(data)
    backup = data / "corpus.sqlite.v1.bak"
    backup.write_bytes(b"")
    with pytest.raises(RagError) as exc:
        SqliteStore.open(data)
    assert exc.value.code == EXIT_MISMATCH
    assert exc.value.message == EN_MESSAGES["store.backup_failed"]
    assert backup.read_bytes() == b""
    assert _versions(db_path) == [1]
    assert "chunks_fts" not in _master_names(db_path)


def test_invalid_existing_backup_is_left_in_place(tmp_path: Path):
    data = tmp_path / "data"
    db_path = build_version_one(data)
    backup = data / "corpus.sqlite.v1.bak"
    backup.write_bytes(b"not a database")
    digest = _digest(backup)
    with pytest.raises(RagError) as exc:
        SqliteStore.open(data)
    assert exc.value.code == EXIT_MISMATCH
    assert _digest(backup) == digest
    assert _versions(db_path) == [1]
    assert "chunks_fts" not in _master_names(db_path)


def test_backup_symlink_is_not_followed(tmp_path: Path):
    data = tmp_path / "data"
    db_path = build_version_one(data)
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"keep-me")
    backup = data / "corpus.sqlite.v1.bak"
    backup.symlink_to(sentinel)
    with pytest.raises(RagError) as exc:
        SqliteStore.open(data)
    assert exc.value.code == EXIT_MISMATCH
    assert exc.value.message == EN_MESSAGES["store.backup_failed"]
    assert sentinel.read_bytes() == b"keep-me"
    assert backup.is_symlink()
    assert _versions(db_path) == [1]
    assert "chunks_fts" not in _master_names(db_path)


def test_failed_migration_rolls_back_without_shadow_tables(tmp_path, monkeypatch):
    data = tmp_path / "data"
    db_path = build_version_one(data)
    monkeypatch.setattr(
        "ragcli.store.sqlite.sqlite3.connect",
        _connect_proxy(fail_backfill=True),
    )
    with pytest.raises(sqlite3.OperationalError):
        SqliteStore.open(data)
    assert _versions(db_path) == [1]
    assert "chunks_fts" not in _master_names(db_path)
    monkeypatch.undo()
    with SqliteStore.open(data) as opened:
        assert opened.schema_version() == 2
        assert opened.count_fts_rows() == 1


def test_fts_follows_embed_state(store: SqliteStore):
    store.insert_document(make_document())
    blocked = make_chunk(ordinal=0, text="refunds", embed_state="blocked")
    store.insert_chunk(blocked)
    store.update_chunk_text(blocked.chunk_id, "updated", 4)
    assert store.count_fts_rows() == 0
    assert store.search_fts('"updated"', 5) == []

    ready = make_chunk(ordinal=1, text="refunds", embed_state="pending")
    store.insert_chunk(ready)
    store.set_embed_state(ready.chunk_id, "done")
    store.set_embed_state(ready.chunk_id, "done")
    assert store.count_fts_rows() == 1
    assert [item[0] for item in store.search_fts('"refunds"', 5)] == [ready.chunk_id]
    store.update_chunk_text(ready.chunk_id, "beta token", 4)
    assert store.count_fts_rows() == 1
    assert store.search_fts('"refunds"', 5) == []
    assert [item[0] for item in store.search_fts('"beta"', 5)] == [ready.chunk_id]

    store.set_embed_state(ready.chunk_id, "pending")
    assert store.search_fts('"beta"', 5) == []
    store.set_embed_state(ready.chunk_id, "done")
    store.delete_chunk(ready.chunk_id)
    assert store.search_fts('"beta"', 5) == []

    other = make_chunk(DOC_B, ordinal=0, text="refunds", embed_state="done")
    store.insert_document(make_document(DOC_B, source_path="/docs/other.md", media_type="text", parser="text"))
    store.insert_chunk(other)
    assert store.count_fts_rows() == 1
    store.delete_document(DOC_B)
    assert store.search_fts('"refunds"', 5) == []
    assert store.count_fts_rows() == 0


def test_chunks_by_ordinals_returns_done_rows_for_one_document(store: SqliteStore):
    store.insert_document(make_document())
    store.insert_document(
        make_document(
            DOC_B,
            source_path="/docs/other.md",
            media_type="text",
            parser="text",
        )
    )
    anchor = make_chunk(ordinal=0, text="ANCHOR-FAR", embed_state="done")
    left = make_chunk(ordinal=1, text="LEFT", embed_state="done", page_start=2, page_end=2)
    figure = make_chunk(
        ordinal=2,
        text="FIG",
        kind="figure",
        embed_state="done",
        page_start=4,
        page_end=4,
        anchor_chunk_id=anchor.chunk_id,
    )
    pending = make_chunk(ordinal=3, text="PENDING", embed_state="pending")
    other = make_chunk(DOC_B, ordinal=1, text="OTHER", embed_state="done")
    for chunk in (anchor, left, figure, pending, other):
        store.insert_chunk(chunk)

    assert store.chunks_by_ordinals(DOC, []) == []
    assert [chunk.ordinal for chunk in store.chunks_by_ordinals(DOC, [3, 1, 9, 1])] == [1]
    assert [chunk.ordinal for chunk in store.chunks_by_ordinals(DOC, [2])] == [2]
    assert [chunk.ordinal for chunk in store.chunks_by_ordinals(DOC, [2, 0])] == [0, 2]
    assert store.chunks_by_ordinals(DOC, [2])[0].kind == "figure"
    assert store.chunks_by_ordinals(DOC, [2])[0].text == "FIG"
    assert [chunk.text for chunk in store.chunks_by_ordinals(DOC_B, [1, 2])] == ["OTHER"]


def test_search_fts_error_omits_the_query(store: SqliteStore, caplog):
    caplog.set_level(logging.WARNING)
    store.insert_document(make_document())
    store.insert_chunk(make_chunk(text=SECRET, embed_state="done"))
    assert store.search_fts("1,200", 5) == []
    assert EN_MESSAGES["log.fts_failed"] in caplog.text
    assert SECRET not in caplog.text
    assert "1,200" not in caplog.text
