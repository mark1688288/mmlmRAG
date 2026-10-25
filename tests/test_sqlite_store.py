"""SQLite schema, permissions, and the corpus store. No parser and no Ollama."""

from __future__ import annotations

import os
import sqlite3
import stat
import struct
from pathlib import Path

import pytest

from ragcli.errors import EXIT_MISMATCH, EXIT_NOT_FOUND, EXIT_UNCLASSIFIED, RagError
from ragcli.store.sqlite import (
    SCHEMA_VERSION,
    ActiveIndex,
    Chunk,
    Document,
    ImageRecord,
    Run,
    SqliteStore,
    VectorRecord,
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
    assert SCHEMA_VERSION == 1
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
            assert opened.schema_version() == 1
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
    assert found == set(expected)
    for table, columns in expected.items():
        info = tuple(row[1] for row in conn.execute(f"PRAGMA table_info({table})"))
        assert info == columns
    versions = [row[0] for row in conn.execute("SELECT version FROM schema_migrations")]
    assert versions == [1]
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
    with pytest.raises(RagError, match="拒絕"):
        store.insert_chunk(chunk)
    assert store.get_chunk(chunk.chunk_id) is None
    store.insert_chunk(make_chunk())


def test_status_and_identity_rejections(store: SqliteStore):
    with pytest.raises(RagError, match="parsed"):
        store.insert_document(make_document(status="partial"))
    with pytest.raises(RagError, match="sha256"):
        store.insert_document(make_document(sha256="d" * 64))
    store.insert_document(make_document())
    with pytest.raises(RagError, match="已存在"):
        store.insert_document(make_document())
    with pytest.raises(RagError, match="路徑"):
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


def test_newer_or_drifted_schema_is_refused_without_rewrite(tmp_path: Path):
    data = tmp_path / "data"
    with SqliteStore.open(data) as opened:
        opened.insert_document(make_document(title=SECRET))
        orphan = data / "images" / ("d" * 64)
        orphan.mkdir()
        (orphan / "keep.txt").write_text(SECRET, encoding="utf-8")
        db_path = opened.db_path
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO schema_migrations (version) VALUES (2)")
    conn.commit()
    conn.close()
    with pytest.raises(RagError) as exc:
        SqliteStore.open(data)
    assert exc.value.code == EXIT_MISMATCH
    assert "備份" in exc.value.message
    assert "不會就地改寫" in exc.value.message
    assert (orphan / "keep.txt").read_text(encoding="utf-8") == SECRET
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT title FROM documents").fetchone()[0] == SECRET
    assert conn.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall() == [(1,), (2,)]
    conn.execute("DELETE FROM schema_migrations WHERE version = 2")
    conn.execute("ALTER TABLE runs ADD COLUMN body TEXT")
    conn.commit()
    conn.close()
    with pytest.raises(RagError) as drifted:
        SqliteStore.open(data)
    assert drifted.value.code == EXIT_MISMATCH
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT title FROM documents").fetchone()[0] == SECRET
    columns = [row[1] for row in conn.execute("PRAGMA table_info(runs)")]
    assert "body" in columns
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
    with pytest.raises(RagError, match="符號連結"):
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
