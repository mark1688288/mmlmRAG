"""Reembed, collection switching, delete, and status. No ingest pipeline and no live Ollama."""

from __future__ import annotations

import json
import logging
import stat
from pathlib import Path

import httpx
import pytest

from ragcli.config import Config, load_config
from ragcli.errors import (
    EXIT_DISK,
    EXIT_MISMATCH,
    EXIT_MODEL_MISSING,
    EXIT_NO_COLLECTION,
    EXIT_NOT_FOUND,
    EXIT_OK,
    EXIT_OLLAMA_UNREACHABLE,
    EXIT_UNCLASSIFIED,
    EXIT_USAGE,
    RagError,
)
from ragcli.reembed import native_width, plan_reembed, run_delete, run_reembed, run_status, run_use
from ragcli.store.chroma import ChromaIndex, collection_name
from ragcli.store.sqlite import ActiveIndex, Run, SqliteStore, VectorRecord
from test_ollama_client import EMBED, FakeOllama
from test_sqlite_store import DOC, DOC_B, WHEN, make_chunk, make_document, make_image

TINY = "lab-embed:tiny"
SECRET = "機密退款句子XYZ"
VECTOR = (3.0, 0.0, 0.0, 4.0)
LATER = "2026-10-26T00:00:00+00:00"
PREFIX = "查詢前綴是為 Qwen3-Embedding 寫的，這個模型可能不適用。"


def write_rag(
    directory: Path,
    *,
    batch_size: int = 8,
    dimensions: int = 4,
    native: int = 4,
    min_free: int = 1,
) -> Config:
    path = directory / "rag.toml"
    path.write_text(
        "\n".join(
            [
                "[paths]",
                'data_dir = "data"',
                "[embed]",
                f'model = "{EMBED}"',
                f"native_dimensions = {native}",
                f"dimensions = {dimensions}",
                "num_ctx = 2048",
                f"batch_size = {batch_size}",
                "[limits]",
                f"min_free_bytes = {min_free}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return load_config(path)


def refuse_ollama(*_args, **_kwargs):
    raise AssertionError("不應建立 OllamaClient")


def expect(func, code: int, message: str) -> RagError:
    with pytest.raises(RagError) as caught:
        func()
    assert caught.value.code == code
    assert message in caught.value.message
    return caught.value


def add_document(store: SqliteStore, doc_id: str = DOC, source: str | None = None) -> None:
    store.insert_document(
        make_document(doc_id=doc_id, source_path=source or f"/missing/{doc_id}.pdf")
    )


def add_chunk(store, doc_id, ordinal, text, **overrides):
    chunk = make_chunk(doc_id=doc_id, ordinal=ordinal, text=text, **overrides)
    store.insert_chunk(chunk)
    return chunk


def add_vector(store, chunk, vector=VECTOR, model=EMBED, num_ctx=2048, at=WHEN) -> None:
    store.upsert_vector(
        VectorRecord(
            chunk_id=chunk.chunk_id,
            embed_model=model,
            native_dim=len(vector),
            vector=vector,
            num_ctx=num_ctx,
            embedded_at=at,
        )
    )


class _Store:
    def __init__(self, config: Config) -> None:
        self.config = config

    def __enter__(self) -> SqliteStore:
        self.store = SqliteStore.open(self.config.data_dir)
        return self.store

    def __exit__(self, *exc: object) -> None:
        self.store.close()


def opened(config: Config) -> _Store:
    return _Store(config)


def active_of(config: Config) -> ActiveIndex | None:
    with opened(config) as store:
        return store.get_active()


def latest_run(config: Config) -> Run:
    with opened(config) as store:
        runs = store.recent_runs()
    assert runs
    return runs[0]


def collection_names(config: Config) -> list[str]:
    index = ChromaIndex.open(config.data_dir)
    try:
        return index.collection_names()
    finally:
        index.close()


def embedding_of(config: Config, chunk_id: str, model: str, dimensions: int, num_ctx: int):
    index = ChromaIndex.open(config.data_dir)
    try:
        found = index.get_embeddings(
            [chunk_id],
            model=model,
            dimensions=dimensions,
            embed_num_ctx=num_ctx,
        )
    finally:
        index.close()
    return found[chunk_id]


def chunk_ids(config: Config, name: str) -> list[str]:
    index = ChromaIndex.open(config.data_dir)
    try:
        return index.chunk_ids(name)
    finally:
        index.close()


class SwitchFake(FakeOllama):
    """One native-width vector per input text. Release still clears ps."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.width = width
        self.tag_names = [EMBED, TINY, "qwen3.5:9b", "qwen3.5:4b"]
        self.vector = [float(index + 1) for index in range(width)]

    def _embed(self, body: dict) -> httpx.Response:
        if body.get("input") == ".":
            self._drop(body["model"])
            return httpx.Response(200, json={"embeddings": [[0.0]]})
        texts = list(body["input"])
        self._show(body["model"], body["options"]["num_ctx"], body.get("keep_alive"))
        return httpx.Response(
            200,
            json={"embeddings": [list(self.vector) for _text in texts]},
        )


def document_bodies(fake: FakeOllama) -> list[dict]:
    return [
        body
        for body in fake.bodies("/api/embed")
        if isinstance(body, dict) and isinstance(body.get("input"), list)
    ]


def test_plan_same_model_short_width_does_not_call_ollama():
    plan = plan_reembed(
        stored_model=EMBED,
        stored_native_dim=4,
        target_model=EMBED,
        target_dimensions=2,
        target_native_dim=4,
    )
    assert plan.calls_ollama is False
    assert plan.reparse is False
    assert plan.dimensions == 2
    assert plan.native_dim == 4


def test_plan_other_model_native_width_is_not_a_mismatch():
    plan = plan_reembed(
        stored_model=EMBED,
        stored_native_dim=1024,
        target_model="qwen3-embedding:4b",
        target_dimensions=2560,
        target_native_dim=2560,
    )
    assert plan.calls_ollama is True
    assert plan.reparse is False
    assert plan.native_dim == 2560
    assert plan.dimensions == 2560


def test_plan_same_model_wider_than_native_is_exit_12():
    expect(
        lambda: plan_reembed(
            stored_model=EMBED,
            stored_native_dim=4,
            target_model=EMBED,
            target_dimensions=8,
            target_native_dim=4,
        ),
        EXIT_MISMATCH,
        "同一模型的維度大於已儲存的 native_dim。",
    )


def test_native_width_uses_config_for_the_active_model_and_known_tags(tmp_path):
    config = write_rag(tmp_path)
    assert native_width(config, EMBED, None) == 4
    assert native_width(config, "qwen3-embedding:4b", None) == 2560
    assert native_width(config, "qwen3-embedding:8b", None) == 4096
    assert native_width(config, TINY, 8) == 8
    expect(lambda: native_width(config, TINY, None), EXIT_USAGE, "--native-dimensions")
    expect(
        lambda: native_width(config, "qwen3-embedding:4b", 8),
        EXIT_MISMATCH,
        "--native-dimensions 與這個模型的原生寬度不一致。",
    )


def test_rebuild_same_width_copies_stored_floats_without_ollama_or_l2(tmp_path, monkeypatch, capsys):
    config = write_rag(tmp_path)
    source = tmp_path / "gone.pdf"
    monkeypatch.setattr("ragcli.reembed.OllamaClient", refuse_ollama)
    logger = logging.getLogger("ragcli")
    previous = logger.level
    logger.setLevel(logging.INFO)
    try:
        with opened(config) as store:
            add_document(store, source=str(source))
            chunk = add_chunk(store, DOC, 0, SECRET, embed_state="done")
            add_vector(store, chunk)
        code = run_reembed(config, clock=lambda: LATER)
    finally:
        logger.setLevel(previous)
    captured = capsys.readouterr()
    name = collection_name(EMBED, 4)
    assert code == EXIT_OK
    assert not source.exists()
    assert f"已重建 {name}，區塊 1。" in captured.out
    assert SECRET not in captured.out
    assert SECRET not in captured.err
    assert not (config.data_dir / "locks" / "ollama.lock").exists()
    log = config.data_dir / "logs" / "rag.log"
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    logged = log.read_text(encoding="utf-8")
    assert SECRET not in logged
    assert "重新嵌入" in logged
    stored = embedding_of(config, chunk.chunk_id, EMBED, 4, 2048)
    assert stored == (3.0, 0.0, 0.0, 4.0)
    with opened(config) as store:
        vector = store.get_vector(chunk.chunk_id, EMBED)
    assert vector is not None
    assert vector.vector == VECTOR
    assert vector.embedded_at == WHEN
    pointer = active_of(config)
    assert pointer is not None
    assert pointer.collection == name
    assert pointer.embed_model == EMBED
    assert pointer.dimensions == 4
    assert pointer.embed_num_ctx == 2048
    run = latest_run(config)
    assert run.command == "reembed"
    assert run.exit_code == 0
    assert run.chunk_count == 1
    assert run.model_name is None
    assert run.ask_latency_ms is None
    assert run.refused_count == 0
    assert run.ollama_error_count == 0


def test_shorter_width_truncates_and_normalizes_without_ollama(tmp_path, monkeypatch, capsys):
    config = write_rag(tmp_path)
    monkeypatch.setattr("ragcli.reembed.OllamaClient", refuse_ollama)
    with opened(config) as store:
        add_document(store)
        chunk = add_chunk(store, DOC, 0, "退款規定", embed_state="done")
        add_vector(store, chunk, num_ctx=1024)
    code = run_reembed(config, dimensions=2, clock=lambda: LATER)
    assert code == EXIT_OK
    assert "已重建" in capsys.readouterr().out
    assert embedding_of(config, chunk.chunk_id, EMBED, 2, 1024) == (1.0, 0.0)
    with opened(config) as store:
        vector = store.get_vector(chunk.chunk_id, EMBED)
    assert vector is not None
    assert vector.vector == VECTOR
    assert vector.embedded_at == WHEN
    pointer = active_of(config)
    assert pointer is not None
    assert pointer.collection == collection_name(EMBED, 2)
    assert pointer.embed_num_ctx == 1024
    assert not (config.data_dir / "locks" / "ollama.lock").exists()


def test_missing_native_blob_does_not_call_ollama(tmp_path, monkeypatch):
    config = write_rag(tmp_path)
    monkeypatch.setattr("ragcli.reembed.OllamaClient", refuse_ollama)
    with opened(config) as store:
        add_document(store)
        add_chunk(store, DOC, 0, "尚未嵌入", embed_state="pending")
    expect(lambda: run_reembed(config), EXIT_MISMATCH, "缺少原生向量。")
    assert not (config.data_dir / "locks" / "ollama.lock").exists()

    partial = tmp_path / "partial"
    partial.mkdir()
    other = write_rag(partial)
    monkeypatch.setattr("ragcli.reembed.OllamaClient", refuse_ollama)
    with opened(other) as store:
        add_document(store)
        first = add_chunk(store, DOC, 0, "有向量", embed_state="done")
        add_chunk(store, DOC, 1, "沒向量", embed_state="pending")
        add_vector(store, first)
    expect(lambda: run_reembed(other), EXIT_MISMATCH, "缺少原生向量。")
    assert collection_name(EMBED, 4) not in collection_names(other)


def test_token_estimate_above_num_ctx_is_exit_12(tmp_path, monkeypatch):
    config = write_rag(tmp_path)
    monkeypatch.setattr("ragcli.reembed.OllamaClient", refuse_ollama)
    with opened(config) as store:
        add_document(store)
        chunk = add_chunk(store, DOC, 0, SECRET, embed_state="done", token_estimate=3000)
        add_vector(store, chunk, num_ctx=2048)
    expect(lambda: run_reembed(config), EXIT_MISMATCH, "區塊可能已被舊的上下文裁斷。")
    assert collection_names(config) == []
    assert latest_run(config).exit_code == EXIT_MISMATCH
    assert latest_run(config).model_name is None


def test_same_model_wider_than_stored_native_is_exit_12(tmp_path, monkeypatch):
    config = write_rag(tmp_path)
    monkeypatch.setattr("ragcli.reembed.OllamaClient", refuse_ollama)
    with opened(config) as store:
        add_document(store)
        chunk = add_chunk(store, DOC, 0, "退款規定", embed_state="done")
        add_vector(store, chunk)
    expect(
        lambda: run_reembed(config, dimensions=8),
        EXIT_MISMATCH,
        "同一模型的維度大於已儲存的 native_dim。",
    )
    assert collection_name(EMBED, 8) not in collection_names(config)
    assert not (config.data_dir / "locks" / "ollama.lock").exists()


def test_zero_vector_cannot_be_truncated(tmp_path, monkeypatch):
    config = write_rag(tmp_path)
    monkeypatch.setattr("ragcli.reembed.OllamaClient", refuse_ollama)
    with opened(config) as store:
        add_document(store)
        chunk = add_chunk(store, DOC, 0, "零", embed_state="done")
        add_vector(store, chunk, vector=(0.0, 0.0, 0.0, 0.0))
    expect(lambda: run_reembed(config, dimensions=2), EXIT_MISMATCH, "零向量不能截斷。")
    assert collection_name(EMBED, 2) not in collection_names(config)
    code = run_reembed(config, clock=lambda: LATER)
    assert code == EXIT_OK
    assert embedding_of(config, chunk.chunk_id, EMBED, 4, 2048) == (0.0, 0.0, 0.0, 0.0)


def test_model_switch_embeds_plain_text_and_keeps_old_vectors(tmp_path, monkeypatch, capsys):
    config = write_rag(tmp_path, batch_size=1)
    source = tmp_path / "gone.pdf"
    monkeypatch.setattr("ragcli.reembed.OllamaClient", refuse_ollama)
    with opened(config) as store:
        add_document(store, source=str(source))
        done = add_chunk(store, DOC, 0, SECRET, embed_state="done")
        blocked = add_chunk(
            store, DOC, 2, "這段描述未完成", embed_state="blocked", kind="figure"
        )
        add_chunk(store, DOC, 3, "", embed_state="pending", token_estimate=1)
        add_vector(store, done)
    assert run_reembed(config, clock=lambda: LATER) == EXIT_OK
    capsys.readouterr()
    with opened(config) as store:
        pending = add_chunk(store, DOC, 1, "第二段待嵌入", embed_state="pending")
    fake = SwitchFake(width=8)
    code = run_reembed(
        config,
        model=TINY,
        native_dimensions=8,
        client=fake.client(),
        clock=lambda: LATER,
    )
    captured = capsys.readouterr()
    assert code == EXIT_OK
    assert not source.exists()
    assert PREFIX in captured.err
    assert SECRET not in captured.out
    assert SECRET not in captured.err
    assert (config.data_dir / "locks" / "ollama.lock").exists()
    assert fake.loaded == []
    bodies = document_bodies(fake)
    assert [body["input"] for body in bodies] == [[SECRET], ["第二段待嵌入"]]
    for body in bodies:
        assert body["truncate"] is False
        assert "dimensions" not in body
        assert body["keep_alive"] == "30m"
        assert body["options"] == {"num_ctx": 2048}
        assert "Instruct" not in json.dumps(body)
    blob = json.dumps(fake.seen, ensure_ascii=False)
    assert "這段描述未完成" not in blob
    assert SECRET not in (config.data_dir / "logs" / "rag.log").read_text(encoding="utf-8")
    new_name = collection_name(TINY, 8)
    old_name = collection_name(EMBED, 4)
    assert old_name in collection_names(config)
    assert new_name in collection_names(config)
    assert embedding_of(config, done.chunk_id, EMBED, 4, 2048) == VECTOR
    assert embedding_of(config, done.chunk_id, TINY, 8, 2048) == pytest.approx(tuple(fake.vector))
    with opened(config) as store:
        old = store.get_vector(done.chunk_id, EMBED)
        new = store.get_vector(done.chunk_id, TINY)
        pending_row = store.get_chunk(pending.chunk_id)
        blocked_row = store.get_chunk(blocked.chunk_id)
        document = store.get_document(DOC)
    assert old is not None and old.vector == VECTOR and old.embedded_at == WHEN
    assert new is not None and new.native_dim == 8 and new.vector == tuple(fake.vector)
    assert pending_row is not None and pending_row.embed_state == "done"
    assert blocked_row is not None and blocked_row.embed_state == "blocked"
    assert store_vector_absent(config, blocked.chunk_id, TINY)
    assert document is not None and document.status == "parsed"
    pointer = active_of(config)
    assert pointer is not None and pointer.collection == new_name
    with opened(config) as store:
        switched = [run for run in store.recent_runs() if run.model_name == TINY]
    assert len(switched) == 1
    run = switched[0]
    assert run.model_name == TINY
    assert run.chunk_count == 2
    assert run.ollama_error_count == 0
    assert run.ask_latency_ms is None
    assert run.refused_count == 0


def store_vector_absent(config: Config, chunk_id: str, model: str) -> bool:
    with opened(config) as store:
        return store.get_vector(chunk_id, model) is None


def test_model_switch_missing_from_tags_is_exit_7(tmp_path, capsys):
    config = write_rag(tmp_path)
    with opened(config) as store:
        add_document(store)
        chunk = add_chunk(store, DOC, 0, SECRET, embed_state="done")
        add_vector(store, chunk)
    fake = FakeOllama()
    expect(
        lambda: run_reembed(
            config,
            model=TINY,
            native_dimensions=8,
            client=fake.client(),
            clock=lambda: LATER,
        ),
        EXIT_MODEL_MISSING,
        f"缺少模型 {TINY}。請執行 rag models pull。",
    )
    captured = capsys.readouterr()
    assert PREFIX in captured.err
    assert "ollama pull" not in captured.err
    assert "/api/embed" not in fake.paths()
    assert SECRET not in json.dumps(fake.seen, ensure_ascii=False)
    assert store_vector_absent(config, chunk.chunk_id, TINY)
    assert active_of(config) is None
    assert latest_run(config).exit_code == EXIT_MODEL_MISSING
    assert latest_run(config).model_name is None


def test_wrong_width_from_ollama_is_exit_12_and_unloads(tmp_path):
    config = write_rag(tmp_path)
    with opened(config) as store:
        add_document(store)
        chunk = add_chunk(store, DOC, 0, SECRET, embed_state="done")
        add_vector(store, chunk)
        store.set_active(
            ActiveIndex(
                collection=collection_name(EMBED, 4),
                embed_model=EMBED,
                dimensions=4,
                embed_num_ctx=2048,
            )
        )
    fake = FakeOllama()
    fake.tag_names.append(TINY)
    expect(
        lambda: run_reembed(
            config,
            model=TINY,
            native_dimensions=8,
            client=fake.client(),
            clock=lambda: LATER,
        ),
        EXIT_MISMATCH,
        "嵌入長度不是這個模型的原生寬度。",
    )
    assert fake.loaded == []
    assert store_vector_absent(config, chunk.chunk_id, TINY)
    pointer = active_of(config)
    assert pointer is not None and pointer.embed_model == EMBED
    assert collection_name(TINY, 8) not in collection_names(config)
    run = latest_run(config)
    assert run.exit_code == EXIT_MISMATCH
    assert run.ollama_error_count == 1
    assert run.model_name == TINY
    assert SECRET not in (config.data_dir / "logs" / "rag.log").read_text(encoding="utf-8")


def test_low_disk_does_not_call_ollama(tmp_path):
    config = write_rag(tmp_path, min_free=10**18)
    fake = FakeOllama()
    error = expect(
        lambda: run_reembed(config, client=fake.client(), free_bytes=lambda: 0),
        EXIT_DISK,
        f"磁碟剩餘 0 位元組，少於 {10**18}。",
    )
    assert error.code == EXIT_DISK
    assert fake.seen == []
    assert not config.data_dir.exists()


def test_no_activate_and_drop_old(tmp_path, monkeypatch, capsys):
    config = write_rag(tmp_path)
    monkeypatch.setattr("ragcli.reembed.OllamaClient", refuse_ollama)
    wide = collection_name(EMBED, 4)
    short = collection_name(EMBED, 2)
    with opened(config) as store:
        add_document(store)
        chunk = add_chunk(store, DOC, 0, "退款規定", embed_state="done")
        add_vector(store, chunk)
    assert run_reembed(config, clock=lambda: LATER) == EXIT_OK
    capsys.readouterr()
    assert run_reembed(config, drop_old=True, clock=lambda: LATER) == EXIT_OK
    same = capsys.readouterr()
    assert "已刪除集合" not in same.out
    assert wide in collection_names(config)
    assert run_reembed(config, dimensions=2, activate=False, clock=lambda: LATER) == EXIT_OK
    built = capsys.readouterr()
    assert "作用中指標未改" in built.out
    assert active_of(config) is not None and active_of(config).collection == wide
    assert short in collection_names(config)
    assert run_reembed(config, dimensions=2, activate=False, drop_old=True, clock=lambda: LATER) == EXIT_OK
    dropped = capsys.readouterr()
    assert f"已刪除集合 {wide}" in dropped.out
    assert "作用中指標仍指向已刪除的集合。" in dropped.err
    assert wide not in collection_names(config)
    assert short in collection_names(config)
    expect(lambda: run_use(config, wide), EXIT_NO_COLLECTION, "尚未有這個集合。")
    assert active_of(config) is not None and active_of(config).collection == wide
    assert run_use(config, short, clock=lambda: LATER) == EXIT_OK
    pointer = active_of(config)
    assert pointer is not None
    assert pointer.collection == short
    assert pointer.dimensions == 2
    with opened(config) as store:
        assert store.recent_runs()
        assert all(run.command != "use" for run in store.recent_runs())


def test_empty_corpus_creates_collection_without_ollama(tmp_path, monkeypatch, capsys):
    config = write_rag(tmp_path)
    monkeypatch.setattr("ragcli.reembed.OllamaClient", refuse_ollama)
    expect(
        lambda: run_reembed(config, model=TINY),
        EXIT_USAGE,
        "未知嵌入模型要同時給 --native-dimensions。",
    )
    assert not config.data_dir.exists()
    code = run_reembed(config, clock=lambda: LATER)
    out = capsys.readouterr().out
    name = collection_name(EMBED, 4)
    assert code == EXIT_OK
    assert f"已重建 {name}，區塊 0。" in out
    assert name in collection_names(config)
    assert active_of(config) is not None
    assert not (config.data_dir / "locks" / "ollama.lock").exists()
    assert latest_run(config).model_name is None


def test_use_switches_pointer_and_rejects_metadata_mismatch(tmp_path, monkeypatch):
    config = write_rag(tmp_path)
    monkeypatch.setattr("ragcli.reembed.OllamaClient", refuse_ollama)
    name = collection_name(EMBED, 4)
    index = ChromaIndex.open(config.data_dir)
    try:
        assert index.upsert([], model=EMBED, dimensions=4, embed_num_ctx=1024) == name
    finally:
        index.close()
    assert run_use(config, name) == EXIT_OK
    pointer = active_of(config)
    assert pointer == ActiveIndex(
        collection=name,
        embed_model=EMBED,
        dimensions=4,
        embed_num_ctx=1024,
    )
    with opened(config) as store:
        assert store.recent_runs() == []

    class Mismatch:
        def __init__(self) -> None:
            self.name = name
            self.metadata = {"embed_model": EMBED, "dimensions": 8, "embed_num_ctx": 2048}
            self._embedding_function = None

    class MismatchClient:
        def get_collection(self, name, embedding_function=None):
            assert embedding_function is None
            return Mismatch()

        def close(self) -> None:
            return None

    def factory(_path, settings):
        assert settings.anonymized_telemetry is False
        return MismatchClient()

    expect(
        lambda: run_use(config, name, client_factory=factory),
        EXIT_MISMATCH,
        "集合名稱與 metadata 不一致。",
    )
    assert active_of(config) == pointer
    expect(lambda: run_use(config, "emb-missing-d4"), EXIT_NO_COLLECTION, "尚未有這個集合。")
    assert active_of(config) == pointer


def test_delete_removes_sqlite_images_and_every_collection(tmp_path, monkeypatch, capsys):
    config = write_rag(tmp_path)
    monkeypatch.setattr("ragcli.reembed.OllamaClient", refuse_ollama)
    note = tmp_path / "note.txt"
    note.write_text("x", encoding="utf-8")
    stored_path = str(note.resolve())
    with opened(config) as store:
        add_document(store, DOC, stored_path)
        add_document(store, DOC_B, f"/missing/{DOC_B}.pdf")
        image = make_image(status="caption_failed")
        store.insert_image(image)
        store.write_image(DOC, image.image_id, b"png-bytes")
        first = add_chunk(store, DOC, 0, SECRET, embed_state="done")
        second = add_chunk(store, DOC_B, 0, "另一份", embed_state="done")
        add_vector(store, first)
        add_vector(store, second)
        image_dir = store.data_dir / "images" / DOC
    assert image_dir.is_dir()
    assert run_reembed(config, clock=lambda: LATER) == EXIT_OK
    assert run_reembed(config, dimensions=2, clock=lambda: LATER) == EXIT_OK
    capsys.readouterr()
    wide = collection_name(EMBED, 4)
    short = collection_name(EMBED, 2)
    expect(lambda: run_delete(config, doc_id="abc"), EXIT_USAGE, "doc-id 必須是 64 位小寫十六進位。")
    expect(lambda: run_delete(config, doc_id="D" * 64), EXIT_USAGE, "doc-id 必須是 64 位小寫十六進位。")
    expect(lambda: run_delete(config, doc_id="d" * 64), EXIT_NOT_FOUND, "找不到文件。")
    expect(lambda: run_delete(config, path=tmp_path / "missing.txt"), EXIT_NOT_FOUND, "找不到文件。")
    alias = tmp_path / "nested" / ".." / note.name
    code = run_delete(config, path=alias, clock=lambda: LATER)
    captured = capsys.readouterr()
    assert code == EXIT_OK
    assert captured.out.strip() == f"已刪除 {DOC}"
    assert "note.txt" not in captured.out
    assert SECRET not in captured.out
    assert not image_dir.exists()
    assert first.chunk_id not in chunk_ids(config, wide)
    assert first.chunk_id not in chunk_ids(config, short)
    assert chunk_ids(config, wide) == [second.chunk_id]
    assert chunk_ids(config, short) == [second.chunk_id]
    with opened(config) as store:
        assert store.get_document(DOC) is None
        assert store.get_document(DOC_B) is not None
    assert run_delete(config, doc_id=DOC_B, clock=lambda: LATER) == EXIT_OK
    assert capsys.readouterr().out.strip() == f"已刪除 {DOC_B}"
    with opened(config) as store:
        assert store.get_document(DOC_B) is None
        deletes = [run for run in store.recent_runs() if run.command == "delete" and run.exit_code == 0]
    assert chunk_ids(config, wide) == []
    assert len(deletes) == 2
    assert all(run.model_name is None and run.ask_latency_ms is None and run.refused_count == 0 for run in deletes)


def test_delete_keeps_sqlite_when_chroma_fails(tmp_path):
    config = write_rag(tmp_path)
    with opened(config) as store:
        add_document(store)
        add_chunk(store, DOC, 0, SECRET, embed_state="done")
        image = make_image()
        store.insert_image(image)
        store.write_image(DOC, image.image_id, b"png-bytes")
        image_dir = store.data_dir / "images" / DOC

    class BoomCollection:
        def __init__(self, name: str) -> None:
            self.name = name

        def delete(self, *, ids) -> None:
            raise RuntimeError("boom")

    class BoomClient:
        def list_collections(self, limit=None, offset=0):
            if offset:
                return []
            return [BoomCollection("emb-qwen3-embedding-0.6b-d4")]

        def get_collection(self, name, embedding_function=None):
            assert embedding_function is None
            return BoomCollection(name)

        def close(self) -> None:
            return None

    def factory(_path, settings):
        assert settings.anonymized_telemetry is False
        return BoomClient()

    expect(
        lambda: run_delete(config, doc_id=DOC, client_factory=factory),
        EXIT_UNCLASSIFIED,
        "向量索引刪除失敗，文件仍在。",
    )
    assert image_dir.is_dir()
    with opened(config) as store:
        assert store.get_document(DOC) is not None
        assert store.get_chunk(add_chunk_id()) is not None


def add_chunk_id() -> str:
    from ragcli.store.sqlite import chunk_id_for

    return chunk_id_for(DOC, 0)


def test_delete_without_chunks_skips_the_index(tmp_path):
    config = write_rag(tmp_path)

    class Guard:
        def list_collections(self, **_kwargs):
            raise AssertionError("不應刪向量")

        def get_collection(self, *_args, **_kwargs):
            raise AssertionError("不應刪向量")

        def close(self) -> None:
            return None

    def factory(_path, _settings):
        return Guard()

    with opened(config) as store:
        add_document(store)
    assert run_delete(config, doc_id=DOC, client_factory=factory, clock=lambda: LATER) == EXIT_OK
    with opened(config) as store:
        assert store.get_document(DOC) is None


def test_status_reports_probe_counts_and_extra_ids(tmp_path, monkeypatch, capsys):
    config = write_rag(tmp_path)
    monkeypatch.setattr("ragcli.reembed.OllamaClient", refuse_ollama)
    with opened(config) as store:
        add_document(store)
        add_document(store, DOC_B)
        orphan = add_chunk(store, DOC, 0, SECRET, embed_state="done")
        add_vector(store, orphan)
        image = make_image(status="caption_failed")
        store.insert_image(image)
    assert run_reembed(config, clock=lambda: LATER) == EXIT_OK
    with opened(config) as store:
        store.delete_chunk(orphan.chunk_id)
        add_chunk(store, DOC_B, 0, "缺少一", embed_state="done")
        add_chunk(store, DOC_B, 1, "缺少二", embed_state="done")
        add_chunk(store, DOC, 1, "還在等", embed_state="pending")
        add_chunk(store, DOC, 2, "描述中", embed_state="blocked", kind="figure")
        store.start_run(Run(run_id="ab" * 16, command="reembed", started_at=WHEN))
        store.finish_run("ab" * 16, finished_at=WHEN, exit_code=0, chunk_count=3)
    fake = FakeOllama()
    fake.loaded = [{"name": EMBED, "context_length": 2048}]
    code = run_status(config, client=fake.client(), free_bytes=lambda: 12345, clock=lambda: LATER)
    captured = capsys.readouterr()
    assert code == EXIT_OK
    assert "Ollama 0.40.11" in captured.out
    assert f"ps {EMBED} context_length=2048" in captured.out
    assert "向量索引多出來的 id" in captured.out
    assert orphan.chunk_id in captured.out
    assert "向量索引缺少的 id" in captured.out
    assert "集合筆數與已嵌入不一致。" in captured.out
    assert "失敗圖像 1" in captured.out
    assert "已嵌入 2" in captured.out
    assert "待完成 1" in captured.out
    assert "描述未完成 1" in captured.out
    assert "磁碟剩餘 12345" in captured.out
    assert "run reembed 結束碼 0 區塊 3" in captured.out
    assert SECRET not in captured.out
    assert SECRET not in captured.err
    with opened(config) as store:
        status_runs = [run for run in store.recent_runs() if run.command == "status"]
    assert len(status_runs) == 1
    run = status_runs[0]
    assert run.exit_code == 0
    assert run.model_name is None
    assert run.ask_latency_ms is None
    assert run.refused_count == 0


def test_status_ollama_down_is_exit_6_and_warns_on_wide_permissions(tmp_path, monkeypatch, capsys):
    config = write_rag(tmp_path)
    monkeypatch.setattr("ragcli.reembed.OllamaClient", refuse_ollama)
    with opened(config) as store:
        add_document(store)
        chunk = add_chunk(store, DOC, 0, "本地事實", embed_state="done")
        add_vector(store, chunk)
    assert run_reembed(config, clock=lambda: LATER) == EXIT_OK
    config.data_dir.chmod(0o755)
    fake = FakeOllama()
    fake.connect_error = True
    code = run_status(config, client=fake.client(), free_bytes=lambda: 50, clock=lambda: LATER)
    captured = capsys.readouterr()
    assert code == EXIT_OLLAMA_UNREACHABLE
    assert stat.S_IMODE(config.data_dir.stat().st_mode) == 0o755
    assert "data/ 對群組或其他人可讀。" in captured.err
    assert "連不上 Ollama。" in captured.out
    assert "已嵌入 1" in captured.out
    assert "作用中集合" in captured.out
    log = (config.data_dir / "logs" / "rag.log").read_text(encoding="utf-8")
    assert "連不上 Ollama。" in log
    with opened(config) as store:
        status_runs = [run for run in store.recent_runs() if run.command == "status"]
    assert len(status_runs) == 1
    run = status_runs[0]
    assert run.command == "status"
    assert run.exit_code == EXIT_OLLAMA_UNREACHABLE
    assert run.model_name is None
    assert run.ask_latency_ms is None


def test_cli_reembed_use_delete_and_status(tmp_path, monkeypatch):
    from test_cli_parse import invoke

    config = write_rag(tmp_path, batch_size=1)
    note = tmp_path / "note.txt"
    note.write_text("x", encoding="utf-8")
    with opened(config) as store:
        add_document(store, DOC, str(note.resolve()))
        add_document(store, DOC_B, f"/missing/{DOC_B}.pdf")
        first = add_chunk(store, DOC, 0, SECRET, embed_state="done")
        second = add_chunk(store, DOC_B, 0, "另一份文件", embed_state="done")
        add_vector(store, first)
        add_vector(store, second)
    monkeypatch.setattr("ragcli.reembed.OllamaClient", refuse_ollama)
    base = ["--config", str(config.config_path)]
    usage, _out, _err = invoke(
        monkeypatch, base + ["reembed", "--native-dimensions", "0"], cwd=tmp_path
    )
    assert usage == EXIT_USAGE
    short = collection_name(EMBED, 2)
    code, out, err = invoke(
        monkeypatch,
        base + ["reembed", "--dimensions", "2", "--no-activate"],
        cwd=tmp_path,
    )
    assert code == EXIT_OK
    assert "作用中指標未改" in out
    assert short in out
    assert SECRET not in out
    assert SECRET not in err
    assert active_of(config) is None
    assert not (config.data_dir / "locks" / "ollama.lock").exists()
    code, out, _err = invoke(monkeypatch, base + ["reembed"], cwd=tmp_path)
    assert code == EXIT_OK
    assert "已重建" in out
    wide = collection_name(EMBED, 4)
    assert active_of(config) is not None and active_of(config).collection == wide
    fake = SwitchFake(width=8)
    monkeypatch.setattr("ragcli.reembed.OllamaClient", lambda base_url: fake.client())
    code, out, err = invoke(
        monkeypatch,
        base
        + [
            "reembed",
            "--model",
            TINY,
            "--native-dimensions",
            "8",
            "--dimensions",
            "8",
            "--drop-old",
        ],
        cwd=tmp_path,
    )
    assert code == EXIT_OK
    assert PREFIX in err
    assert f"已刪除集合 {wide}" in out
    assert "ollama pull" not in err
    assert SECRET not in err
    assert [body["input"] for body in document_bodies(fake)] == [[SECRET], ["另一份文件"]]
    assert fake.loaded == []
    tiny = collection_name(TINY, 8)
    assert active_of(config) is not None and active_of(config).collection == tiny
    assert wide not in collection_names(config)
    monkeypatch.setattr("ragcli.reembed.OllamaClient", refuse_ollama)
    code, out, err = invoke(
        monkeypatch, base + ["use", "--collection", short], cwd=tmp_path
    )
    assert code == EXIT_OK
    assert f"作用中集合 {short}" in out
    pointer = active_of(config)
    assert pointer is not None and pointer.collection == short and pointer.dimensions == 2
    code, out, err = invoke(
        monkeypatch, base + ["delete", "--path", str(tmp_path / "nested" / ".." / note.name)], cwd=tmp_path
    )
    assert code == EXIT_OK
    assert out.strip() == f"已刪除 {DOC}"
    assert "note.txt" not in out
    code, out, _err = invoke(monkeypatch, base + ["delete", "--doc-id", DOC_B], cwd=tmp_path)
    assert code == EXIT_OK
    assert out.strip() == f"已刪除 {DOC_B}"
    monkeypatch.setattr("ragcli.reembed.OllamaClient", lambda base_url: FakeOllama().client())
    code, out, err = invoke(monkeypatch, base + ["status"], cwd=tmp_path)
    assert code == EXIT_OK
    assert "Ollama 0.40.11" in out
    assert "ps 空" in out
    assert SECRET not in out
    assert SECRET not in err
    with opened(config) as store:
        commands = [run.command for run in store.recent_runs()]
    assert "use" not in commands
    assert "reembed" in commands
    assert "delete" in commands
    assert "status" in commands
