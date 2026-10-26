"""Live ask against loopback Ollama.

Skipped unless RAG_INTEGRATION=1. A missing server or a missing model skips
too. This file must not pull weights.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

from ragcli.ask import query_text, run_ask
from ragcli.config import load_config
from ragcli.errors import RagError
from ragcli.ollama import OllamaClient, available_bytes, choose_generator
from ragcli.store.chroma import ChromaIndex, IndexRecord
from ragcli.store.sqlite import ActiveIndex, SqliteStore, chunk_id_for
from test_sqlite_store import make_chunk, make_document

pytestmark = pytest.mark.integration

SENTENCE = "退款須於購買後七日內以書面提出。"
QUESTION = "退款須於購買後七日內以書面提出。"


class _RecordingTransport(httpx.BaseTransport):
    """Record JSON bodies and keep the pool open across Ollama's per-call clients."""

    def __init__(self) -> None:
        self._inner = httpx.HTTPTransport(trust_env=False)
        self.calls: list[tuple[str, str, object]] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode()) if request.content else None
        self.calls.append((request.method, request.url.path, body))
        return self._inner.handle_request(request)

    def close(self) -> None:
        return None

    def shutdown(self) -> None:
        self._inner.close()


def _write_config(directory: Path) -> None:
    (directory / "rag.toml").write_text(
        "\n".join(
            [
                "[paths]",
                'data_dir = "data"',
                "[limits]",
                "min_free_bytes = 1",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_live_ask_retrieves_the_sentence_and_sends_think_false(tmp_path: Path):
    if os.environ.get("RAG_INTEGRATION") != "1":
        pytest.skip("RAG_INTEGRATION 不是 1")
    _write_config(tmp_path)
    config = load_config(tmp_path / "rag.toml")
    probe = OllamaClient(config.ollama_base_url)
    try:
        probe.version()
        tags = probe.tags()
    except RagError as exc:
        pytest.skip(f"Ollama 連不上：{exc.message}")
    if config.embed_model not in tags:
        pytest.skip(f"缺少嵌入模型 {config.embed_model}")
    try:
        chosen = choose_generator(
            preferred=config.generate_model,
            fallback=config.generate_fallback_model,
            tags=tags,
            available=available_bytes(),
            nine_b_min_bytes=config.nine_b_min_bytes,
            four_b_min_bytes=config.four_b_min_bytes,
            allow_fallback=True,
        )
    except RagError as exc:
        pytest.skip(f"生成模型閘門不可用：{exc.message}")
    if chosen not in tags:
        pytest.skip(f"缺少生成模型 {chosen}")

    transport = _RecordingTransport()
    client = OllamaClient(config.ollama_base_url, transport=transport)
    try:
        _seed(config, probe, SENTENCE)
        code = run_ask(
            config,
            QUESTION,
            client=client,
            as_json=True,
        )
        assert code == 0
        embeds = [
            body
            for _method, path, body in transport.calls
            if path == "/api/embed" and isinstance(body, dict) and body.get("input") != "."
        ]
        chats = [body for _method, path, body in transport.calls if path == "/api/chat"]
        pulls = [path for _method, path, _body in transport.calls if path == "/api/pull"]
        assert pulls == []
        assert embeds
        assert embeds[0]["input"] == [query_text(QUESTION)]
        assert "dimensions" not in embeds[0]
        assert chats, "檢索沒有進入生成"
        chat = chats[0]
        assert chat["think"] is False
        assert chat["options"]["num_ctx"] == 8192
        assert SENTENCE in chat["messages"][1]["content"]
        store = SqliteStore.open(config.data_dir)
        try:
            chunk = store.get_chunk(chunk_id_for(_doc_id(), 0))
        finally:
            store.close()
        assert chunk is not None
        assert chunk.text == SENTENCE
        assert not chunk.text.startswith("Instruct:")
    finally:
        transport.shutdown()
        try:
            OllamaClient(config.ollama_base_url).unload_all()
        except RagError:
            pass


def _doc_id() -> str:
    import hashlib

    return hashlib.sha256(SENTENCE.encode("utf-8")).hexdigest()


def _seed(config, client: OllamaClient, sentence: str) -> None:
    vectors = client.embed(
        config.embed_model,
        [sentence],
        num_ctx=config.embed_num_ctx,
        keep_alive="0",
    )
    try:
        client.release_stage(config.embed_model)
    except RagError:
        pass
    vector = vectors[0]
    assert len(vector) == config.embed_native_dimensions
    doc_id = _doc_id()
    store = SqliteStore.open(config.data_dir)
    try:
        store.insert_document(
            make_document(
                doc_id=doc_id,
                source_path=f"/tmp/{doc_id}.txt",
                filename="note.txt",
                media_type="text",
                parser="text",
            )
        )
        chunk = make_chunk(doc_id=doc_id, ordinal=0, text=sentence, embed_state="done")
        store.insert_chunk(chunk)
        index = ChromaIndex.open(config.data_dir)
        try:
            name = index.upsert(
                [
                    IndexRecord(
                        chunk_id=chunk.chunk_id,
                        doc_id=doc_id,
                        text=sentence,
                        embedding=tuple(vector),
                        source_path=f"/tmp/{doc_id}.txt",
                        filename="note.txt",
                        kind="text",
                        ordinal=0,
                        page_start=-1,
                        page_end=-1,
                    )
                ],
                model=config.embed_model,
                dimensions=config.embed_dimensions,
                embed_num_ctx=config.embed_num_ctx,
            )
        finally:
            index.close()
        store.set_active(
            ActiveIndex(
                collection=name,
                embed_model=config.embed_model,
                dimensions=config.embed_dimensions,
                embed_num_ctx=config.embed_num_ctx,
            )
        )
    finally:
        store.close()
