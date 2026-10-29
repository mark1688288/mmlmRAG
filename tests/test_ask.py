"""rag ask: query prefix, distance gate, context packing, and citations.

These tests never open a socket and never pull a model. The Ollama client is
a scripted loopback. Chroma is the in-memory fake, so MiniLM is not loaded.
"""

from __future__ import annotations

import json
import logging
import stat
from pathlib import Path

import pytest

from ragcli.ask import (
    document_embed_text,
    pack_excerpts,
    query_text,
    require_question_limit,
    retrieval_budget,
    run_ask,
)
from ragcli.chunking import estimate_tokens
from ragcli.config import Config, load_config
from ragcli.errors import (
    EXIT_DISK,
    EXIT_MISMATCH,
    EXIT_MODEL_MISSING,
    EXIT_NO_COLLECTION,
    EXIT_OK,
    EXIT_OLLAMA_UNREACHABLE,
    EXIT_QUESTION,
    EXIT_RESOURCES,
    EXIT_UNCLASSIFIED,
    EXIT_USAGE,
    RagError,
)
from ragcli.i18n import set_language
from ragcli.ollama import FOUR_B_MODEL, NINE_B_MODEL
from ragcli.prompt import (
    Excerpt,
    format_excerpt,
    format_user_message,
    is_refusal,
    refusal_sentence,
    system_prompt,
)
from ragcli.store.chroma import ChromaIndex, IndexRecord
from ragcli.store.sqlite import ActiveIndex, SqliteStore, chunk_id_for
from test_chroma_factory import FakeClient
from test_cli_parse import invoke
from test_ollama_client import EMBED, FakeOllama
from test_sqlite_store import DOC, make_chunk, make_document

SECRET_CHUNK = "ZXQ-SECRET-CHUNK"
SECRET_QUESTION = "ZXQ-SECRET-QUESTION"
WHEN = "2026-10-26T00:00:00+00:00"
VECTOR = (3.0, 0.0, 0.0, 4.0)
PLENTY = 20_000_000_000
SEVEN_GIB = 7 * 1024**3
_PREFIX_WARNING = (
    "The query prefix was written for Qwen3-Embedding, "
    "and this model may not apply."
)


def write_rag(
    directory: Path,
    *,
    native: int = 4,
    min_free: int = 1,
    num_ctx: int = 8192,
    num_predict: int = 768,
    max_distance: float = 0.85,
    max_question: int = 512,
    k: int = 5,
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
                "dimensions = 4",
                "num_ctx = 2048",
                "[generate]",
                f"num_ctx = {num_ctx}",
                f"num_predict = {num_predict}",
                "[retrieve]",
                f"k = {k}",
                f"max_cosine_distance = {max_distance}",
                f"max_question_tokens = {max_question}",
                "[limits]",
                f"min_free_bytes = {min_free}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return load_config(path)


def expect(func, code: int, message: str) -> RagError:
    with pytest.raises(RagError) as caught:
        func()
    assert caught.value.code == code
    assert message in caught.value.message
    return caught.value


class Corpus:
    """One temp rag.toml, one fake Ollama, and one fake Chroma client."""

    def __init__(self, tmp_path: Path, **kwargs) -> None:
        self.config = write_rag(tmp_path, **kwargs)
        self.fake = FakeOllama()
        self.fake.vectors = [list(VECTOR)]
        self.fake.chat_message = {"role": "assistant", "content": "七日內可以退款。", "thinking": ""}
        self._chroma: FakeClient | None = None

        def factory(path: Path, settings) -> FakeClient:
            if self._chroma is None:
                self._chroma = FakeClient(path, settings)
            return self._chroma

        self.factory = factory

    def seed(
        self,
        rows: list[dict],
        *,
        model: str = EMBED,
        dimensions: int = 4,
        num_ctx: int = 2048,
        embedding: tuple[float, ...] | None = None,
    ) -> str:
        vector = VECTOR if embedding is None else embedding
        store = SqliteStore.open(self.config.data_dir)
        try:
            if rows and store.get_document(DOC) is None:
                store.insert_document(make_document())
            records: list[IndexRecord] = []
            for ordinal, row in enumerate(rows):
                page = int(row.get("page", -1))
                chunk = make_chunk(
                    ordinal=ordinal,
                    text=row["text"],
                    embed_state=row.get("embed_state", "done"),
                    kind=row.get("kind", "text"),
                    page_start=None if page < 1 else page,
                    page_end=None if page < 1 else page,
                )
                store.insert_chunk(chunk)
                records.append(
                    IndexRecord(
                        chunk_id=chunk.chunk_id,
                        doc_id=DOC,
                        text=row["text"],
                        embedding=vector,
                        source_path=f"/docs/{DOC}.pdf",
                        filename=row.get("filename", "退款規定.pdf"),
                        kind=row.get("kind", "text"),
                        ordinal=ordinal,
                        page_start=-1 if page < 1 else page,
                        page_end=-1 if page < 1 else page,
                    )
                )
            index = ChromaIndex.open(self.config.data_dir, client_factory=self.factory)
            try:
                name = index.upsert(
                    records,
                    model=model,
                    dimensions=dimensions,
                    embed_num_ctx=num_ctx,
                )
            finally:
                index.close()
            store.set_active(
                ActiveIndex(
                    collection=name,
                    embed_model=model,
                    dimensions=dimensions,
                    embed_num_ctx=num_ctx,
                )
            )
        finally:
            store.close()
        assert self._chroma is not None
        collection = self._chroma.collections[name]
        for row, record in zip(rows, records, strict=True):
            if "distance" in row:
                collection.distance_for[record.chunk_id] = row["distance"]
        return name

    def ask(self, question: str, **kwargs) -> int:
        kwargs.setdefault("client", self.fake.client())
        kwargs.setdefault("available", PLENTY)
        kwargs.setdefault("client_factory", self.factory)
        return run_ask(self.config, question, **kwargs)

    def collection(self, name: str):
        assert self._chroma is not None
        return self._chroma.collections[name]


def latest_run(config: Config):
    store = SqliteStore.open(config.data_dir)
    try:
        runs = store.recent_runs()
    finally:
        store.close()
    assert len(runs) == 1
    return runs[0]


def log_text(config: Config) -> str:
    return (config.data_dir / "logs" / "rag.log").read_text(encoding="utf-8")


def query_embeds(fake: FakeOllama) -> list:
    return [body for body in fake.bodies("/api/embed") if body.get("input") != "."]


def chats(fake: FakeOllama) -> list:
    return fake.bodies("/api/chat")


@pytest.fixture(autouse=True)
def reset_language():
    set_language("en")
    yield
    set_language("en")


@pytest.fixture(autouse=True)
def no_live_ollama(monkeypatch):
    monkeypatch.setattr(
        "ragcli.ask.available_bytes",
        lambda: (_ for _ in ()).throw(AssertionError("不應讀 vm_stat")),
    )

    def refuse_client(*_args, **_kwargs):
        raise AssertionError("不應建立 OllamaClient")

    monkeypatch.setattr("ragcli.ask.OllamaClient", refuse_client)


def test_query_prefix_matches_the_qwen_card():
    question = "退款規定？"
    text = query_text(question)
    assert text == (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        f"Query:{question}"
    )
    assert "Query: " not in text


def test_document_text_is_not_prefixed():
    text = "Query:這段已經在文件裡"
    assert document_embed_text(text) == text
    assert not document_embed_text(text).startswith("Instruct:")


def test_question_limit_rejects_without_truncating():
    require_question_limit("文" * 511, 512)
    assert estimate_tokens("文" * 511) == 512
    caught = expect(lambda: require_question_limit("文" * 512, 512), EXIT_QUESTION, "not shortened")
    assert "文文文" not in caught.message


def test_budget_reserves_the_answer_and_the_margin():
    assert retrieval_budget(
        num_ctx=8192, system="系統", question="問題", num_predict=768
    ) == (8192 - estimate_tokens("系統") - estimate_tokens("問題") - 768 - 128)


def test_pack_stops_when_the_next_chunk_does_not_fit():
    near = Excerpt("near", "a.pdf", 1, "text", "近", 0.1)
    far = Excerpt("far", "b.pdf", 2, "text", "遠" * 40, 0.2)
    later = Excerpt("x", "c.pdf", 3, "text", "短", 0.3)
    first = estimate_tokens(format_excerpt(1, near))
    packed = pack_excerpts([later, far, near], first)
    assert [item.chunk_id for item in packed] == ["near"]
    assert packed[0].text == "近"
    remainder = first - estimate_tokens(format_excerpt(1, near))
    assert estimate_tokens(format_excerpt(2, later)) <= first
    assert remainder == 0


def test_pack_rejects_when_the_first_chunk_does_not_fit():
    excerpt = Excerpt("only", "a.pdf", -1, "text", "整段都要留下", 0.1)
    caught = expect(lambda: pack_excerpts([excerpt], 1), EXIT_QUESTION, "partial prompt")
    assert "整段都要留下" not in caught.message
    assert excerpt.text not in caught.message


def test_user_message_omits_a_missing_page_and_keeps_the_system_template():
    missing = Excerpt("c1", "a.txt", -1, "ocr", "正文", 0.2)
    paged = Excerpt("c2", "b.txt", 4, "figure", "圖", 0.3)
    hidden = format_user_message("問題本身", [missing])
    shown = format_user_message("問題本身", [paged])
    assert hidden == (
        "Question:\n"
        "問題本身\n"
        "\n"
        "Excerpts:\n"
        "[1] source=a.txt chunk=c1 kind=ocr\n"
        "正文"
    )
    assert shown == (
        "Question:\n"
        "問題本身\n"
        "\n"
        "Excerpts:\n"
        "[1] source=b.txt page=4 chunk=c2 kind=figure\n"
        "圖"
    )
    for message in (hidden, shown):
        assert "頁=" not in message
        assert "來源=" not in message
        assert "種類=" not in message
        assert "問題：" not in message
        assert "摘錄：" not in message
    assert "page=" not in hidden
    set_language("zh")
    assert format_user_message("問題本身", [missing]) == (
        "問題：\n"
        "問題本身\n"
        "\n"
        "摘錄：\n"
        "[1] 來源=a.txt chunk=c1 種類=ocr\n"
        "正文"
    )
    assert format_user_message("問題本身", [paged]) == (
        "問題：\n"
        "問題本身\n"
        "\n"
        "摘錄：\n"
        "[1] 來源=b.txt 頁=4 chunk=c2 種類=figure\n"
        "圖"
    )
    for language in ("en", "zh"):
        set_language(language)
        prompt = system_prompt()
        assert "[來源: {filename} p.{page} #{chunk_id}]" in prompt
        assert "{filename}" in prompt
        assert refusal_sentence() in prompt
        assert "只可以根據下方「摘錄」回答問題" in prompt
        assert "Excerpts" not in prompt
        assert "Question:" not in prompt


def test_refusal_sentence_follows_the_catalog_and_the_prompt():
    english = "The documents do not contain the answer."
    chinese = "資料中找不到答案。"
    set_language("en")
    prompt = system_prompt()
    assert prompt.count(english) == 1
    assert "{filename}" in prompt
    assert "不得譯成問題的語言" in prompt
    assert "{refusal}" not in prompt
    assert is_refusal(f"  {english}\n")
    assert not is_refusal(english + " extra")
    assert not is_refusal(chinese)

    set_language("zh")
    prompt = system_prompt()
    assert refusal_sentence() == chinese
    assert prompt.count(chinese) == 1
    assert "{filename}" in prompt
    assert "不得譯成問題的語言" in prompt
    assert "{refusal}" not in prompt
    assert is_refusal(f"  {chinese}\n")
    assert not is_refusal(english)


def test_low_disk_does_not_open_the_corpus(tmp_path):
    corpus = Corpus(tmp_path, min_free=1)
    expect(lambda: corpus.ask("問題", free_bytes=lambda: 0), EXIT_DISK, "Free disk space")
    assert not corpus.config.data_dir.exists()


def test_blank_question_and_k_are_usage_errors(tmp_path):
    corpus = Corpus(tmp_path)
    expect(lambda: corpus.ask("  "), EXIT_USAGE, "Missing question")
    expect(lambda: corpus.ask("問題", k=9), EXIT_USAGE, "--k")
    assert not corpus.config.data_dir.exists()
    assert corpus.fake.seen == []


@pytest.mark.parametrize("state", ["pending", "blocked"])
def test_unfinished_chunks_exit_12_without_calling_ollama(tmp_path, state):
    corpus = Corpus(tmp_path)
    corpus.seed([{"text": SECRET_CHUNK, "embed_state": state, "distance": 0.1}])
    caught = expect(
        lambda: corpus.ask(SECRET_QUESTION),
        EXIT_MISMATCH,
        "rag ingest --resume",
    )
    assert SECRET_CHUNK not in caught.message
    assert SECRET_QUESTION not in caught.message
    assert corpus.fake.seen == []
    assert not (corpus.config.data_dir / "locks" / "ollama.lock").exists()
    run = latest_run(corpus.config)
    assert run.command == "ask"
    assert run.exit_code == EXIT_MISMATCH
    assert run.refused_count == 0
    assert run.ask_latency_ms is not None
    assert SECRET_CHUNK not in log_text(corpus.config)


def test_missing_active_collection_follows_zh(tmp_path):
    set_language("zh")
    corpus = Corpus(tmp_path)
    expect(lambda: corpus.ask("有集合嗎"), EXIT_NO_COLLECTION, "尚未有作用中集合。")


def test_missing_active_collection_is_exit_11(tmp_path):
    corpus = Corpus(tmp_path)
    expect(lambda: corpus.ask("有集合嗎"), EXIT_NO_COLLECTION, "There is no active collection.")
    assert corpus.fake.seen == []
    run = latest_run(corpus.config)
    assert run.exit_code == EXIT_NO_COLLECTION
    assert run.ask_latency_ms is not None
    assert stat.S_IMODE((corpus.config.data_dir / "logs" / "rag.log").stat().st_mode) == 0o600


def test_long_question_is_exit_9_and_is_not_sent(tmp_path):
    corpus = Corpus(tmp_path)
    corpus.seed([{"text": "短區塊", "distance": 0.1}])
    question = "問" * 600
    caught = expect(lambda: corpus.ask(question), EXIT_QUESTION, "not shortened")
    assert question not in caught.message
    assert corpus.fake.seen == []
    run = latest_run(corpus.config)
    assert run.exit_code == EXIT_QUESTION
    assert run.refused_count == 0
    assert question not in log_text(corpus.config)


def test_distance_gate_refuses_without_a_generator(tmp_path, capsys):
    corpus = Corpus(tmp_path)
    corpus.fake.loaded = [{"name": NINE_B_MODEL, "context_length": 8192}]
    name = corpus.seed(
        [{"text": SECRET_CHUNK, "distance": 0.9, "page": -1}],
        num_ctx=1024,
    )
    code = corpus.ask(SECRET_QUESTION, show_chunks=True)
    out = capsys.readouterr()
    assert code == EXIT_OK
    assert out.out == "The documents do not contain the answer.\n"
    assert SECRET_CHUNK not in out.err
    assert SECRET_QUESTION not in out.err
    assert "頁=" not in out.err
    for token in ("source=", "來源=", "Question:", "Excerpts:", "問題：", "摘錄：", "種類="):
        assert token not in out.err
    assert "chunk_id=" in out.err
    assert "distance=0.900000" in out.err
    assert chats(corpus.fake) == []
    assert query_embeds(corpus.fake)[0]["options"]["num_ctx"] == 1024
    assert corpus.fake.loaded == []
    assert "/api/pull" not in corpus.fake.paths()
    assert corpus.collection(name).queries == [] or corpus.collection(name).queries
    run = latest_run(corpus.config)
    assert run.refused_count == 1
    assert run.model_name is None
    assert run.chunk_count == 0
    assert run.exit_code == EXIT_OK
    assert SECRET_CHUNK not in log_text(corpus.config)


def test_exact_distance_limit_still_calls_the_generator(tmp_path):
    corpus = Corpus(tmp_path)
    corpus.seed([{"text": "剛好在閘門上", "distance": 0.85}])
    assert corpus.ask("閘門") == EXIT_OK
    assert len(chats(corpus.fake)) == 1


def test_empty_collection_refuses_without_chat(tmp_path, capsys):
    corpus = Corpus(tmp_path)
    corpus.seed([])
    assert corpus.ask("空集合") == EXIT_OK
    assert capsys.readouterr().out == "The documents do not contain the answer.\n"
    assert query_embeds(corpus.fake)
    assert chats(corpus.fake) == []


def test_empty_collection_refusal_follows_zh(tmp_path, capsys):
    set_language("zh")
    corpus = Corpus(tmp_path)
    corpus.seed([])
    assert corpus.ask("空集合") == EXIT_OK
    assert capsys.readouterr().out == "資料中找不到答案。\n"


def test_first_chunk_that_does_not_fit_is_not_sent(tmp_path):
    corpus = Corpus(tmp_path, num_ctx=64, num_predict=1)
    corpus.seed([{"text": "BLOCK-SHOULD-NOT-BE-SENT", "distance": 0.1}])
    question = "完整的問題"
    caught = expect(lambda: corpus.ask(question), EXIT_QUESTION, "partial prompt")
    assert "BLOCK-SHOULD-NOT-BE-SENT" not in caught.message
    bodies = json.dumps(corpus.fake.seen, ensure_ascii=False)
    assert "BLOCK-SHOULD-NOT-BE-SENT" not in bodies
    assert "BLOCK-SHOULD" not in bodies
    assert chats(corpus.fake) == []
    assert query_embeds(corpus.fake)[0]["input"] == [query_text(question)]
    assert corpus.fake.loaded == []


def test_later_chunk_is_dropped_whole(tmp_path, capsys):
    question = "短問"
    near = Excerpt(chunk_id_for(DOC, 0), "退款規定.pdf", 3, "text", "NEAR-ALPHA", 0.1)
    far = Excerpt(chunk_id_for(DOC, 1), "退款規定.pdf", 4, "caption", "FAR-BETA", 0.4)
    budget = estimate_tokens(format_excerpt(1, near))
    num_ctx = (
        budget
        + estimate_tokens(system_prompt())
        + estimate_tokens(question)
        + 1
        + 128
    )
    corpus = Corpus(tmp_path, num_ctx=num_ctx, num_predict=1)
    corpus.seed(
        [
            {"text": far.text, "distance": far.distance, "page": 4, "kind": "caption"},
            {"text": near.text, "distance": near.distance, "page": 3, "kind": "text"},
        ]
    )
    assert corpus.ask(question, k=8) == EXIT_OK
    out = capsys.readouterr()
    user = chats(corpus.fake)[0]["messages"][1]["content"]
    assert near.text in user
    assert far.text not in user
    assert "page=3" in user
    assert "頁=" not in user
    assert "Excerpts:" in user
    assert user.index(near.text) < user.find("FAR") if "FAR" in user else True
    assert out.out == "七日內可以退款。\n"
    query = corpus._chroma.collections[next(iter(corpus._chroma.collections))].queries[0]
    assert query["n_results"] == 8
    assert "query_texts" not in query
    run = latest_run(corpus.config)
    assert run.chunk_count == 1
    assert run.model_name == NINE_B_MODEL


def test_answer_uses_raw_vectors_and_the_chat_contract(tmp_path, capsys):
    corpus = Corpus(tmp_path)
    name = corpus.seed([{"text": SECRET_CHUNK, "distance": 0.25, "page": -1}])
    logger = logging.getLogger("ragcli")
    previous = logger.level
    logger.setLevel(logging.INFO)
    try:
        code = corpus.ask(SECRET_QUESTION, show_chunks=True, as_json=True, clock=lambda: WHEN)
    finally:
        logger.setLevel(previous)
    captured = capsys.readouterr()
    assert code == EXIT_OK
    payload = json.loads(captured.out)
    assert set(payload) == {"answer", "refused", "model", "sources"}
    assert payload["answer"] == "七日內可以退款。"
    assert payload["refused"] is False
    assert payload["model"] == NINE_B_MODEL
    assert payload["sources"] == [
        {
            "chunk_id": chunk_id_for(DOC, 0),
            "filename": "退款規定.pdf",
            "page": None,
            "distance": 0.25,
            "kind": "text",
        }
    ]
    assert SECRET_CHUNK not in captured.out
    assert SECRET_CHUNK not in captured.err
    assert SECRET_QUESTION not in captured.out
    assert SECRET_QUESTION not in captured.err
    assert "頁=" not in captured.err
    for token in ("source=", "來源=", "Question:", "Excerpts:", "問題：", "摘錄：", "種類="):
        assert token not in captured.err
    assert _PREFIX_WARNING not in captured.err
    embed = query_embeds(corpus.fake)[0]
    assert embed["input"] == [query_text(SECRET_QUESTION)]
    assert embed["truncate"] is False
    assert embed["keep_alive"] == "0"
    assert embed["options"] == {"num_ctx": 2048}
    assert "dimensions" not in embed
    chat = chats(corpus.fake)[0]
    assert chat["think"] is False
    assert chat["stream"] is False
    assert chat["keep_alive"] == "0"
    assert chat["options"]["num_ctx"] == 8192
    assert chat["options"]["num_predict"] == 768
    assert chat["options"]["temperature"] == 0.2
    assert "think" not in chat["options"]
    assert chat["messages"][0] == {"role": "system", "content": system_prompt()}
    assert "images" not in chat["messages"][0]
    assert "images" not in chat["messages"][1]
    content = chat["messages"][1]["content"]
    assert SECRET_CHUNK in content
    assert "Question:\n" in content
    assert "\n\nExcerpts:\n" in content
    assert "source=" in content
    assert "chunk=" in content
    assert "kind=" in content
    assert "page=" not in content
    assert "頁=" not in content
    assert "來源=" not in content
    assert "問題：" not in content
    assert "摘錄：" not in content
    sent = corpus.collection(name).queries[0]["query_embeddings"]
    assert sent == [[3.0, 0.0, 0.0, 4.0]]
    assert ":" not in name
    assert corpus.fake.loaded == []
    assert "/api/pull" not in corpus.fake.paths()
    run = latest_run(corpus.config)
    assert run.command == "ask"
    assert run.exit_code == EXIT_OK
    assert run.refused_count == 0
    assert run.chunk_count == 1
    assert run.model_name == NINE_B_MODEL
    assert run.ask_latency_ms is not None
    assert run.ask_latency_ms >= 0
    assert run.started_at == WHEN
    assert run.ocr_count == 0
    assert run.vision_count == 0
    logged = log_text(corpus.config)
    assert SECRET_CHUNK not in logged
    assert SECRET_QUESTION not in logged
    assert "Exit code 0" in logged
    assert stat.S_IMODE((corpus.config.data_dir / "logs" / "rag.log").stat().st_mode) == 0o600


def test_chat_user_message_headers_follow_zh(tmp_path):
    set_language("zh")
    corpus = Corpus(tmp_path)
    corpus.seed([{"text": SECRET_CHUNK, "distance": 0.25, "page": 4}])
    assert corpus.ask(SECRET_QUESTION) == EXIT_OK
    content = chats(corpus.fake)[0]["messages"][1]["content"]
    assert "來源=" in content
    assert " 頁=" in content
    assert " 種類=" in content
    assert content.startswith("問題：\n")
    assert "\n\n摘錄：\n" in content
    assert "chunk=" in content
    assert "Question:" not in content
    assert "Excerpts:" not in content
    assert "source=" not in content
    assert " page=" not in content


def test_shorter_width_uses_one_truncate_normalize(tmp_path):
    corpus = Corpus(tmp_path)
    name = corpus.seed(
        [{"text": "縮維", "distance": 0.1}],
        dimensions=2,
    )
    assert corpus.ask("縮") == EXIT_OK
    sent = corpus.collection(name).queries[0]["query_embeddings"]
    assert sent == [[1.0, 0.0]]


def test_wrong_native_width_is_exit_12(tmp_path):
    corpus = Corpus(tmp_path)
    corpus.fake.vectors = [[0.25, 0.5]]
    name = corpus.seed([{"text": "寬度不對", "distance": 0.1}])
    expect(lambda: corpus.ask("寬度"), EXIT_MISMATCH, "native width")
    assert corpus.collection(name).queries == []
    assert chats(corpus.fake) == []


def test_wider_than_native_is_exit_12(tmp_path):
    corpus = Corpus(tmp_path)
    name = corpus.seed(
        [{"text": "不能加長", "distance": 0.1}],
        dimensions=8,
        embedding=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    expect(lambda: corpus.ask("加長"), EXIT_MISMATCH, "cannot be lengthened")
    assert corpus.collection(name).queries == []
    assert chats(corpus.fake) == []
    assert corpus.fake.loaded == []


def test_zero_head_cannot_be_truncated(tmp_path):
    corpus = Corpus(tmp_path)
    corpus.fake.vectors = [[0.0, 0.0, 0.0, 1.0]]
    corpus.seed([{"text": "零", "distance": 0.1}], dimensions=2)
    expect(lambda: corpus.ask("零"), EXIT_MISMATCH, "zero vector")
    assert chats(corpus.fake) == []


def test_non_qwen_model_keeps_the_prefix_and_warns(tmp_path, capsys):
    corpus = Corpus(tmp_path)
    corpus.seed(
        [{"text": "其他模型", "distance": 0.1}],
        model="lab-embed:tiny",
    )
    assert corpus.ask("其他") == EXIT_OK
    err = capsys.readouterr().err
    assert _PREFIX_WARNING in err
    assert "其他" not in err
    assert query_embeds(corpus.fake)[0]["input"] == [query_text("其他")]


def test_non_qwen_prefix_warning_follows_zh(tmp_path, capsys):
    set_language("zh")
    corpus = Corpus(tmp_path)
    corpus.seed([{"text": "其他模型", "distance": 0.1}], model="lab-embed:tiny")
    assert corpus.ask("其他") == EXIT_OK
    assert "查詢前綴是為 Qwen3-Embedding 寫的，這個模型可能不適用。" in capsys.readouterr().err


def test_metadata_mismatch_is_exit_12_without_ollama(tmp_path):
    corpus = Corpus(tmp_path)
    name = corpus.seed([{"text": "指標", "distance": 0.1}])
    store = SqliteStore.open(corpus.config.data_dir)
    try:
        store.set_active(
            ActiveIndex(
                collection=name,
                embed_model="qwen3-embedding:4b",
                dimensions=4,
                embed_num_ctx=2048,
            )
        )
    finally:
        store.close()
    expect(lambda: corpus.ask("指標"), EXIT_MISMATCH, "metadata")
    assert corpus.fake.seen == []


def test_model_refusal_is_success_and_json_marks_it(tmp_path, capsys):
    corpus = Corpus(tmp_path)
    corpus.fake.chat_message = {
        "role": "assistant",
        "content": f"\n{refusal_sentence()}\n",
        "thinking": "",
    }
    corpus.seed([{"text": "證據不足", "distance": 0.2, "page": 2}])
    assert corpus.ask("證據", as_json=True) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["refused"] is True
    assert payload["answer"] == refusal_sentence()
    assert payload["model"] == NINE_B_MODEL
    assert payload["sources"][0]["page"] == 2
    assert payload["sources"][0]["chunk_id"] == chunk_id_for(DOC, 0)
    run = latest_run(corpus.config)
    assert run.refused_count == 1
    assert run.exit_code == EXIT_OK
    assert run.chunk_count == 1


def test_thinking_only_is_exit_1_and_is_not_the_answer(tmp_path, capsys):
    corpus = Corpus(tmp_path)
    corpus.fake.chat_message = {"role": "assistant", "content": "", "thinking": "THOUGHT-SECRET"}
    corpus.seed([{"text": "正文", "distance": 0.1}])
    caught = expect(lambda: corpus.ask("思考"), EXIT_UNCLASSIFIED, "Try again")
    assert "THOUGHT-SECRET" not in caught.message
    assert capsys.readouterr().out == ""
    assert corpus.fake.loaded == []
    run = latest_run(corpus.config)
    assert run.exit_code == EXIT_UNCLASSIFIED
    assert run.model_name == NINE_B_MODEL
    assert run.refused_count == 0


def test_low_memory_falls_back_to_4b(tmp_path):
    corpus = Corpus(tmp_path)
    corpus.seed([{"text": "記憶體", "distance": 0.1}])
    assert corpus.ask("記憶體", available=SEVEN_GIB) == EXIT_OK
    assert chats(corpus.fake)[0]["model"] == FOUR_B_MODEL
    assert "switching" in log_text(corpus.config)


def test_no_fallback_low_memory_is_exit_13(tmp_path):
    corpus = Corpus(tmp_path)
    corpus.seed([{"text": "不後備", "distance": 0.1}])
    caught = expect(
        lambda: corpus.ask("不後備", available=SEVEN_GIB, allow_fallback=False),
        EXIT_RESOURCES,
        "Did not switch to the fallback model.",
    )
    assert "Available memory" in caught.message
    assert chats(corpus.fake) == []


def test_missing_generator_is_exit_7_and_names_models_pull(tmp_path):
    corpus = Corpus(tmp_path)
    corpus.fake.tag_names = [EMBED]
    corpus.seed([{"text": "沒模型", "distance": 0.1}])
    caught = expect(lambda: corpus.ask("沒模型"), EXIT_MODEL_MISSING, "rag models pull")
    assert "ollama pull" not in caught.message
    assert chats(corpus.fake) == []


def test_ollama_down_is_exit_6(tmp_path):
    corpus = Corpus(tmp_path)
    corpus.fake.connect_error = True
    corpus.seed([{"text": "離線", "distance": 0.1}])
    expect(lambda: corpus.ask("離線"), EXIT_OLLAMA_UNREACHABLE, "Cannot reach Ollama.")
    run = latest_run(corpus.config)
    assert run.exit_code == EXIT_OLLAMA_UNREACHABLE
    assert run.ollama_error_count >= 1
    assert run.ask_latency_ms is not None


def test_cli_json_show_chunks_and_model_flag(monkeypatch, tmp_path):
    corpus = Corpus(tmp_path)
    corpus.seed([{"text": SECRET_CHUNK, "distance": 0.2, "page": 5}])
    monkeypatch.setattr("ragcli.ask.OllamaClient", lambda *_args, **_kwargs: corpus.fake.client())
    monkeypatch.setattr("ragcli.ask.available_bytes", lambda: PLENTY)
    original = ChromaIndex.open.__func__

    def open_shared(cls, data_dir, client_factory=None):
        return original(cls, data_dir, client_factory=corpus.factory)

    monkeypatch.setattr(ChromaIndex, "open", classmethod(open_shared))
    code, out, err = invoke(
        monkeypatch,
        ["ask", "--json", "--show-chunks", "--model", FOUR_B_MODEL, SECRET_QUESTION],
        cwd=tmp_path,
    )
    assert code == EXIT_OK
    payload = json.loads(out)
    assert payload["model"] == FOUR_B_MODEL
    assert payload["sources"][0]["page"] == 5
    assert SECRET_CHUNK not in out
    assert SECRET_CHUNK not in err
    assert SECRET_QUESTION not in out
    assert SECRET_QUESTION not in err
    assert "page=5" in err
    assert "頁=" not in err
    for token in ("source=", "來源=", "Question:", "Excerpts:", "問題：", "摘錄：", "種類="):
        assert token not in err
    assert chats(corpus.fake)[0]["model"] == FOUR_B_MODEL
    assert chats(corpus.fake)[0]["think"] is False


def test_cli_no_fallback_reaches_the_gate(monkeypatch, tmp_path):
    corpus = Corpus(tmp_path)
    corpus.seed([{"text": "旗標", "distance": 0.2}])
    monkeypatch.setattr("ragcli.ask.OllamaClient", lambda *_args, **_kwargs: corpus.fake.client())
    monkeypatch.setattr("ragcli.ask.available_bytes", lambda: SEVEN_GIB)
    original = ChromaIndex.open.__func__

    def open_shared(cls, data_dir, client_factory=None):
        return original(cls, data_dir, client_factory=corpus.factory)

    monkeypatch.setattr(ChromaIndex, "open", classmethod(open_shared))
    code, _out, err = invoke(
        monkeypatch,
        ["ask", "--no-fallback", "旗標問題"],
        cwd=tmp_path,
    )
    assert code == EXIT_RESOURCES
    assert "Did not switch to the fallback model." in err
    assert chats(corpus.fake) == []
