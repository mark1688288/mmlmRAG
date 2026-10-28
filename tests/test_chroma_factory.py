"""Chroma names, truncation, and the index factory.

The fake client never imports Chroma. The persistent test supplies vectors
itself and fails if MiniLM is constructed or called.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import sys
from pathlib import Path

import pytest

from ragcli.errors import (
    EXIT_MISMATCH,
    EXIT_NO_COLLECTION,
    EXIT_UNCLASSIFIED,
    EXIT_USAGE,
    RagError,
)
from ragcli.store.chroma import (
    CHROMA_DIRNAME,
    ChromaIndex,
    ClientSettings,
    IndexRecord,
    collection_name,
    embedding_for_width,
    truncate_normalize,
)
from ragcli.store.sqlite import chunk_id_for, image_id_for

DOC = "a" * 64
DOC_B = "b" * 64
MODEL = "qwen3-embedding:0.6b"
NUM_CTX = 2048
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*[a-z0-9]$")


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def norm(values) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def make_record(doc_id: str = DOC, ordinal: int = 0, **overrides) -> IndexRecord:
    embedding = overrides.pop("embedding", (3.0, 0.0, 0.0, 4.0))
    values = {
        "chunk_id": chunk_id_for(doc_id, ordinal),
        "doc_id": doc_id,
        "text": "退款規定只存在索引測試",
        "embedding": tuple(embedding),
        "source_path": f"/docs/{doc_id}.pdf",
        "filename": "退款規定.pdf",
        "kind": "text",
        "ordinal": ordinal,
        "page_start": -1,
        "page_end": -1,
        "image_id": "",
    }
    values.update(overrides)
    if "chunk_id" not in overrides:
        values["chunk_id"] = chunk_id_for(values["doc_id"], values["ordinal"])
    if not isinstance(values["embedding"], tuple):
        values["embedding"] = tuple(values["embedding"])
    return IndexRecord(**values)


class FakeCollection:
    def __init__(self, **kwargs) -> None:
        self.name = kwargs["name"]
        self.metadata = dict(kwargs["metadata"])
        self.configuration = kwargs["configuration"]
        self._embedding_function = kwargs["embedding_function"]
        self.upserts: list[dict] = []
        self.queries: list[dict] = []
        self.deletes: list[list[str]] = []
        self.rows: dict[str, tuple] = {}
        self.distance_for: dict[str, float] = {}

    def upsert(self, *, ids, embeddings, documents, metadatas) -> None:
        self.upserts.append(
            {
                "ids": list(ids),
                "embeddings": [list(vector) for vector in embeddings],
                "documents": list(documents),
                "metadatas": [dict(item) for item in metadatas],
            }
        )
        for chunk_id, vector, document, metadata in zip(
            ids, embeddings, documents, metadatas, strict=True
        ):
            self.rows[chunk_id] = (list(vector), document, dict(metadata))

    def query(self, **kwargs):
        if "query_texts" in kwargs:
            raise AssertionError("query_texts")
        if kwargs.get("query_embeddings") is None:
            raise AssertionError("missing query_embeddings")
        self.queries.append(kwargs)
        ids = list(self.rows)
        limit = kwargs["n_results"]
        chosen = ids[:limit]
        return {
            "ids": [chosen],
            "documents": [[self.rows[chunk_id][1] for chunk_id in chosen]],
            "metadatas": [[self.rows[chunk_id][2] for chunk_id in chosen]],
            "distances": [[self.distance_for.get(chunk_id, 0.01) for chunk_id in chosen]],
        }

    def delete(self, *, ids) -> None:
        self.deletes.append(list(ids))
        for chunk_id in ids:
            self.rows.pop(chunk_id, None)

    def get(self, *, ids, include):
        found = [chunk_id for chunk_id in ids if chunk_id in self.rows]
        return {
            "ids": found,
            "embeddings": [self.rows[chunk_id][0] for chunk_id in found],
        }

    def count(self) -> int:
        return len(self.rows)


class FakeClient:
    def __init__(self, path: Path, settings) -> None:
        self.path = path
        self.settings = settings
        self.created: list[dict] = []
        self.collections: dict[str, FakeCollection] = {}
        self.closed = False
        self.max_batch = 1

    def get_or_create_collection(self, **kwargs) -> FakeCollection:
        self.created.append(kwargs)
        name = kwargs["name"]
        if name not in self.collections:
            self.collections[name] = FakeCollection(**kwargs)
        return self.collections[name]

    def get_collection(self, name: str, embedding_function=None) -> FakeCollection:
        if embedding_function is not None:
            raise AssertionError("embedding_function")
        if name not in self.collections:
            raise ValueError(f"Collection {name} does not exist")
        return self.collections[name]

    def list_collections(self, limit=None, offset=None):
        rows = list(self.collections.values())
        start = 0 if offset is None else offset
        if limit is None:
            return rows[start:]
        return rows[start : start + limit]

    def get_max_batch_size(self) -> int:
        return self.max_batch

    def close(self) -> None:
        self.closed = True


def open_fake(tmp_path: Path) -> tuple[ChromaIndex, FakeClient]:
    holder: dict[str, FakeClient] = {}

    def factory(path: Path, settings) -> FakeClient:
        client = FakeClient(path, settings)
        holder["client"] = client
        return client

    index = ChromaIndex.open(tmp_path / "data", client_factory=factory)
    return index, holder["client"]


def test_default_collection_name_has_no_colon():
    assert "chromadb" not in sys.modules
    name = collection_name("  Qwen3-Embedding:0.6b  ", 1024)
    assert name == "emb-qwen3-embedding-0.6b-d1024"
    assert ":" not in name
    assert "qwen3-embedding:0.6b" not in name
    assert collection_name("org/qwen3-embedding:0.6b", 4) == "emb-org-qwen3-embedding-0.6b-d4"
    assert collection_name("a::b", 4) == "emb-a-b-d4"
    assert ".." not in collection_name("a..b", 8)
    assert collection_name("a..b", 8) == "emb-a.b-d8"


def test_long_model_tag_inserts_a_hash_and_stays_legal():
    short = "a" * 53
    assert collection_name(short, 1024) == f"emb-{short}-d1024"
    assert len(collection_name(short, 1024)) == 63

    longer = "b" * 54
    name = collection_name(longer, 1024)
    digest = hashlib.sha256(longer.encode("utf-8")).hexdigest()[:8]
    assert len(name) == 63
    assert name.startswith("emb-" + "b" * 21 + "-" + digest)
    assert name.endswith("b" * 22 + "-d1024")
    assert ":" not in name
    assert NAME_RE.fullmatch(name)

    tagged = "vendor/" + ("Qwen3-Embedding:" * 8) + "0.6b"
    tagged_name = collection_name(tagged, 1024)
    tagged_digest = hashlib.sha256(tagged.encode("utf-8")).hexdigest()[:8]
    assert tagged not in tagged_name
    assert ":" not in tagged_name
    assert f"-{tagged_digest}-" in tagged_name
    assert 3 <= len(tagged_name) <= 63
    assert NAME_RE.fullmatch(tagged_name)
    assert ".." not in tagged_name

    empty = "???"
    empty_digest = hashlib.sha256(empty.encode("utf-8")).hexdigest()[:8]
    assert collection_name(f"  {empty}  ", 1024) == f"emb-{empty_digest}-d1024"


def test_collection_name_rejects_bad_inputs():
    with pytest.raises(ValueError):
        collection_name("", 1024)
    with pytest.raises(ValueError):
        collection_name("qwen3-embedding:0.6b", 0)
    with pytest.raises(ValueError):
        collection_name("qwen3-embedding:0.6b", True)  # type: ignore[arg-type]


def test_truncate_normalize_has_unit_norm_and_rejects_zero():
    vector = [float(value) for value in range(1, 17)]
    shortened = truncate_normalize(vector, 8)
    head = vector[:8]
    head_norm = math.sqrt(sum(value * value for value in head))
    assert shortened == [value / head_norm for value in head]
    assert math.isclose(norm(shortened), 1.0)
    with pytest.raises(ValueError, match="zero vector"):
        truncate_normalize([0.0, 0.0, 5.0], 2)


def test_native_width_is_not_normalized():
    native = [3.0, 0.0, 0.0, 4.0]
    assert embedding_for_width(native, 4) == native
    assert norm(native) == 5
    fitted = embedding_for_width([3.0, 1.0, 9.0, 9.0], 2)
    assert fitted == truncate_normalize([3.0, 1.0, 9.0, 9.0], 2)
    assert math.isclose(norm(fitted), 1.0)
    with pytest.raises(RagError) as exc:
        embedding_for_width(native, 8)
    assert exc.value.code == EXIT_MISMATCH


def test_factory_passes_telemetry_cosine_and_no_embedding_function(tmp_path: Path):
    assert "chromadb" not in sys.modules
    index, client = open_fake(tmp_path)
    try:
        assert isinstance(client.settings, ClientSettings)
        assert client.settings.anonymized_telemetry is False
        assert client.path.name == CHROMA_DIRNAME
        assert mode(index.data_dir) == 0o700
        assert mode(index.chroma_dir) == 0o700

        native = make_record()
        other = make_record(
            DOC_B,
            1,
            embedding=(0.0, 5.0, 0.0, 0.0),
            text="另一句",
            filename="其他.pdf",
            source_path="/docs/other.pdf",
            page_start=2,
            page_end=3,
            kind="ocr",
            image_id=image_id_for(DOC_B, 1),
        )
        name = index.upsert([native, other], model=MODEL, dimensions=4, embed_num_ctx=NUM_CTX)
        assert name == "emb-qwen3-embedding-0.6b-d4"
        assert len(client.created) == 1
        created = client.created[0]
        assert created["name"] == name
        assert ":" not in created["name"]
        assert MODEL not in created["name"]
        assert created["embedding_function"] is None
        assert created["configuration"]["hnsw"]["space"] == "cosine"
        assert created["metadata"] == {
            "embed_model": MODEL,
            "dimensions": 4,
            "embed_num_ctx": NUM_CTX,
        }
        assert client.collections[name].upserts[0]["embeddings"] == [[3.0, 0.0, 0.0, 4.0]]
        assert client.collections[name].upserts[1]["embeddings"] == [[0.0, 5.0, 0.0, 0.0]]
        assert norm(client.collections[name].upserts[0]["embeddings"][0]) == 5
        stored_meta = client.collections[name].upserts[1]["metadatas"][0]
        assert stored_meta["page_start"] == 2
        assert stored_meta["image_id"] == image_id_for(DOC_B, 1)
        assert None not in stored_meta.values()

        index.upsert([native], model=MODEL, dimensions=4, embed_num_ctx=NUM_CTX)
        assert len(client.created) == 1

        hits = index.query(native.embedding, model=MODEL, dimensions=4, embed_num_ctx=NUM_CTX, n_results=1)
        assert len(hits) == 1
        assert hits[0].chunk_id == native.chunk_id
        assert hits[0].filename == "退款規定.pdf"
        assert client.collections[name].queries[0]["query_embeddings"] == [[3.0, 0.0, 0.0, 4.0]]
        assert "query_texts" not in client.collections[name].queries[0]

        short = make_record(embedding=(3.0, 1.0, 9.0, 9.0))
        short_name = index.upsert([short], model=MODEL, dimensions=2, embed_num_ctx=NUM_CTX)
        sent = client.collections[short_name].upserts[0]["embeddings"][0]
        assert sent == truncate_normalize([3.0, 1.0, 9.0, 9.0], 2)
        assert math.isclose(norm(sent), 1.0)

        index.delete([native.chunk_id, other.chunk_id])
        assert client.collections[name].deletes == [[native.chunk_id, other.chunk_id]]
        assert client.collections[short_name].deletes == [[native.chunk_id, other.chunk_id]]
        assert index.collection_names() == sorted([name, short_name])
    finally:
        index.close()
    assert client.closed
    with pytest.raises(RagError, match="closed"):
        index.count(model=MODEL, dimensions=4, embed_num_ctx=NUM_CTX)


def test_missing_collection_is_exit_11_and_partial_delete_is_usage(tmp_path: Path):
    index, _client = open_fake(tmp_path)
    try:
        with pytest.raises(RagError) as missing:
            index.query((1.0, 0.0), model=MODEL, dimensions=2, embed_num_ctx=NUM_CTX, n_results=1)
        assert missing.value.code == EXIT_NO_COLLECTION
        with pytest.raises(RagError) as usage:
            index.delete(["x"], model=MODEL)
        assert usage.value.code == EXIT_USAGE
        index.delete(["x"])
    finally:
        index.close()


def test_existing_directory_mode_is_left_alone(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    os.chmod(data, 0o755)
    index, _client = open_fake(tmp_path)
    try:
        assert mode(data) == 0o755
        assert mode(data / CHROMA_DIRNAME) == 0o700
    finally:
        index.close()


def test_symlink_directories_are_refused(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(RagError, match="symbolic link") as exc:
        ChromaIndex.open(link)
    assert exc.value.code == EXIT_UNCLASSIFIED

    data = tmp_path / "data"
    data.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (data / CHROMA_DIRNAME).symlink_to(outside, target_is_directory=True)
    with pytest.raises(RagError, match="symbolic link"):
        ChromaIndex.open(data)


def test_persistent_client_keeps_native_vectors_and_does_not_load_minilm(tmp_path: Path):
    import chromadb
    import chromadb.api.types as chroma_types
    import chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 as mini

    calls: list[str] = []
    original_init = mini.ONNXMiniLM_L6_V2.__init__
    original_call = mini.ONNXMiniLM_L6_V2.__call__

    def init_spy(self, *args, **kwargs):
        calls.append("init")
        return original_init(self, *args, **kwargs)

    def call_spy(self, *args, **kwargs):
        calls.append("call")
        return original_call(self, *args, **kwargs)

    def default_call_spy(self, input):  # noqa: A002
        calls.append("default")
        raise AssertionError("default embedding function was called")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mini.ONNXMiniLM_L6_V2, "__init__", init_spy)
    monkeypatch.setattr(mini.ONNXMiniLM_L6_V2, "__call__", call_spy)
    monkeypatch.setattr(chroma_types.DefaultEmbeddingFunction, "__call__", default_call_spy)
    seen: dict[str, object] = {}
    original_client = chromadb.PersistentClient

    def persist(*args, **kwargs):
        seen["settings"] = kwargs.get("settings")
        client = original_client(*args, **kwargs)
        original_open = client.get_or_create_collection

        def open_collection(*collection_args, **collection_kwargs):
            seen["create"] = collection_kwargs
            return original_open(*collection_args, **collection_kwargs)

        client.get_or_create_collection = open_collection
        return client

    monkeypatch.setattr(chromadb, "PersistentClient", persist)
    cache = Path.home() / ".cache" / "chroma" / "onnx_models"
    cache_existed = cache.exists()
    data = tmp_path / "data"
    index = None
    try:
        index = ChromaIndex.open(data)
        from chromadb.config import Settings

        assert isinstance(seen["settings"], Settings)
        assert seen["settings"].anonymized_telemetry is False

        with pytest.raises(RagError) as missing:
            index.count(model=MODEL, dimensions=4, embed_num_ctx=NUM_CTX)
        assert missing.value.code == EXIT_NO_COLLECTION

        native = make_record()
        far = make_record(DOC_B, 1, embedding=(0.0, 0.0, 3.0, 4.0), text="較遠的一句")
        name = index.upsert([native, far], model=MODEL, dimensions=4, embed_num_ctx=NUM_CTX)
        create = seen["create"]
        assert create["embedding_function"] is None
        assert create["configuration"]["hnsw"]["space"] == "cosine"
        assert ":" not in create["name"]
        assert create["metadata"]["embed_model"] == MODEL

        stored = index.get_embeddings(
            [native.chunk_id], model=MODEL, dimensions=4, embed_num_ctx=NUM_CTX
        )
        assert stored[native.chunk_id] == pytest.approx((3.0, 0.0, 0.0, 4.0))
        assert norm(stored[native.chunk_id]) == pytest.approx(5.0)

        hits = index.query(
            (3.0, 0.0, 0.0, 4.0),
            model=MODEL,
            dimensions=4,
            embed_num_ctx=NUM_CTX,
            n_results=2,
        )
        assert [hit.chunk_id for hit in hits] == [native.chunk_id, far.chunk_id]
        assert hits[0].distance < 1e-4
        assert hits[0].distance < hits[1].distance
        assert hits[0].text == native.text
        assert hits[0].page_start == -1
        assert hits[0].image_id == ""

        changed = make_record(text="路徑更新後的句子", source_path="/docs/moved.pdf")
        index.upsert([changed], model=MODEL, dimensions=4, embed_num_ctx=NUM_CTX)
        refreshed = index.query(
            (3.0, 0.0, 0.0, 4.0),
            model=MODEL,
            dimensions=4,
            embed_num_ctx=NUM_CTX,
            n_results=1,
        )
        assert refreshed[0].text == "路徑更新後的句子"
        assert refreshed[0].source_path == "/docs/moved.pdf"

        short = make_record(embedding=(3.0, 1.0, 9.0, 9.0))
        short_name = index.upsert([short], model=MODEL, dimensions=2, embed_num_ctx=NUM_CTX)
        short_stored = index.get_embeddings(
            [short.chunk_id], model=MODEL, dimensions=2, embed_num_ctx=NUM_CTX
        )
        assert len(short_stored[short.chunk_id]) == 2
        assert norm(short_stored[short.chunk_id]) == pytest.approx(1.0)
        assert short_stored[short.chunk_id][0] == pytest.approx(3.0 / math.sqrt(10.0))
        short_hits = index.query(
            (3.0, 1.0, 9.0, 9.0),
            model=MODEL,
            dimensions=2,
            embed_num_ctx=NUM_CTX,
            n_results=1,
        )
        assert short_hits[0].chunk_id == short.chunk_id
        assert short_hits[0].distance < 1e-4

        index.delete([native.chunk_id], model=MODEL, dimensions=4, embed_num_ctx=NUM_CTX)
        assert index.count(model=MODEL, dimensions=4, embed_num_ctx=NUM_CTX) == 1
        assert index.collection_names() == sorted([name, short_name])
        index.close()
        index = None

        reopened = ChromaIndex.open(data)
        try:
            hits = reopened.query(
                (0.0, 0.0, 3.0, 4.0),
                model=MODEL,
                dimensions=4,
                embed_num_ctx=NUM_CTX,
                n_results=1,
            )
            assert hits[0].chunk_id == far.chunk_id
            assert hits[0].text == "較遠的一句"
        finally:
            reopened.close()
    finally:
        if index is not None:
            index.close()
        monkeypatch.undo()

    assert calls == []
    if not cache_existed:
        assert not cache.exists()


def test_persistent_client_rejects_a_collection_that_breaks_the_contract(tmp_path: Path):
    import chromadb
    from chromadb.config import Settings

    data = tmp_path / "data"
    chroma_dir = data / CHROMA_DIRNAME
    chroma_dir.mkdir(parents=True)
    raw = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    raw.get_or_create_collection(
        name=collection_name(MODEL, 4),
        embedding_function=None,
        configuration={"hnsw": {"space": "l2"}},
        metadata={"embed_model": MODEL, "dimensions": 4, "embed_num_ctx": NUM_CTX},
    )
    raw.get_or_create_collection(
        name=collection_name("qwen3-embedding:4b", 8),
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embed_model": "other-model", "dimensions": 8, "embed_num_ctx": NUM_CTX},
    )
    raw.close()

    index = ChromaIndex.open(data)
    try:
        with pytest.raises(RagError, match="cosine") as space:
            index.upsert([make_record()], model=MODEL, dimensions=4, embed_num_ctx=NUM_CTX)
        assert space.value.code == EXIT_MISMATCH
        with pytest.raises(RagError, match="embedding model") as model:
            index.upsert(
                [make_record(embedding=(1.0,) * 8)],
                model="qwen3-embedding:4b",
                dimensions=8,
                embed_num_ctx=NUM_CTX,
            )
        assert model.value.code == EXIT_MISMATCH
    finally:
        index.close()

    check = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        assert check.get_collection(collection_name(MODEL, 4), embedding_function=None).count() == 0
        assert (
            check.get_collection(
                collection_name("qwen3-embedding:4b", 8),
                embedding_function=None,
            ).count()
            == 0
        )
    finally:
        check.close()
