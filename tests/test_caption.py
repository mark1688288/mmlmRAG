"""Vision captions against a fake Ollama. No socket, no pull, and no database."""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path

import httpx
import pytest
from PIL import Image

from ragcli.config import load_config
from ragcli.errors import (
    EXIT_MODEL_MISSING,
    EXIT_OLLAMA_UNREACHABLE,
    EXIT_RESOURCES,
    EXIT_UNCLASSIFIED,
    RagError,
)
from ragcli.images.caption import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    VISION_JPEG_QUALITY,
    VISION_LONG_SIDE,
    CaptionResult,
    VisionImage,
    VisionSettings,
    describe_batch,
    jpeg_for_vision,
)
from ragcli.ollama import NINE_B_MODEL, OllamaClient, VISION_READ_TIMEOUT_SECONDS

REPO = Path(__file__).resolve().parents[1]
NINE_BYTES = 9_663_676_416
FOUR_BYTES = 6_442_450_944
EMBED = "qwen3-embedding:0.6b"
DESIGN_SYSTEM = (
    "你是圖像描述器。只根據圖像內容用繁體中文寫一段描述，120字以內。"
    "如果圖入面有讀得到的文字，照實抄低。不要估計圖入面沒有出現的事實。不要輸出思考過程。"
)


def solid_png(width: int, height: int, *, rgba: bool = False) -> bytes:
    mode = "RGBA" if rgba else "RGB"
    color = (10, 20, 30, 128) if rgba else (10, 20, 30)
    image = Image.new(mode, (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def png_ihdr(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )


def image_for(image_id: str, png: bytes | None = None) -> VisionImage:
    return VisionImage(image_id, solid_png(32, 24) if png is None else png)


class FakeOllama:
    """Scripted loopback. State changes only when a request mutates it."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str, object]] = []
        self.loaded: list[dict] = []
        self.stuck = False
        self.tag_names = [EMBED, NINE_B_MODEL, "qwen3.5:4b"]
        self.replies: list[dict] = []
        self.fail_after: int | None = None
        self.chat_count = 0

    def client(self) -> OllamaClient:
        return OllamaClient("http://127.0.0.1:11434", transport=httpx.MockTransport(self.handler))

    def paths(self) -> list[str]:
        return [path for _method, path, _body in self.seen]

    def bodies(self, path: str) -> list:
        return [body for _method, seen, body in self.seen if seen == path]

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode()) if request.content else None
        self.seen.append((request.method, request.url.path, body))
        path = request.url.path
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": name} for name in self.tag_names]})
        if path == "/api/ps":
            return httpx.Response(200, json={"models": [dict(item) for item in self.loaded]})
        if path == "/api/generate":
            return self._generate(body)
        if path == "/api/embed":
            return self._embed(body)
        if path == "/api/chat":
            return self._chat(body)
        return httpx.Response(500, json={"error": "unexpected"})

    def _generate(self, body: dict) -> httpx.Response:
        model = body["model"]
        if str(model).startswith("qwen3-embedding"):
            return httpx.Response(400, json={"error": "does not support generate"})
        if body.get("keep_alive") == 0:
            self._drop(model)
        return httpx.Response(200, json={"response": ""})

    def _embed(self, body: dict) -> httpx.Response:
        if body.get("input") == ".":
            self._drop(body["model"])
            return httpx.Response(200, json={"embeddings": [[0.0]]})
        return httpx.Response(500, json={"error": "unexpected embed"})

    def _chat(self, body: dict) -> httpx.Response:
        self.chat_count += 1
        if self.fail_after is not None and self.chat_count > self.fail_after:
            raise httpx.ConnectError("down")
        self._show(body["model"], body["options"]["num_ctx"])
        if self.replies:
            message = self.replies.pop(0)
        else:
            message = {"role": "assistant", "content": "預設描述", "thinking": ""}
        return httpx.Response(
            200,
            json={"message": message, "prompt_eval_count": 3, "eval_count": 5},
        )

    def _drop(self, model: str) -> None:
        if self.stuck:
            return
        self.loaded = [item for item in self.loaded if item.get("name") != model]

    def _show(self, model: str, num_ctx: int) -> None:
        self.loaded = [item for item in self.loaded if item.get("name") != model]
        self.loaded.append({"name": model, "context_length": num_ctx})


def test_prompts_and_settings_match_the_design():
    config = load_config(REPO / "rag.toml")
    assert SYSTEM_PROMPT == DESIGN_SYSTEM
    assert USER_PROMPT == "描述這張圖，並保留圖中文字。"
    assert VISION_LONG_SIDE == 1280
    assert VISION_JPEG_QUALITY == 85
    assert VisionSettings.from_config(config) == VisionSettings()
    assert VisionSettings().num_ctx == 4096
    assert VisionSettings().num_predict == 400
    assert VisionSettings().temperature == 0.1
    assert VisionSettings().nine_b_min_bytes == NINE_BYTES
    assert VisionSettings().four_b_min_bytes == FOUR_BYTES


def test_jpeg_shrinks_the_long_side_and_does_not_enlarge(monkeypatch):
    saved = {}
    real_save = Image.Image.save

    def spy(self, fp, format=None, **kwargs):
        saved["format"] = format
        saved["quality"] = kwargs.get("quality")
        return real_save(self, fp, format=format, **kwargs)

    monkeypatch.setattr(Image.Image, "save", spy)
    wide = jpeg_for_vision(solid_png(2000, 1000))
    tall = jpeg_for_vision(solid_png(1000, 2000))
    small = jpeg_for_vision(solid_png(64, 32))
    thin = jpeg_for_vision(solid_png(1281, 10))
    flat = jpeg_for_vision(solid_png(40, 40, rgba=True))
    assert saved["format"] == "JPEG"
    assert saved["quality"] == 85
    assert _jpeg_size(wide) == (1280, 640)
    assert _jpeg_size(tall) == (640, 1280)
    assert _jpeg_size(small) == (64, 32)
    assert _jpeg_size(thin) == (1280, 9)
    assert _jpeg_size(flat) == (40, 40)
    assert wide.startswith(b"\xff\xd8")
    assert not wide.startswith(b"\x89PNG")


def test_over_pixel_png_is_not_decoded(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("decoded")

    monkeypatch.setattr("ragcli.images.caption.Image.open", boom)
    with pytest.raises(RagError) as exc:
        jpeg_for_vision(png_ihdr(20_000, 20_000))
    assert exc.value.code == EXIT_UNCLASSIFIED
    assert exc.value.message == "圖像超過像素上限。"


def test_request_shape_unload_once_and_logs_omit_the_caption(caplog, monkeypatch):
    def forbid():
        raise AssertionError("vm_stat")

    monkeypatch.setattr("ragcli.images.caption.available_bytes", forbid)
    caplog.set_level(logging.DEBUG, logger="ragcli")
    fake = FakeOllama()
    fake.replies = [
        {"role": "assistant", "content": "SENTINEL_CAPTION", "thinking": "SENTINEL_THINKING"}
    ]
    client = fake.client()
    calls = []
    original = client.chat

    def wrapped(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    client.chat = wrapped  # type: ignore[method-assign]
    result = describe_batch(
        [image_for("img1")],
        client,
        available=NINE_BYTES,
    )
    assert result == (CaptionResult("img1", "caption_done", "SENTINEL_CAPTION", NINE_B_MODEL),)
    body = fake.bodies("/api/chat")[0]
    assert body["model"] == NINE_B_MODEL
    assert body["stream"] is False
    assert body["think"] is False
    assert "think" not in body["options"]
    assert body["keep_alive"] == "30m"
    assert body["options"] == {"num_ctx": 4096, "num_predict": 400, "temperature": 0.1}
    assert body["messages"][0] == {"role": "system", "content": DESIGN_SYSTEM}
    assert body["messages"][1]["content"] == USER_PROMPT
    encoded = body["messages"][1]["images"][0]
    assert isinstance(encoded, str) and not encoded.startswith("data:")
    raw = base64.b64decode(encoded)
    assert raw.startswith(b"\xff\xd8")
    assert _jpeg_size(raw) == (32, 24)
    assert calls[0]["read_timeout"] == VISION_READ_TIMEOUT_SECONDS
    assert calls[0]["keep_alive"] == "30m"
    chat_at = [index for index, item in enumerate(fake.seen) if item[1] == "/api/chat"]
    unload_at = _unload_indexes(fake)
    assert chat_at
    assert unload_at == [unload_at[-1]]
    assert unload_at[0] > chat_at[-1]
    assert fake.loaded == []
    assert "SENTINEL_CAPTION" not in caplog.text
    assert "SENTINEL_THINKING" not in caplog.text
    assert encoded not in caplog.text


def test_two_images_keep_the_model_until_the_batch_ends():
    fake = FakeOllama()
    results = describe_batch(
        [image_for("img1"), image_for("img2")],
        fake.client(),
        available=NINE_BYTES,
    )
    assert [item.status for item in results] == ["caption_done", "caption_done"]
    assert [item.text for item in results] == ["預設描述", "預設描述"]
    chat_at = [index for index, item in enumerate(fake.seen) if item[1] == "/api/chat"]
    assert len(chat_at) == 2
    between = fake.seen[chat_at[0] + 1 : chat_at[1]]
    assert all(item[1] != "/api/generate" for item in between)
    assert all(fake.seen[index][2]["keep_alive"] == "30m" for index in chat_at)
    unloads = [
        body
        for body in fake.bodies("/api/generate")
        if isinstance(body, dict) and body.get("keep_alive") == 0
    ]
    assert unloads == [{"model": NINE_B_MODEL, "keep_alive": 0}]


def test_foreign_model_and_wrong_context_are_unloaded_before_chat():
    fake = FakeOllama()
    fake.loaded = [
        {"name": EMBED, "context_length": 2048},
        {"name": NINE_B_MODEL, "context_length": 8192},
    ]
    describe_batch([image_for("img1")], fake.client(), available=NINE_BYTES)
    chat_at = fake.paths().index("/api/chat")
    assert fake.paths().index("/api/embed") < chat_at
    assert fake.bodies("/api/embed")[0]["input"] == "."
    assert fake.bodies("/api/embed")[0]["model"] == EMBED
    generates = [
        (index, item[2])
        for index, item in enumerate(fake.seen)
        if item[1] == "/api/generate" and isinstance(item[2], dict)
    ]
    assert generates[0][1] == {"model": EMBED, "keep_alive": 0}
    assert generates[1][1] == {"model": NINE_B_MODEL, "keep_alive": 0}
    assert generates[0][0] < generates[1][0] < chat_at
    assert generates[-1][0] > chat_at
    assert generates[-1][1] == {"model": NINE_B_MODEL, "keep_alive": 0}
    assert fake.bodies("/api/chat")[0]["options"]["num_ctx"] == 4096
    assert fake.bodies("/api/chat")[0]["model"] == NINE_B_MODEL
    assert fake.loaded == []


def test_a_second_model_that_survives_unload_is_exit_13_before_chat():
    fake = FakeOllama()
    fake.stuck = True
    fake.loaded = [{"name": EMBED, "context_length": 2048}]
    with pytest.raises(RagError) as exc:
        describe_batch([image_for("img1")], fake.client(), available=NINE_BYTES)
    assert exc.value.code == EXIT_RESOURCES
    assert "/api/chat" not in fake.paths()


def test_empty_content_is_retried_once_and_thinking_is_not_the_caption(caplog):
    caplog.set_level(logging.DEBUG, logger="ragcli")
    fake = FakeOllama()
    fake.replies = [
        {"role": "assistant", "content": "", "thinking": "SENTINEL_THINKING"},
        {"role": "assistant", "content": "橫條圖顯示增長", "thinking": "SENTINEL_THINKING"},
    ]
    result = describe_batch([image_for("img1")], fake.client(), available=NINE_BYTES)
    assert result == (CaptionResult("img1", "caption_done", "橫條圖顯示增長", NINE_B_MODEL),)
    assert len(fake.bodies("/api/chat")) == 2
    assert "SENTINEL_THINKING" not in caplog.text
    assert result[0].text != "SENTINEL_THINKING"


def test_two_empty_replies_are_caption_failed_and_the_next_image_still_runs(caplog):
    caplog.set_level(logging.DEBUG, logger="ragcli")
    fake = FakeOllama()
    fake.replies = [
        {"role": "assistant", "content": "  \n", "thinking": "SENTINEL_THINKING"},
        {"role": "assistant", "content": "", "thinking": "SENTINEL_THINKING"},
        {"role": "assistant", "content": "第二張", "thinking": ""},
    ]
    results = describe_batch(
        [image_for("img1"), image_for("img2")],
        fake.client(),
        available=NINE_BYTES,
    )
    assert results[0] == CaptionResult("img1", "caption_failed", None, NINE_B_MODEL)
    assert results[1] == CaptionResult("img2", "caption_done", "第二張", NINE_B_MODEL)
    assert len(fake.bodies("/api/chat")) == 3
    assert len([body for body in fake.bodies("/api/generate") if body.get("keep_alive") == 0]) == 1
    assert "SENTINEL_THINKING" not in caplog.text


def test_a_failed_request_still_unloads():
    fake = FakeOllama()
    fake.fail_after = 1
    with pytest.raises(RagError) as exc:
        describe_batch(
            [image_for("img1"), image_for("img2")],
            fake.client(),
            available=NINE_BYTES,
        )
    assert exc.value.code == EXIT_OLLAMA_UNREACHABLE
    chat_at = [index for index, item in enumerate(fake.seen) if item[1] == "/api/chat"]
    unload_at = _unload_indexes(fake)
    assert len(chat_at) == 2
    assert unload_at
    assert unload_at[-1] > chat_at[-1]


def test_seven_gib_selects_4b_and_a_tighter_gate_does_not_chat():
    fake = FakeOllama()
    result = describe_batch([image_for("img1")], fake.client(), available=7 * 1024**3)
    assert result[0].model == "qwen3.5:4b"
    assert fake.bodies("/api/chat")[0]["model"] == "qwen3.5:4b"

    blocked = FakeOllama()
    with pytest.raises(RagError) as exc:
        describe_batch([image_for("img1")], blocked.client(), available=5 * 1024**3)
    assert exc.value.code == EXIT_RESOURCES
    assert blocked.bodies("/api/chat") == []
    assert "9663676416" in exc.value.message
    assert "6442450944" in exc.value.message

    no_fallback = FakeOllama()
    with pytest.raises(RagError) as exc:
        describe_batch(
            [image_for("img1")],
            no_fallback.client(),
            VisionSettings(allow_fallback=False),
            available=7 * 1024**3,
        )
    assert exc.value.code == EXIT_RESOURCES
    assert "沒有改用後備模型" in exc.value.message
    assert no_fallback.bodies("/api/chat") == []

    missing_4b = FakeOllama()
    missing_4b.tag_names = [EMBED, NINE_B_MODEL]
    with pytest.raises(RagError) as exc:
        describe_batch([image_for("img1")], missing_4b.client(), available=7 * 1024**3)
    assert exc.value.code == EXIT_RESOURCES
    assert "後備模型不在已拉取的清單裡" in exc.value.message

    missing = FakeOllama()
    missing.tag_names = [EMBED]
    with pytest.raises(RagError) as exc:
        describe_batch([image_for("img1")], missing.client(), available=NINE_BYTES)
    assert exc.value.code == EXIT_MODEL_MISSING
    assert "rag models pull" in exc.value.message
    assert "ollama pull" not in exc.value.message
    assert missing.bodies("/api/chat") == []


def test_default_available_bytes_is_used_when_the_caller_does_not_pass_one(monkeypatch):
    seen = {}

    def fake_available():
        seen["called"] = True
        return NINE_BYTES

    monkeypatch.setattr("ragcli.images.caption.available_bytes", fake_available)
    fake = FakeOllama()
    describe_batch([image_for("img1")], fake.client())
    assert seen["called"] is True


def test_decode_failures_do_not_load_a_model_when_nothing_can_be_sent():
    fake = FakeOllama()
    results = describe_batch(
        [VisionImage("bad", b"not an image")],
        fake.client(),
        available=NINE_BYTES,
    )
    assert results == (CaptionResult("bad", "caption_failed", None, None),)
    assert fake.seen == []


def test_one_bad_image_does_not_drop_the_other():
    fake = FakeOllama()
    results = describe_batch(
        [VisionImage("bad", b"not an image"), image_for("good")],
        fake.client(),
        available=NINE_BYTES,
    )
    assert results[0] == CaptionResult("bad", "caption_failed", None, None)
    assert results[1].status == "caption_done"
    assert results[1].image_id == "good"
    assert results[1].model == NINE_B_MODEL


def test_empty_batch_and_duplicate_ids_do_not_call_ollama(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = FakeOllama()
    assert describe_batch((), fake.client(), available=NINE_BYTES) == ()
    png = solid_png(16, 16)
    with pytest.raises(ValueError, match="重複"):
        describe_batch(
            [VisionImage("same", png), VisionImage("same", png)],
            fake.client(),
            available=NINE_BYTES,
        )
    assert fake.seen == []
    assert list(tmp_path.iterdir()) == []


def test_caption_module_does_not_touch_storage_or_memory_pressure():
    source = Path(describe_batch.__code__.co_filename).read_text(encoding="utf-8")
    for banned in ("sqlite", "memory_pressure", "subprocess", "chromadb"):
        assert banned not in source


def _unload_indexes(fake: FakeOllama) -> list[int]:
    return [
        index
        for index, item in enumerate(fake.seen)
        if item[1] == "/api/generate"
        and isinstance(item[2], dict)
        and item[2].get("keep_alive") == 0
    ]


def _jpeg_size(raw: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        return image.size
