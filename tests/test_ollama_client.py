"""Ollama HTTP shape, context reload, and the vm_stat gate.

These tests never open a socket and never run a model pull. Memory probes
are either parsed from text or pointed at a fake process runner.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import httpx
import pytest

from ragcli.errors import (
    EXIT_MODEL_MISSING,
    EXIT_OLLAMA_UNREACHABLE,
    EXIT_RESOURCES,
    EXIT_UNCLASSIFIED,
    EXIT_USAGE,
    RagError,
)
from ragcli.ollama import (
    EMBED_READ_TIMEOUT_SECONDS,
    FOUR_B_MODEL,
    GENERATE_READ_TIMEOUT_SECONDS,
    NINE_B_MODEL,
    VISION_READ_TIMEOUT_SECONDS,
    ChatMessage,
    OllamaClient,
    _run_readonly,
    available_bytes,
    choose_generator,
    generation_min_bytes,
    parse_vm_stat,
    request_timeout,
)

REPO = Path(__file__).resolve().parents[1]
NINE_BYTES = 9663676416
FOUR_BYTES = 6442450944
EMBED = "qwen3-embedding:0.6b"
SECRET = "僅供嵌入的秘密句子"
THOUGHT = "不可以當成答案的思考"


class FakeOllama:
    """Scripted loopback. State changes only when a request mutates it."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str, object]] = []
        self.loaded: list[dict] = []
        self.stuck = False
        self.force_context: int | None = None
        self.remember = True
        self.drop_on_zero_keep_alive = False
        self.vectors = [[0.25, 0.5]]
        self.chat_message: dict = {"role": "assistant", "content": "答案", "thinking": ""}
        self.tag_names = [EMBED, NINE_B_MODEL, FOUR_B_MODEL]
        self.pull_body = '{"status":"pulling manifest"}\n{"status":"success"}\n'
        self.connect_error = False
        self.status_for: dict[str, int] = {}

    def client(self, base_url: str = "http://127.0.0.1:11434") -> OllamaClient:
        return OllamaClient(base_url, transport=httpx.MockTransport(self.handler))

    def paths(self) -> list[str]:
        return [path for _method, path, _body in self.seen]

    def bodies(self, path: str) -> list:
        return [body for _method, seen, body in self.seen if seen == path]

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.connect_error:
            raise httpx.ConnectError("down")
        body = json.loads(request.content.decode()) if request.content else None
        self.seen.append((request.method, request.url.path, body))
        path = request.url.path
        override = self.status_for.get(path)
        if override is not None:
            return httpx.Response(override, json={"error": "model not found"})
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.11"})
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
        if path == "/api/pull":
            return httpx.Response(200, text=self.pull_body)
        return httpx.Response(500, json={"error": "unexpected"})

    def _generate(self, body: dict) -> httpx.Response:
        model = body["model"]
        if str(model).startswith("qwen3-embedding"):
            return httpx.Response(400, json={"error": "does not support generate"})
        if body.get("keep_alive") == 0:
            self._drop(model)
        return httpx.Response(200, json={"response": ""})

    def _embed(self, body: dict) -> httpx.Response:
        model = body["model"]
        if body.get("input") == ".":
            self._drop(model)
            return httpx.Response(200, json={"embeddings": [[0.0]], "prompt_eval_count": 1})
        self._show(model, body["options"]["num_ctx"], body.get("keep_alive"))
        return httpx.Response(
            200,
            json={"embeddings": self.vectors, "prompt_eval_count": 4},
        )

    def _chat(self, body: dict) -> httpx.Response:
        self._show(body["model"], body["options"]["num_ctx"], body.get("keep_alive"))
        return httpx.Response(
            200,
            json={
                "message": self.chat_message,
                "prompt_eval_count": 3,
                "eval_count": 5,
            },
        )

    def _drop(self, model: str) -> None:
        if self.stuck:
            return
        self.loaded = [item for item in self.loaded if item.get("name") != model]

    def _show(self, model: str, num_ctx: int, keep_alive) -> None:
        if keep_alive in (0, "0") and self.drop_on_zero_keep_alive:
            self._drop(model)
            return
        if not self.remember:
            self._drop(model)
            return
        ctx = self.force_context if self.force_context is not None else num_ctx
        self.loaded = [item for item in self.loaded if item.get("name") != model]
        self.loaded.append({"name": model, "context_length": ctx})


def vm_text(free: int, inactive: int, speculative: int, *, page_size=16384, purgeable=0, header=True) -> str:
    lines = []
    if header:
        lines.append(f"Mach Virtual Memory Statistics: (page size of {page_size} bytes)")
    lines.extend(
        [
            f"Pages free:                                {free}.",
            f"Pages inactive:                           {inactive}.",
            f"Pages speculative:                        {speculative}.",
            f"Pages purgeable:                          {purgeable}.",
        ]
    )
    return "\n".join(lines) + "\n"


SEVEN_GIB = vm_text(200000, 200000, 58752, purgeable=9_999_999)
FIVE_GIB = vm_text(100000, 100000, 127680, purgeable=50)


def _gate(**overrides):
    values = {
        "preferred": NINE_B_MODEL,
        "fallback": FOUR_B_MODEL,
        "tags": [NINE_B_MODEL, FOUR_B_MODEL],
        "available": 7 * 1024**3,
        "nine_b_min_bytes": NINE_BYTES,
        "four_b_min_bytes": FOUR_BYTES,
        "allow_fallback": True,
    }
    values.update(overrides)
    return choose_generator(**values)


@pytest.fixture
def logs(caplog):
    parent = logging.getLogger("ragcli")
    parent.handlers.clear()
    parent.propagate = True
    caplog.set_level(logging.DEBUG, logger="ragcli.ollama")
    return caplog


def test_thresholds_match_the_design():
    assert 9 * 1024**3 == NINE_BYTES
    assert 6 * 1024**3 == FOUR_BYTES
    assert generation_min_bytes(NINE_B_MODEL, nine_b_min_bytes=NINE_BYTES, four_b_min_bytes=FOUR_BYTES) == NINE_BYTES
    assert generation_min_bytes(FOUR_B_MODEL, nine_b_min_bytes=NINE_BYTES, four_b_min_bytes=FOUR_BYTES) == FOUR_BYTES
    assert generation_min_bytes("custom:large", nine_b_min_bytes=NINE_BYTES, four_b_min_bytes=FOUR_BYTES) == NINE_BYTES


def test_vm_stat_ignores_purgeable_pages():
    assert parse_vm_stat(SEVEN_GIB) == 7 * 1024**3
    bare = vm_text(200000, 200000, 58752, purgeable=0)
    huge = vm_text(200000, 200000, 58752, purgeable=10**9)
    assert parse_vm_stat(bare) == parse_vm_stat(huge)


def test_vm_stat_header_page_size_wins_over_the_argument():
    text = vm_text(1, 1, 1, page_size=4096)
    assert parse_vm_stat(text, page_size=16384) == 3 * 4096


def test_vm_stat_without_header_uses_the_supplied_page_size():
    text = vm_text(2, 3, 4, header=False)
    assert parse_vm_stat(text, page_size=16384) == 9 * 16384


def test_vm_stat_missing_a_counter_is_exit_13():
    text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 1.\n"
    with pytest.raises(RagError) as exc:
        parse_vm_stat(text)
    assert exc.value.code == EXIT_RESOURCES


def test_available_bytes_uses_only_vm_stat_when_the_header_is_present():
    calls = []

    def run(argv):
        calls.append(list(argv))
        return SEVEN_GIB

    assert available_bytes(run=run) == 7 * 1024**3
    assert calls == [["/usr/bin/vm_stat"]]


def test_available_bytes_asks_sysctl_when_the_header_has_no_page_size():
    calls = []

    def run(argv):
        calls.append(list(argv))
        if argv[-1:] == ["hw.pagesize"]:
            return "16384\n"
        return vm_text(1, 1, 1, header=False)

    assert available_bytes(run=run) == 3 * 16384
    assert calls == [
        ["/usr/bin/vm_stat"],
        ["/usr/sbin/sysctl", "-n", "hw.pagesize"],
    ]


def test_default_runner_is_vm_stat_without_a_shell(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))

        class Result:
            returncode = 0
            stdout = SEVEN_GIB
            stderr = ""

        return Result()

    monkeypatch.setattr("ragcli.ollama.subprocess.run", fake_run)
    assert available_bytes() == 7 * 1024**3
    assert calls[0][0] == ["/usr/bin/vm_stat"]
    assert "shell" not in calls[0][1]


def test_probe_allowlist_does_not_execute_other_commands(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("不應執行探針以外的命令")

    monkeypatch.setattr(subprocess, "run", boom)
    for argv in (["/usr/bin/memory_pressure"], ["/bin/echo", "x"], ["vm_stat"]):
        with pytest.raises(RagError) as exc:
            _run_readonly(argv)
        assert exc.value.code == EXIT_RESOURCES


def test_source_never_names_a_shell_pull_or_memory_probe():
    text = (REPO / "src" / "ragcli" / "ollama.py").read_text(encoding="utf-8")
    lock = (REPO / "src" / "ragcli" / "models_lock.py").read_text(encoding="utf-8")
    for banned in ("memory_pressure", "shell=True", "os.system", "ollama pull", "Popen"):
        assert banned not in text
        assert banned not in lock


def test_seven_gib_does_not_load_9b_and_selects_4b(logs):
    available = parse_vm_stat(SEVEN_GIB)
    chosen = _gate(available=available)
    assert chosen == FOUR_B_MODEL
    assert str(available) in logs.text
    assert str(NINE_BYTES) in logs.text
    assert "改用" in logs.text


def test_seven_gib_without_4b_in_tags_is_exit_13():
    available = parse_vm_stat(SEVEN_GIB)
    with pytest.raises(RagError) as exc:
        _gate(available=available, tags=[NINE_B_MODEL])
    assert exc.value.code == EXIT_RESOURCES
    assert str(available) in exc.value.message
    assert str(NINE_BYTES) in exc.value.message
    assert "ollama pull" not in exc.value.message
    assert "後備模型不在已拉取的清單裡" in exc.value.message


def test_no_fallback_does_not_select_4b_when_9b_does_not_fit():
    with pytest.raises(RagError) as exc:
        _gate(allow_fallback=False)
    assert exc.value.code == EXIT_RESOURCES
    assert FOUR_B_MODEL not in exc.value.message.split("沒有改用")[0]
    assert "沒有改用後備模型" in exc.value.message
    assert str(7 * 1024**3) in exc.value.message
    assert str(NINE_BYTES) in exc.value.message


def test_both_models_below_their_bars_is_exit_13():
    available = parse_vm_stat(FIVE_GIB)
    assert available == 5 * 1024**3
    with pytest.raises(RagError) as exc:
        _gate(available=available)
    assert exc.value.code == EXIT_RESOURCES
    assert str(NINE_BYTES) in exc.value.message
    assert str(FOUR_BYTES) in exc.value.message
    assert str(available) in exc.value.message


def test_exact_threshold_loads_that_model(logs):
    assert _gate(available=NINE_BYTES) == NINE_B_MODEL
    assert logs.text == ""
    assert _gate(preferred=FOUR_B_MODEL, available=FOUR_BYTES) == FOUR_B_MODEL


def test_preferred_4b_is_not_upgraded_to_9b():
    assert _gate(preferred=FOUR_B_MODEL, available=10 * 1024**3) == FOUR_B_MODEL
    with pytest.raises(RagError) as exc:
        _gate(preferred=FOUR_B_MODEL, available=FOUR_BYTES - 1)
    assert exc.value.code == EXIT_RESOURCES
    assert NINE_B_MODEL not in exc.value.message
    assert str(FOUR_BYTES) in exc.value.message


def test_missing_tags_are_exit_7_without_a_shell_command():
    with pytest.raises(RagError) as exc:
        _gate(tags=[])
    assert exc.value.code == EXIT_MODEL_MISSING
    assert "rag models pull" in exc.value.message
    assert NINE_B_MODEL in exc.value.message
    assert FOUR_B_MODEL in exc.value.message
    assert "ollama pull" not in exc.value.message


def test_only_4b_is_selected_when_9b_was_never_pulled(logs):
    assert _gate(tags=[FOUR_B_MODEL], available=10 * 1024**3) == FOUR_B_MODEL
    assert "改用" in logs.text
    with pytest.raises(RagError) as exc:
        _gate(tags=[FOUR_B_MODEL], allow_fallback=False)
    assert exc.value.code == EXIT_MODEL_MISSING


def test_unknown_generator_cannot_use_the_4b_bar():
    assert _gate(preferred="custom:large", tags=["custom:large"], available=NINE_BYTES) == "custom:large"
    with pytest.raises(RagError) as exc:
        _gate(preferred="custom:large", tags=["custom:large"], available=7 * 1024**3, allow_fallback=False)
    assert exc.value.code == EXIT_RESOURCES
    assert str(NINE_BYTES) in exc.value.message


def test_non_loopback_client_does_not_open_a_connection(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("不應建立 HTTP 客戶端")

    monkeypatch.setattr("ragcli.ollama.httpx.Client", boom)
    with pytest.raises(RagError) as exc:
        OllamaClient("http://192.168.1.2:11434")
    assert exc.value.code == EXIT_USAGE


def test_version_tags_and_ps_round_trip():
    fake = FakeOllama()
    fake.loaded = [{"name": NINE_B_MODEL, "context_length": 8192}]
    client = fake.client()
    assert client.version() == "0.40.11"
    assert client.tags() == (EMBED, NINE_B_MODEL, FOUR_B_MODEL)
    loaded = client.ps()
    assert loaded[0].name == NINE_B_MODEL
    assert loaded[0].context_length == 8192


def test_requests_stay_on_the_normalized_loopback_host():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        return httpx.Response(200, json={"version": "0.40.11"})

    client = OllamaClient("HTTP://LOCALHOST:11434/", transport=httpx.MockTransport(handler))
    assert client.base_url == "http://localhost:11434"
    assert client.version() == "0.40.11"
    assert seen == ["localhost"]


def test_redirect_is_not_followed():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        return httpx.Response(302, headers={"Location": "http://example.com/api/version"})

    client = OllamaClient("http://127.0.0.1:11434", transport=httpx.MockTransport(handler))
    with pytest.raises(RagError) as exc:
        client.version()
    assert exc.value.code == EXIT_UNCLASSIFIED
    assert seen == ["127.0.0.1"]
    assert "重新導向" in exc.value.message


def test_client_disables_proxy_env_and_uses_a_two_second_connect(monkeypatch):
    seen = {}
    real = httpx.Client

    def capture(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr("ragcli.ollama.httpx.Client", capture)
    fake = FakeOllama()
    assert fake.client().version() == "0.40.11"
    assert seen["trust_env"] is False
    assert seen["follow_redirects"] is False
    assert seen["timeout"].connect == 2.0


def test_operation_timeouts(monkeypatch):
    reads = []
    real = request_timeout

    def wrap(read):
        reads.append(read)
        return real(read)

    monkeypatch.setattr("ragcli.ollama.request_timeout", wrap)
    fake = FakeOllama()
    client = fake.client()
    client.embed(EMBED, [SECRET], num_ctx=2048)
    client.chat(
        NINE_B_MODEL,
        [ChatMessage("user", "問題")],
        num_ctx=8192,
        num_predict=768,
        temperature=0.2,
    )
    client.chat(
        NINE_B_MODEL,
        [ChatMessage("user", "描述這張圖，並保留圖中文字。", ("QUJDRA==",))],
        num_ctx=4096,
        num_predict=400,
        temperature=0.1,
        keep_alive="30m",
        read_timeout=VISION_READ_TIMEOUT_SECONDS,
    )
    client.pull(EMBED)
    assert EMBED_READ_TIMEOUT_SECONDS in reads
    assert GENERATE_READ_TIMEOUT_SECONDS in reads
    assert VISION_READ_TIMEOUT_SECONDS in reads
    assert None in reads
    assert request_timeout(120).connect == 2.0
    assert request_timeout(120).read == 120.0


def test_embed_request_shape_and_logs_omit_the_text(logs):
    fake = FakeOllama()
    vectors = fake.client().embed(EMBED, [SECRET], num_ctx=2048)
    assert vectors == [[0.25, 0.5]]
    body = fake.bodies("/api/embed")[-1]
    assert body == {
        "model": EMBED,
        "input": [SECRET],
        "truncate": False,
        "keep_alive": "30m",
        "options": {"num_ctx": 2048},
    }
    assert "dimensions" not in body
    assert "think" not in body
    assert SECRET not in logs.text
    assert "prompt_eval_count=4" in logs.text


def test_chat_think_is_top_level_and_thinking_is_not_the_answer(logs):
    fake = FakeOllama()
    fake.chat_message = {"role": "assistant", "content": "", "thinking": THOUGHT}
    result = fake.client().chat(
        NINE_B_MODEL,
        [
            ChatMessage("system", "系統提示"),
            ChatMessage("user", SECRET),
        ],
        num_ctx=8192,
        num_predict=768,
        temperature=0.2,
    )
    body = fake.bodies("/api/chat")[-1]
    assert body["stream"] is False
    assert body["think"] is False
    assert "think" not in body["options"]
    assert body["keep_alive"] == "0"
    assert body["options"] == {"num_ctx": 8192, "num_predict": 768, "temperature": 0.2}
    assert body["messages"][1] == {"role": "user", "content": SECRET}
    assert result.content == ""
    assert result.thinking == THOUGHT
    assert result.thinking_only is True
    assert result.content != THOUGHT
    assert SECRET not in logs.text
    assert THOUGHT not in logs.text
    assert "eval_count=5" in logs.text


def test_vision_request_uses_raw_base64_and_vision_options():
    fake = FakeOllama()
    image = "QUJDRA=="
    fake.client().chat(
        NINE_B_MODEL,
        [ChatMessage("user", "描述這張圖，並保留圖中文字。", (image,))],
        num_ctx=4096,
        num_predict=400,
        temperature=0.1,
        keep_alive="30m",
        read_timeout=VISION_READ_TIMEOUT_SECONDS,
    )
    body = fake.bodies("/api/chat")[-1]
    assert body["options"] == {"num_ctx": 4096, "num_predict": 400, "temperature": 0.1}
    assert body["messages"][0]["images"] == [image]
    assert body["think"] is False
    with pytest.raises(RagError) as exc:
        fake.client().chat(
            NINE_B_MODEL,
            [ChatMessage("user", "圖", ("data:image/jpeg;base64,QQ==",))],
            num_ctx=4096,
            num_predict=400,
            temperature=0.1,
        )
    assert exc.value.code == EXIT_USAGE
    assert len(fake.bodies("/api/chat")) == 1


def test_matching_context_is_not_reloaded():
    fake = FakeOllama()
    fake.loaded = [{"name": EMBED, "context_length": 2048}]
    fake.client().embed(EMBED, [SECRET], num_ctx=2048)
    assert fake.bodies("/api/generate") == []
    assert fake.bodies("/api/embed") == [
        {
            "model": EMBED,
            "input": [SECRET],
            "truncate": False,
            "keep_alive": "30m",
            "options": {"num_ctx": 2048},
        }
    ]


def test_wrong_context_unloads_before_the_request():
    fake = FakeOllama()
    fake.loaded = [{"name": EMBED, "context_length": 32768}]
    fake.client().embed(EMBED, [SECRET], num_ctx=2048)
    inputs = [body["input"] for body in fake.bodies("/api/embed")]
    assert inputs == [".", [SECRET]]
    assert fake.bodies("/api/generate") == [{"model": EMBED, "keep_alive": 0}]
    assert fake.loaded == [{"name": EMBED, "context_length": 2048}]


def test_missing_context_length_is_treated_as_a_mismatch():
    fake = FakeOllama()
    fake.loaded = [{"name": EMBED}]
    fake.client().embed(EMBED, [SECRET], num_ctx=2048)
    assert fake.bodies("/api/embed")[0]["input"] == "."
    assert fake.bodies("/api/embed")[1]["options"] == {"num_ctx": 2048}


def test_context_still_wrong_after_load_is_exit_13():
    fake = FakeOllama()
    fake.force_context = 4096
    with pytest.raises(RagError) as exc:
        fake.client().embed(EMBED, [SECRET], num_ctx=2048)
    assert exc.value.code == EXIT_RESOURCES
    assert "4096" in exc.value.message
    assert "2048" in exc.value.message
    inputs = [body["input"] for body in fake.bodies("/api/embed")]
    assert inputs[0] == [SECRET]
    assert inputs[-1] == "."
    assert fake.loaded == []


def test_a_second_model_that_survives_unload_is_exit_13_before_embed():
    fake = FakeOllama()
    fake.stuck = True
    fake.loaded = [{"name": "other:model", "context_length": 100}]
    with pytest.raises(RagError) as exc:
        fake.client().embed(EMBED, [SECRET], num_ctx=2048)
    assert exc.value.code == EXIT_RESOURCES
    assert "/api/embed" not in fake.paths()
    assert SECRET not in json.dumps(fake.seen)


def test_keep_alive_zero_may_leave_ps_empty():
    fake = FakeOllama()
    fake.drop_on_zero_keep_alive = True
    vectors = fake.client().embed(EMBED, [SECRET], num_ctx=2048, keep_alive="0")
    assert vectors == [[0.25, 0.5]]
    assert fake.loaded == []


def test_batch_keep_alive_must_still_show_the_model():
    fake = FakeOllama()
    fake.remember = False
    with pytest.raises(RagError) as exc:
        fake.client().embed(EMBED, [SECRET], num_ctx=2048, keep_alive="30m")
    assert exc.value.code == EXIT_RESOURCES
    assert EMBED in exc.value.message


def test_unload_tries_generate_then_embed_for_an_embedding_model():
    fake = FakeOllama()
    fake.loaded = [{"name": EMBED, "context_length": 2048}]
    client = fake.client()
    assert client.unload_all() == [EMBED]
    assert fake.bodies("/api/generate") == [{"model": EMBED, "keep_alive": 0}]
    assert fake.bodies("/api/embed") == [{"model": EMBED, "input": ".", "keep_alive": 0}]
    assert fake.loaded == []


def test_unload_of_a_generator_does_not_call_embed():
    fake = FakeOllama()
    fake.loaded = [{"name": NINE_B_MODEL, "context_length": 8192}]
    fake.client().unload_model(NINE_B_MODEL)
    assert fake.bodies("/api/generate") == [{"model": NINE_B_MODEL, "keep_alive": 0}]
    assert fake.bodies("/api/embed") == []
    assert fake.loaded == []


def test_unsafe_model_names_are_not_sent():
    fake = FakeOllama()
    client = fake.client()
    with pytest.raises(RagError) as exc:
        client.unload_model("qwen;rm")
    assert exc.value.code == EXIT_USAGE
    assert fake.seen == []
    fake.loaded = [{"name": "bad name", "context_length": 1}]
    with pytest.raises(RagError) as exc:
        client.unload_all()
    assert exc.value.code == EXIT_RESOURCES
    assert "/api/generate" not in fake.paths()


def test_connection_errors_are_exit_6_and_missing_models_are_exit_7():
    fake = FakeOllama()
    fake.connect_error = True
    with pytest.raises(RagError) as exc:
        fake.client().version()
    assert exc.value.code == EXIT_OLLAMA_UNREACHABLE
    fake.connect_error = False
    fake.status_for["/api/embed"] = 404
    with pytest.raises(RagError) as exc:
        fake.client().embed(EMBED, [SECRET], num_ctx=2048)
    assert exc.value.code == EXIT_MODEL_MISSING
    assert "rag models pull" in exc.value.message
    assert EMBED in exc.value.message
    assert "ollama pull" not in exc.value.message


def test_ps_still_loaded_after_unload_is_exit_13():
    fake = FakeOllama()
    fake.stuck = True
    fake.loaded = [{"name": NINE_B_MODEL, "context_length": 8192}]
    with pytest.raises(RagError) as exc:
        fake.client().unload_all()
    assert exc.value.code == EXIT_RESOURCES
    assert "ps" in exc.value.message
