"""rag models pull / unload, and the Ollama flock.

Pull is a loopback HTTP call. These tests fail if a subprocess is asked to
run, and they do not download a model.
"""

from __future__ import annotations

import importlib.util
import io
import os
import stat
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from ragcli.cli import main
from ragcli.config import load_config
from ragcli.errors import EXIT_OK, EXIT_RESOURCES, EXIT_UNCLASSIFIED, EXIT_USAGE, RagError
from ragcli.models_lock import LOCK_WAIT_MESSAGE, exclusive_model, ollama_lock
from ragcli.ollama import (
    FOUR_B_MODEL,
    NINE_B_MODEL,
    ChatMessage,
    OllamaClient,
    model_pull_list,
    pull_models,
)

_CLIENT_TESTS = importlib.util.spec_from_file_location(
    "ollama_client_tests",
    Path(__file__).with_name("test_ollama_client.py"),
)
_CLIENT_MODULE = importlib.util.module_from_spec(_CLIENT_TESTS)
assert _CLIENT_TESTS is not None and _CLIENT_TESTS.loader is not None
_CLIENT_TESTS.loader.exec_module(_CLIENT_MODULE)
EMBED = _CLIENT_MODULE.EMBED
FakeOllama = _CLIENT_MODULE.FakeOllama

HOLDER = """
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR | os.O_NOFOLLOW)
fcntl.flock(fd, fcntl.LOCK_EX)
sys.stdout.write("held\\n")
sys.stdout.flush()
sys.stdin.read()
"""

PROBE = """
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR | os.O_NOFOLLOW)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.stdout.write("busy\\n")
else:
    sys.stdout.write("free\\n")
"""


def invoke(monkeypatch, argv, *, cwd):
    monkeypatch.chdir(cwd)
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def write_config(directory: Path, body: str = "") -> None:
    (directory / "rag.toml").write_text(body, encoding="utf-8")


def use_fake(monkeypatch, fake: FakeOllama) -> None:
    def factory(base_url, **kwargs):
        return OllamaClient(base_url, transport=httpx.MockTransport(fake.handler))

    monkeypatch.setattr("ragcli.cli.OllamaClient", factory)


def pulled(fake: FakeOllama) -> list[str]:
    return [body["model"] for body in fake.bodies("/api/pull")]


def lock_path(data: Path) -> Path:
    path = data / "locks" / "ollama.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    return path


def hold(path: Path):
    proc = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert proc.stdout.readline() == b"held\n"
    return proc


def let_go(proc) -> None:
    if proc.poll() is None:
        proc.stdin.close()
        proc.wait(timeout=5)


def probe(data: Path) -> str:
    path = data / "locks" / "ollama.lock"
    completed = subprocess.run(
        [sys.executable, "-c", PROBE, str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def test_default_pull_order_is_embed_then_9b_then_4b(tmp_path):
    write_config(tmp_path)
    config = load_config(tmp_path / "rag.toml", cwd=tmp_path)
    assert model_pull_list(config) == [EMBED, NINE_B_MODEL, FOUR_B_MODEL]


def test_pull_flags_select_the_documented_subsets(tmp_path):
    write_config(tmp_path)
    config = load_config(tmp_path / "rag.toml", cwd=tmp_path)
    assert model_pull_list(config, embed_only=True) == [EMBED]
    assert model_pull_list(config, generator_only=True) == [NINE_B_MODEL, FOUR_B_MODEL]
    assert model_pull_list(config, no_fallback_model=True) == [EMBED, NINE_B_MODEL]
    assert model_pull_list(config, generator_only=True, no_fallback_model=True) == [NINE_B_MODEL]
    assert model_pull_list(config, embed_only=True, no_fallback_model=True) == [EMBED]


def test_pull_list_uses_config_names_and_drops_duplicates(tmp_path):
    write_config(
        tmp_path,
        '[embed]\nmodel = "custom-embed:1b"\n'
        '[generate]\nmodel = "custom-gen:9b"\nfallback_model = "custom-gen:4b"\n',
    )
    config = load_config(tmp_path / "rag.toml", cwd=tmp_path)
    assert model_pull_list(config) == ["custom-embed:1b", "custom-gen:9b", "custom-gen:4b"]
    write_config(
        tmp_path,
        '[embed]\nmodel = "qwen3.5:9b"\n'
        '[generate]\nmodel = "qwen3.5:9b"\nfallback_model = "qwen3.5:4b"\n',
    )
    config = load_config(tmp_path / "rag.toml", cwd=tmp_path)
    assert model_pull_list(config) == [NINE_B_MODEL, FOUR_B_MODEL]


def test_pull_list_rejects_a_name_that_cannot_be_sent(tmp_path):
    write_config(tmp_path, '[embed]\nmodel = "has space"\n')
    config = load_config(tmp_path / "rag.toml", cwd=tmp_path)
    with pytest.raises(RagError) as exc:
        model_pull_list(config)
    assert exc.value.code == EXIT_USAGE


def test_pull_models_posts_one_at_a_time_then_unloads(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError(f"不應執行子程序 {args} {kwargs}")

    monkeypatch.setattr(subprocess, "run", boom)
    fake = FakeOllama()
    fake.loaded = []
    pull_models(fake.client(), [EMBED, NINE_B_MODEL, FOUR_B_MODEL])
    assert pulled(fake) == [EMBED, NINE_B_MODEL, FOUR_B_MODEL]
    assert fake.bodies("/api/pull") == [
        {"model": name, "stream": True} for name in pulled(fake)
    ]
    assert "/api/generate" not in fake.paths()
    assert fake.paths().count("/api/ps") >= 1


def test_pull_then_unload_clears_a_model_left_in_ps():
    fake = FakeOllama()

    def handler(request: httpx.Request) -> httpx.Response:
        response = FakeOllama.handler(fake, request)
        if request.url.path == "/api/pull":
            fake.loaded = [{"name": NINE_B_MODEL, "context_length": 4096}]
        return response

    fake.handler = handler
    pull_models(fake.client(), [NINE_B_MODEL])
    assert {"model": NINE_B_MODEL, "keep_alive": 0} in fake.bodies("/api/generate")
    assert fake.loaded == []


def test_pull_that_leaves_ps_occupied_is_exit_13():
    fake = FakeOllama()
    fake.stuck = True

    def handler(request: httpx.Request) -> httpx.Response:
        response = FakeOllama.handler(fake, request)
        if request.url.path == "/api/pull":
            fake.loaded = [{"name": NINE_B_MODEL, "context_length": 4096}]
        return response

    fake.handler = handler
    with pytest.raises(RagError) as exc:
        pull_models(fake.client(), [NINE_B_MODEL])
    assert exc.value.code == EXIT_RESOURCES


def test_cli_pull_writes_the_names_and_does_not_shell_out(monkeypatch, tmp_path):
    write_config(tmp_path)
    fake = FakeOllama()
    use_fake(monkeypatch, fake)

    def boom(*args, **kwargs):
        raise AssertionError(f"不應執行子程序 {args} {kwargs}")

    monkeypatch.setattr(subprocess, "run", boom)
    code, out, err = invoke(monkeypatch, ["models", "pull"], cwd=tmp_path)
    assert code == EXIT_OK, err
    assert out == f"Pulled {EMBED}\nPulled {NINE_B_MODEL}\nPulled {FOUR_B_MODEL}\n"
    assert pulled(fake) == [EMBED, NINE_B_MODEL, FOUR_B_MODEL]
    assert "ollama pull" not in err
    assert "ollama pull" not in out
    assert mode(tmp_path / "data") == 0o700
    assert mode(tmp_path / "data" / "locks") == 0o700
    assert mode(tmp_path / "data" / "locks" / "ollama.lock") == 0o600


@pytest.mark.parametrize(
    ("argv", "names"),
    [
        (["models", "pull", "--embed-only"], [EMBED]),
        (["models", "pull", "--generator-only"], [NINE_B_MODEL, FOUR_B_MODEL]),
        (["models", "pull", "--no-fallback-model"], [EMBED, NINE_B_MODEL]),
        (["models", "pull", "--generator-only", "--no-fallback-model"], [NINE_B_MODEL]),
    ],
)
def test_cli_pull_flags(monkeypatch, tmp_path, argv, names):
    write_config(tmp_path)
    fake = FakeOllama()
    use_fake(monkeypatch, fake)
    code, out, err = invoke(monkeypatch, argv, cwd=tmp_path)
    assert code == EXIT_OK, err
    assert pulled(fake) == names
    assert out == "".join(f"Pulled {name}\n" for name in names)
    assert "ollama pull" not in err


def test_cli_pull_uses_config_model_names(monkeypatch, tmp_path):
    write_config(
        tmp_path,
        '[embed]\nmodel = "custom-embed:1b"\n'
        '[generate]\nmodel = "custom-gen:9b"\nfallback_model = "custom-gen:4b"\n',
    )
    fake = FakeOllama()
    use_fake(monkeypatch, fake)
    code, out, err = invoke(monkeypatch, ["models", "pull"], cwd=tmp_path)
    assert code == EXIT_OK, err
    assert pulled(fake) == ["custom-embed:1b", "custom-gen:9b", "custom-gen:4b"]
    assert "custom-embed:1b" in out


def test_cli_unload_clears_ps_and_reports_the_names(monkeypatch, tmp_path):
    write_config(tmp_path)
    fake = FakeOllama()
    fake.loaded = [{"name": EMBED, "context_length": 2048}]
    use_fake(monkeypatch, fake)
    code, out, err = invoke(monkeypatch, ["models", "unload"], cwd=tmp_path)
    assert code == EXIT_OK, err
    assert out == f"Unloaded {EMBED}\n"
    assert fake.loaded == []
    assert fake.bodies("/api/embed") == [{"model": EMBED, "input": ".", "keep_alive": 0}]
    assert "ollama pull" not in err


def test_cli_unload_of_an_empty_ps(monkeypatch, tmp_path):
    write_config(tmp_path)
    fake = FakeOllama()
    use_fake(monkeypatch, fake)
    code, out, err = invoke(monkeypatch, ["models", "unload"], cwd=tmp_path)
    assert code == EXIT_OK, err
    assert out == "No models are loaded.\n"
    assert err == "" or "ollama pull" not in err


def test_cli_unreachable_ollama_is_exit_6(monkeypatch, tmp_path):
    write_config(tmp_path)
    fake = FakeOllama()
    fake.connect_error = True
    use_fake(monkeypatch, fake)
    code, out, err = invoke(monkeypatch, ["models", "pull", "--embed-only"], cwd=tmp_path)
    assert code == 6
    assert out == ""
    assert "連不上" in err
    assert "ollama pull" not in err


def test_cli_pull_failure_does_not_print_success(monkeypatch, tmp_path):
    write_config(tmp_path)
    fake = FakeOllama()
    fake.pull_body = '{"error":"missing"}\n'
    use_fake(monkeypatch, fake)
    code, out, err = invoke(monkeypatch, ["models", "pull", "--embed-only"], cwd=tmp_path)
    assert code == EXIT_UNCLASSIFIED
    assert out == ""
    assert "失敗" in err
    assert "ollama pull" not in err


def test_bad_model_name_does_not_create_data_or_connect(monkeypatch, tmp_path):
    write_config(tmp_path, '[embed]\nmodel = "has space"\n')

    def boom(*_args, **_kwargs):
        raise AssertionError("不應連線")

    monkeypatch.setattr("ragcli.cli.OllamaClient", boom)
    code, out, err = invoke(monkeypatch, ["models", "pull"], cwd=tmp_path)
    assert code == EXIT_USAGE
    assert out == ""
    assert "模型名稱" in err
    assert not (tmp_path / "data").exists()


def test_lock_timeout_flag_rejects_a_negative_number(monkeypatch, tmp_path):
    write_config(tmp_path)
    code, _out, _err = invoke(
        monkeypatch,
        ["models", "pull", "--lock-timeout", "-1"],
        cwd=tmp_path,
    )
    assert code == EXIT_USAGE
    assert not (tmp_path / "data").exists()


def test_models_pull_help_lists_the_flags(monkeypatch, tmp_path):
    code, out, err = invoke(monkeypatch, ["models", "pull", "--help"], cwd=tmp_path)
    assert code == EXIT_OK
    assert err == ""
    for flag in ("--embed-only", "--generator-only", "--no-fallback-model", "--lock-timeout"):
        assert flag in out
    assert "Pull only the embedding model" in out
    assert "只拉取嵌入模型" not in out


def test_new_data_dir_is_private(tmp_path):
    data = tmp_path / "data"
    with ollama_lock(data, timeout=1):
        pass
    assert mode(data) == 0o700
    assert mode(data / "locks") == 0o700
    assert mode(data / "locks" / "ollama.lock") == 0o600


def test_existing_data_dir_keeps_its_mode(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    os.chmod(data, 0o755)
    with ollama_lock(data, timeout=1):
        pass
    assert mode(data) == 0o755
    assert mode(data / "locks") == 0o700


def test_symlink_data_dir_and_lock_file_are_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(RagError) as exc:
        with ollama_lock(link, timeout=0):
            pass
    assert exc.value.code != 0
    data = tmp_path / "data"
    outside = tmp_path / "outside"
    outside.write_text("keep", encoding="utf-8")
    lock = lock_path(data)
    lock.unlink()
    lock.symlink_to(outside)
    with pytest.raises(RagError):
        with ollama_lock(data, timeout=0):
            pass
    assert outside.read_text(encoding="utf-8") == "keep"


def test_lock_timeout_zero_does_not_print_the_wait_notice(tmp_path):
    data = tmp_path / "data"
    path = lock_path(data)
    proc = hold(path)
    stderr = io.StringIO()
    try:
        with pytest.raises(RagError) as exc:
            with ollama_lock(data, timeout=0, stderr=stderr):
                pass
    finally:
        let_go(proc)
    assert exc.value.code == EXIT_RESOURCES
    assert "等待 Ollama 鎖超過 0 秒" in exc.value.message
    assert stderr.getvalue() == ""
    assert probe(data) == "free"


def test_lock_wait_notice_every_fifteen_seconds(tmp_path):
    data = tmp_path / "data"
    path = lock_path(data)
    proc = hold(path)
    stderr = io.StringIO()
    clock = Clock()
    try:
        with pytest.raises(RagError) as exc:
            with ollama_lock(
                data,
                timeout=31,
                stderr=stderr,
                now=clock.now,
                sleep=clock.sleep,
                poll=5,
            ):
                pass
    finally:
        let_go(proc)
    assert exc.value.code == EXIT_RESOURCES
    assert stderr.getvalue().splitlines() == [LOCK_WAIT_MESSAGE, LOCK_WAIT_MESSAGE]
    assert "31" in exc.value.message
    assert probe(data) == "free"


def test_keyboard_interrupt_releases_the_lock(tmp_path):
    data = tmp_path / "data"
    with pytest.raises(KeyboardInterrupt):
        with ollama_lock(data, timeout=1):
            raise KeyboardInterrupt
    assert probe(data) == "free"


def test_exclusive_model_unloads_strangers_and_the_target(tmp_path):
    fake = FakeOllama()
    fake.loaded = [{"name": FOUR_B_MODEL, "context_length": 8192}]
    client = fake.client()
    with exclusive_model(tmp_path / "data", client, NINE_B_MODEL, 8192):
        client.chat(
            NINE_B_MODEL,
            [ChatMessage("user", "問題")],
            num_ctx=8192,
            num_predict=768,
            temperature=0.2,
        )
    assert fake.bodies("/api/generate")[0] == {"model": FOUR_B_MODEL, "keep_alive": 0}
    assert fake.bodies("/api/chat")[0]["think"] is False
    assert fake.bodies("/api/chat")[0]["options"]["num_ctx"] == 8192
    assert fake.bodies("/api/generate")[-1] == {"model": NINE_B_MODEL, "keep_alive": 0}
    assert fake.loaded == []


def test_exclusive_model_releases_the_lock_after_failure(tmp_path):
    data = tmp_path / "data"
    fake = FakeOllama()
    with pytest.raises(RuntimeError):
        with exclusive_model(data, fake.client(), EMBED, 2048):
            raise RuntimeError("停止")
    assert probe(data) == "free"


def test_cli_lock_timeout_does_not_open_ollama(monkeypatch, tmp_path):
    write_config(tmp_path)
    data = tmp_path / "data"
    # The command resolves data_dir from rag.toml, which is tmp_path / "data".
    proc = hold(lock_path(data))

    def boom(*_args, **_kwargs):
        raise AssertionError("鎖逾時前不應建立客戶端")

    monkeypatch.setattr("ragcli.cli.OllamaClient", boom)
    try:
        code, out, err = invoke(
            monkeypatch,
            ["--data-dir", str(data), "models", "unload", "--lock-timeout", "0"],
            cwd=tmp_path,
        )
    finally:
        let_go(proc)
    assert code == EXIT_RESOURCES
    assert out == ""
    assert "等待 Ollama 鎖" in err
    assert "ollama pull" not in err
