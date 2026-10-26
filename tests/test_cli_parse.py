"""CLI parsing, exit codes, config paths, and the loopback URL gate."""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

import pytest

from ragcli.cli import SUPPORTED_SUFFIXES, main
from ragcli.errors import (
    EXIT_DISK,
    EXIT_DOCUMENT,
    EXIT_MISMATCH,
    EXIT_MODEL_MISSING,
    EXIT_NO_CHUNKS,
    EXIT_NO_COLLECTION,
    EXIT_NOT_FOUND,
    EXIT_OK,
    EXIT_OLLAMA_UNREACHABLE,
    EXIT_QUESTION,
    EXIT_RESOURCES,
    EXIT_UNCLASSIFIED,
    EXIT_UNSUPPORTED_TYPE,
    EXIT_USAGE,
)

REPO = Path(__file__).resolve().parents[1]


def invoke(monkeypatch, argv, *, cwd=None, stdin=""):
    if cwd is not None:
        monkeypatch.chdir(cwd)
    stream = stdin if not isinstance(stdin, str) else io.StringIO(stdin)
    monkeypatch.setattr(sys, "stdin", stream)
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def write_config(directory: Path, body: str = "") -> Path:
    path = directory / "rag.toml"
    path.write_text(body, encoding="utf-8")
    return path


def shown(text: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in text.splitlines())


class _StdinBoom(io.StringIO):
    def read(self, *args, **kwargs):
        raise AssertionError("不應讀取標準輸入")

    def readline(self, *args, **kwargs):
        raise AssertionError("不應讀取標準輸入")


def test_exit_code_constants():
    assert [
        EXIT_OK,
        EXIT_UNCLASSIFIED,
        EXIT_USAGE,
        EXIT_UNSUPPORTED_TYPE,
        EXIT_NOT_FOUND,
        EXIT_DOCUMENT,
        EXIT_OLLAMA_UNREACHABLE,
        EXIT_MODEL_MISSING,
        EXIT_NO_CHUNKS,
        EXIT_QUESTION,
        EXIT_DISK,
        EXIT_NO_COLLECTION,
        EXIT_MISMATCH,
        EXIT_RESOURCES,
    ] == list(range(14))


def test_supported_suffixes_match_design():
    assert SUPPORTED_SUFFIXES == {
        ".pdf",
        ".docx",
        ".pptx",
        ".txt",
        ".md",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".tif",
        ".tiff",
        ".gif",
        ".bmp",
        ".heic",
        ".heif",
    }


def test_help_lists_commands_without_reading_config(monkeypatch, tmp_path):
    write_config(tmp_path, "[\n")
    code, out, err = invoke(monkeypatch, ["--help"], cwd=tmp_path)
    assert code == EXIT_OK
    assert err == ""
    for name in (
        "ingest",
        "ask",
        "reembed",
        "use",
        "delete",
        "status",
        "models",
        "config",
    ):
        assert name in out


def test_ingest_help_lists_flags(monkeypatch, tmp_path):
    code, out, _err = invoke(monkeypatch, ["ingest", "--help"], cwd=tmp_path)
    assert code == EXIT_OK
    for flag in (
        "--force",
        "--resume",
        "--no-ocr",
        "--no-vision",
        "--no-embed",
        "--strict",
        "--max-pages",
        "--max-bytes",
    ):
        assert flag in out


def test_repo_config_show_matches_design(monkeypatch, tmp_path):
    code, out, err = invoke(
        monkeypatch,
        ["--config", str(REPO / "rag.toml"), "config", "show"],
        cwd=tmp_path,
    )
    assert code == EXIT_OK
    assert err == ""
    values = shown(out)
    assert values["config_path"] == str((REPO / "rag.toml").resolve())
    assert values["data_dir"] == str((REPO / "data").resolve())
    assert Path(values["data_dir"]).is_absolute()
    assert values == {
        "config_path": values["config_path"],
        "data_dir": values["data_dir"],
        "ollama.base_url": "http://127.0.0.1:11434",
        "embed.model": "qwen3-embedding:0.6b",
        "embed.native_dimensions": "1024",
        "embed.dimensions": "1024",
        "embed.num_ctx": "2048",
        "embed.batch_size": "8",
        "generate.model": "qwen3.5:9b",
        "generate.fallback_model": "qwen3.5:4b",
        "generate.num_ctx": "8192",
        "generate.vision_num_ctx": "4096",
        "generate.num_predict": "768",
        "generate.vision_num_predict": "400",
        "generate.temperature": "0.2",
        "generate.vision_temperature": "0.1",
        "retrieve.k": "5",
        "retrieve.max_cosine_distance": "0.85",
        "retrieve.max_question_tokens": "512",
        "chunk.target_tokens": "480",
        "chunk.max_tokens": "640",
        "chunk.overlap_tokens": "80",
        "ocr.languages": "zh-Hant,zh-Hans,en-US",
        "ocr.recognition_level": "accurate",
        "ocr.confidence_threshold": "0.3",
        "ocr.rich_tokens": "40",
        "limits.max_pages": "500",
        "limits.max_images": "300",
        "limits.max_bytes": "209715200",
        "limits.min_free_bytes": "2147483648",
        "limits.max_pixels": "20000000",
        "limits.nine_b_min_bytes": "9663676416",
        "limits.four_b_min_bytes": "6442450944",
    }
    assert "think" not in out
    assert "api_key" not in out


def test_data_dir_is_relative_to_config_file_not_cwd(monkeypatch, tmp_path):
    config_dir = tmp_path / "project"
    config_dir.mkdir()
    write_config(config_dir, '[paths]\ndata_dir = "../corpus"\n')
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    code, out, err = invoke(
        monkeypatch,
        ["--config", str(config_dir / "rag.toml"), "config", "show"],
        cwd=cwd,
    )
    assert code == EXIT_OK, err
    assert shown(out)["data_dir"] == str((tmp_path / "corpus").resolve())
    assert not (tmp_path / "corpus").exists()


def test_data_dir_override_is_relative_to_cwd(monkeypatch, tmp_path):
    config_dir = tmp_path / "project"
    config_dir.mkdir()
    write_config(config_dir, '[paths]\ndata_dir = "data"\n')
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    code, out, err = invoke(
        monkeypatch,
        [
            "--config",
            str(config_dir / "rag.toml"),
            "--data-dir",
            "other",
            "config",
            "show",
        ],
        cwd=cwd,
    )
    assert code == EXIT_OK, err
    assert shown(out)["data_dir"] == str((cwd / "other").resolve())


def test_tilde_data_dir_expands(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    write_config(tmp_path, '[paths]\ndata_dir = "~/corpus"\n')
    code, out, err = invoke(
        monkeypatch,
        ["--config", str(tmp_path / "rag.toml"), "config", "show"],
        cwd=tmp_path,
    )
    assert code == EXIT_OK, err
    assert shown(out)["data_dir"] == str((home / "corpus").resolve())


def test_walk_up_finds_rag_toml(monkeypatch, tmp_path):
    write_config(tmp_path, '[paths]\ndata_dir = "data"\n')
    cwd = tmp_path / "a" / "b"
    cwd.mkdir(parents=True)
    code, out, err = invoke(monkeypatch, ["config", "show"], cwd=cwd)
    assert code == EXIT_OK, err
    values = shown(out)
    assert values["config_path"] == str((tmp_path / "rag.toml").resolve())
    assert values["data_dir"] == str((tmp_path / "data").resolve())


def test_missing_explicit_config_does_not_walk(monkeypatch):
    missing = REPO / "no-such-rag.toml"
    code, _out, err = invoke(
        monkeypatch,
        ["--config", str(missing), "config", "show"],
        cwd=REPO,
    )
    assert code == EXIT_USAGE
    assert "找不到設定檔" in err


def test_invalid_toml(monkeypatch, tmp_path):
    write_config(tmp_path, "[\n")
    code, _out, err = invoke(
        monkeypatch,
        ["--config", str(tmp_path / "rag.toml"), "config", "show"],
        cwd=tmp_path,
    )
    assert code == EXIT_USAGE
    assert "TOML" in err


def test_think_setting_is_rejected(monkeypatch, tmp_path):
    write_config(tmp_path, "[generate]\nthink = false\n")
    code, _out, err = invoke(
        monkeypatch,
        ["--config", str(tmp_path / "rag.toml"), "config", "show"],
        cwd=tmp_path,
    )
    assert code == EXIT_USAGE
    assert "think" in err


def test_bool_k_is_rejected(monkeypatch, tmp_path):
    write_config(tmp_path, "[retrieve]\nk = true\n")
    code, _out, err = invoke(
        monkeypatch,
        ["--config", str(tmp_path / "rag.toml"), "status"],
        cwd=tmp_path,
    )
    assert code == EXIT_USAGE
    assert "retrieve.k" in err


@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.1.2:11434",
        "http://10.0.0.1:11434",
        "http://0.0.0.0:11434",
        "http://127.0.0.2:11434",
        "http://example.com:11434",
        "file:///tmp/ollama",
    ],
)
def test_non_loopback_url_is_rejected(monkeypatch, tmp_path, url):
    write_config(tmp_path, f'[ollama]\nbase_url = "{url}"\n')
    code, out, err = invoke(
        monkeypatch,
        ["--config", str(tmp_path / "rag.toml"), "--data-dir", "elsewhere", "config", "show"],
        cwd=tmp_path,
    )
    assert code == EXIT_USAGE
    assert out == ""
    assert "迴路" in err
    assert not (tmp_path / "elsewhere").exists()


def test_userinfo_is_rejected_without_echoing_the_secret(monkeypatch, tmp_path):
    secret = "s3cret-token"
    write_config(
        tmp_path,
        f'[ollama]\nbase_url = "http://user:{secret}@127.0.0.1:11434"\n',
    )
    code, out, err = invoke(
        monkeypatch,
        ["--config", str(tmp_path / "rag.toml"), "config", "show"],
        cwd=tmp_path,
    )
    assert code == EXIT_USAGE
    assert secret not in out
    assert secret not in err
    assert "帳號或密碼" in err


def test_base_url_path_is_rejected(monkeypatch, tmp_path):
    write_config(tmp_path, '[ollama]\nbase_url = "http://127.0.0.1:11434/api"\n')
    code, _out, err = invoke(
        monkeypatch,
        ["--config", str(tmp_path / "rag.toml"), "config", "show"],
        cwd=tmp_path,
    )
    assert code == EXIT_USAGE
    assert "路徑" in err


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("http://127.0.0.1:11434/", "http://127.0.0.1:11434"),
        ("HTTP://LOCALHOST:11434", "http://localhost:11434"),
        ("http://[::1]:11434", "http://[::1]:11434"),
        ("https://127.0.0.1:11434", "https://127.0.0.1:11434"),
    ],
)
def test_loopback_urls_are_normalized(monkeypatch, tmp_path, url, expected):
    write_config(tmp_path, f'[ollama]\nbase_url = "{url}"\n')
    code, out, err = invoke(
        monkeypatch,
        ["--config", str(tmp_path / "rag.toml"), "config", "show"],
        cwd=tmp_path,
    )
    assert code == EXIT_OK, err
    assert shown(out)["ollama.base_url"] == expected


def test_verbose_before_or_after_subcommand(monkeypatch, tmp_path):
    config = write_config(tmp_path)
    before, _out, err = invoke(
        monkeypatch,
        ["--verbose", "--config", str(config), "config", "show"],
        cwd=tmp_path,
    )
    assert before == EXIT_OK, err
    assert logging.getLogger("ragcli").level == logging.DEBUG
    after, _out, err = invoke(
        monkeypatch,
        ["config", "show", "--config", str(config), "--verbose"],
        cwd=tmp_path,
    )
    assert after == EXIT_OK, err
    assert logging.getLogger("ragcli").level == logging.DEBUG
    middle, _out, err = invoke(
        monkeypatch,
        ["config", "--verbose", "--config", str(config), "show"],
        cwd=tmp_path,
    )
    assert middle == EXIT_OK, err
    assert logging.getLogger("ragcli").level == logging.DEBUG


def test_ingest_directory_exits_2(monkeypatch, tmp_path):
    write_config(tmp_path)
    folder = tmp_path / "docs"
    folder.mkdir()
    code, out, err = invoke(
        monkeypatch,
        ["--config", str(tmp_path / "rag.toml"), "ingest", str(folder)],
        cwd=tmp_path,
    )
    assert code == EXIT_USAGE
    assert out == ""
    assert "第一版要檔案路徑" in err
    assert not (tmp_path / "data").exists()


def test_ingest_directory_with_file_suffix_exits_2(monkeypatch, tmp_path):
    write_config(tmp_path)
    folder = tmp_path / "notes.pdf"
    folder.mkdir()
    code, _out, err = invoke(
        monkeypatch,
        ["ingest", str(folder)],
        cwd=tmp_path,
    )
    assert code == EXIT_USAGE
    assert "第一版要檔案路徑" in err


def test_ingest_missing_file_exits_4(monkeypatch, tmp_path):
    write_config(tmp_path)
    code, out, err = invoke(
        monkeypatch,
        ["ingest", str(tmp_path / "missing.pdf")],
        cwd=tmp_path,
    )
    assert code == EXIT_NOT_FOUND
    assert out == ""
    assert "不存在或不可讀" in err


def test_ingest_unsupported_suffix_exits_3(monkeypatch, tmp_path):
    write_config(tmp_path)
    document = tmp_path / "sheet.xlsx"
    document.write_bytes(b"")
    code, out, err = invoke(monkeypatch, ["ingest", str(document)], cwd=tmp_path)
    assert code == EXIT_UNSUPPORTED_TYPE
    assert out == ""
    assert ".xlsx" in err


def test_ingest_text_file_without_models_exits_0(monkeypatch, tmp_path):
    write_config(tmp_path, "[limits]\nmin_free_bytes = 1\n")
    document = tmp_path / "note.txt"
    document.write_text("你好", encoding="utf-8")
    code, out, err = invoke(
        monkeypatch,
        [
            "ingest",
            str(document),
            "--force",
            "--no-ocr",
            "--no-vision",
            "--no-embed",
            "--strict",
            "--max-pages",
            "10",
            "--max-bytes",
            "1000",
        ],
        cwd=tmp_path,
    )
    assert code == EXIT_OK
    assert "已入庫 note.txt" in out
    assert "你好" not in out
    assert "你好" not in err
    assert "尚未實作" not in err
    assert (tmp_path / "data").is_dir()


def test_ingest_unreadable_file_exits_4(monkeypatch, tmp_path):
    write_config(tmp_path)
    document = tmp_path / "secret.txt"
    document.write_text("secret", encoding="utf-8")
    document.chmod(0)
    try:
        code, _out, err = invoke(monkeypatch, ["ingest", str(document)], cwd=tmp_path)
    finally:
        document.chmod(0o644)
    assert code == EXIT_NOT_FOUND
    assert "不可讀" in err


def test_ingest_fifo_is_not_opened(monkeypatch, tmp_path):
    write_config(tmp_path)
    fifo = tmp_path / "note.txt"
    os.mkfifo(fifo)
    code, _out, err = invoke(monkeypatch, ["ingest", str(fifo)], cwd=tmp_path)
    assert code == EXIT_NOT_FOUND
    assert "不可讀" in err


def test_ingest_without_path_exits_2(monkeypatch, tmp_path):
    write_config(tmp_path)
    code, _out, err = invoke(monkeypatch, ["ingest"], cwd=tmp_path)
    assert code == EXIT_USAGE
    assert "第一版要檔案路徑" in err


def test_ingest_resume_without_path_exits_0(monkeypatch, tmp_path):
    write_config(tmp_path, "[limits]\nmin_free_bytes = 1\n")
    code, out, err = invoke(
        monkeypatch,
        ["ingest", "--resume", "--no-embed", "--no-vision"],
        cwd=tmp_path,
    )
    assert code == EXIT_OK
    assert "已繼續入庫" in out
    assert "尚未實作" not in err


def test_max_pages_zero_exits_2(monkeypatch, tmp_path):
    write_config(tmp_path)
    code, _out, _err = invoke(
        monkeypatch,
        ["ingest", "--max-pages", "0", "note.txt"],
        cwd=tmp_path,
    )
    assert code == EXIT_USAGE


def test_ask_without_question_exits_2(monkeypatch, tmp_path):
    write_config(tmp_path)
    code, out, err = invoke(monkeypatch, ["ask"], cwd=tmp_path, stdin="")
    assert code == EXIT_USAGE
    assert out == ""
    assert "缺少問題" in err


def test_ask_reads_stdin_without_echoing_it(monkeypatch, tmp_path):
    write_config(tmp_path)
    question = "這份文件對退款有什麼規定？"
    code, out, err = invoke(monkeypatch, ["ask"], cwd=tmp_path, stdin=question)
    assert code == EXIT_UNCLASSIFIED
    assert question not in out
    assert question not in err
    assert "尚未實作" in err


def test_ask_argument_does_not_read_stdin(monkeypatch, tmp_path):
    write_config(tmp_path)
    code, out, err = invoke(
        monkeypatch,
        ["ask", "--k", "8", "hello"],
        cwd=tmp_path,
        stdin=_StdinBoom(),
    )
    assert code == EXIT_UNCLASSIFIED
    assert "hello" not in out
    assert "hello" not in err


@pytest.mark.parametrize("k", ["0", "9", "x"])
def test_k_out_of_range_exits_2(monkeypatch, tmp_path, k):
    write_config(tmp_path)
    code, _out, _err = invoke(
        monkeypatch,
        ["ask", "--k", k, "hello"],
        cwd=tmp_path,
    )
    assert code == EXIT_USAGE


def test_models_pull_flags_conflict(monkeypatch, tmp_path):
    write_config(tmp_path)
    code, _out, err = invoke(
        monkeypatch,
        ["models", "pull", "--embed-only", "--generator-only"],
        cwd=tmp_path,
    )
    assert code == EXIT_USAGE
    assert "不能同時使用" in err
    assert not (tmp_path / "data").exists()


def test_delete_requires_exactly_one_selector(monkeypatch, tmp_path):
    write_config(tmp_path)
    neither, _out, err = invoke(monkeypatch, ["delete"], cwd=tmp_path)
    assert neither == EXIT_USAGE
    assert "兩者擇一" in err
    both, _out, err = invoke(
        monkeypatch,
        ["delete", "--path", "a", "--doc-id", "b"],
        cwd=tmp_path,
    )
    assert both == EXIT_USAGE
    assert "兩者擇一" in err


def test_use_requires_collection(monkeypatch, tmp_path):
    write_config(tmp_path)
    code, _out, err = invoke(monkeypatch, ["use"], cwd=tmp_path)
    assert code == EXIT_USAGE
    assert "--collection" in err


def test_dimensions_must_be_positive(monkeypatch, tmp_path):
    write_config(tmp_path)
    code, _out, _err = invoke(
        monkeypatch,
        ["reembed", "--dimensions", "0"],
        cwd=tmp_path,
    )
    assert code == EXIT_USAGE


def test_shell_modules_do_not_call_the_network_or_parsers():
    root = REPO / "src" / "ragcli"
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.glob("*.py")
        if path.name != "ollama.py"
    )
    for banned in (
        "import httpx",
        "import chromadb",
        "urllib.request",
        "urlopen",
        "subprocess",
        "os.system",
        "Popen",
        "import socket",
        "pymupdf",
        "ocrmac",
    ):
        assert banned not in text
