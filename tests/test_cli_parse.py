"""CLI parsing, exit codes, config paths, and the loopback URL gate."""

from __future__ import annotations

import ast
import io
import logging
import os
import sys
from pathlib import Path

import pytest

from ragcli.cli import SUPPORTED_SUFFIXES, _peek_option, _prepare_language, main
from ragcli.config import load_config, peek_ui_language
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
    RagError,
)
from ragcli.i18n import current_language, set_language

REPO = Path(__file__).resolve().parents[1]
EN_LANG_HELP = "Interface language, en or zh. Overrides ui.language"
ZH_LANG_HELP = "介面語言，en 或 zh。覆蓋 ui.language"
EN_DESCRIPTION = "Local mixed-language RAG. Documents stay on this Mac."
ZH_DESCRIPTION = "本機混合語言 RAG。文件留在這部 Mac。"
EN_INGEST_HELP = "Ingest one file"
ZH_INGEST_HELP = "把一個檔案入庫"
EN_FILE = "This version expects a file path, not a directory."
ZH_FILE = "第一版要檔案路徑，不接受資料夾。"
EN_LANG_INVALID = "Invalid language {value} for {source}. Use en or zh."
EN_CONFIG_MISSING = "Missing value for --config."
ZH_CONFIG_MISSING = "缺少 --config 的值。"


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


def flat(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture(autouse=True)
def reset_language():
    set_language("en")
    yield
    set_language("en")


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
    assert EN_LANG_HELP in flat(out)
    assert ZH_LANG_HELP not in flat(out)
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
    assert "[ui]" not in (REPO / "rag.toml").read_text(encoding="utf-8")
    assert values == {
        "config_path": values["config_path"],
        "ui.language": "en",
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
    assert "Cannot find the config file" in err


def test_invalid_toml(monkeypatch, tmp_path):
    write_config(tmp_path, "[\n")
    code, _out, err = invoke(
        monkeypatch,
        ["--config", str(tmp_path / "rag.toml"), "config", "show"],
        cwd=tmp_path,
    )
    assert code == EXIT_USAGE
    assert "not valid TOML" in err


def test_think_setting_is_rejected(monkeypatch, tmp_path):
    write_config(tmp_path, "[generate]\nthink = false\n")
    code, _out, err = invoke(
        monkeypatch,
        ["--config", str(tmp_path / "rag.toml"), "config", "show"],
        cwd=tmp_path,
    )
    assert code == EXIT_USAGE
    assert "think" in err
    assert "unknown field" in err


def test_bool_k_is_rejected(monkeypatch, tmp_path):
    write_config(tmp_path, "[retrieve]\nk = true\n")
    code, _out, err = invoke(
        monkeypatch,
        ["--config", str(tmp_path / "rag.toml"), "status"],
        cwd=tmp_path,
    )
    assert code == EXIT_USAGE
    assert "retrieve.k must be an integer." in err


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("[retrieve]\nk = 9\n", "retrieve.k must be from 1 to 8."),
        (
            '[ocr]\nrecognition_level = "fast"\n',
            "ocr.recognition_level can only be accurate in this version.",
        ),
        (
            "[ocr]\nlanguages = []\n",
            "ocr.languages must be a non-empty list of strings.",
        ),
        (
            "[retrieve]\nmax_cosine_distance = -0.1\n",
            "retrieve.max_cosine_distance cannot be negative.",
        ),
        (
            "[limits]\nmax_pages = 0\n",
            "limits.max_pages must be greater than or equal to 1.",
        ),
        (
            '[extra]\ncolor = "y"\n[theme]\ncolor = "x"\n',
            "Config file has an unknown section: extra, theme",
        ),
    ],
)
def test_config_value_errors_are_english(monkeypatch, tmp_path, body, expected):
    write_config(tmp_path, body)
    code, _out, err = invoke(
        monkeypatch,
        ["--config", str(tmp_path / "rag.toml"), "config", "show"],
        cwd=tmp_path,
    )
    assert code == EXIT_USAGE
    assert expected in err


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
    assert "loopback" in err
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
    assert "username or password" in err


def test_base_url_path_is_rejected(monkeypatch, tmp_path):
    write_config(tmp_path, '[ollama]\nbase_url = "http://127.0.0.1:11434/api"\n')
    code, _out, err = invoke(
        monkeypatch,
        ["--config", str(tmp_path / "rag.toml"), "config", "show"],
        cwd=tmp_path,
    )
    assert code == EXIT_USAGE
    assert "path or query" in err


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
    assert EN_FILE in err
    assert ZH_FILE not in err
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
    assert EN_FILE in err


def test_ingest_missing_file_exits_4(monkeypatch, tmp_path):
    write_config(tmp_path)
    code, out, err = invoke(
        monkeypatch,
        ["ingest", str(tmp_path / "missing.pdf")],
        cwd=tmp_path,
    )
    assert code == EXIT_NOT_FOUND
    assert out == ""
    assert "File does not exist or is unreadable:" in err


def test_ingest_unsupported_suffix_exits_3(monkeypatch, tmp_path):
    write_config(tmp_path)
    document = tmp_path / "sheet.xlsx"
    document.write_bytes(b"")
    code, out, err = invoke(monkeypatch, ["ingest", str(document)], cwd=tmp_path)
    assert code == EXIT_UNSUPPORTED_TYPE
    assert out == ""
    assert "Suffix is not supported in this version: .xlsx" in err


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
    assert "Ingested note.txt" in out
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
    assert "File does not exist or is unreadable:" in err


def test_ingest_fifo_is_not_opened(monkeypatch, tmp_path):
    write_config(tmp_path)
    fifo = tmp_path / "note.txt"
    os.mkfifo(fifo)
    code, _out, err = invoke(monkeypatch, ["ingest", str(fifo)], cwd=tmp_path)
    assert code == EXIT_NOT_FOUND
    assert "File does not exist or is unreadable:" in err


def test_ingest_without_path_exits_2(monkeypatch, tmp_path):
    write_config(tmp_path)
    code, _out, err = invoke(monkeypatch, ["ingest"], cwd=tmp_path)
    assert code == EXIT_USAGE
    assert EN_FILE in err


def test_ingest_directory_follows_lang_zh(monkeypatch, tmp_path):
    write_config(tmp_path)
    folder = tmp_path / "docs"
    folder.mkdir()
    code, out, err = invoke(
        monkeypatch,
        ["--lang", "zh", "--config", str(tmp_path / "rag.toml"), "ingest", str(folder)],
        cwd=tmp_path,
    )
    assert code == EXIT_USAGE
    assert out == ""
    assert ZH_FILE in err
    assert EN_FILE not in err


def test_ingest_directory_follows_ui_language_zh(monkeypatch, tmp_path):
    write_config(tmp_path, '[ui]\nlanguage = "zh"\n')
    folder = tmp_path / "docs"
    folder.mkdir()
    code, out, err = invoke(
        monkeypatch,
        ["ingest", str(folder)],
        cwd=tmp_path,
    )
    assert code == EXIT_USAGE
    assert out == ""
    assert ZH_FILE in err
    assert EN_FILE not in err


def test_ingest_resume_without_path_exits_0(monkeypatch, tmp_path):
    write_config(tmp_path, "[limits]\nmin_free_bytes = 1\n")
    code, out, err = invoke(
        monkeypatch,
        ["ingest", "--resume", "--no-embed", "--no-vision"],
        cwd=tmp_path,
    )
    assert code == EXIT_OK
    assert "Resumed ingest" in out
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
    assert "Missing question." in err


def test_ask_reads_stdin_without_echoing_it(monkeypatch, tmp_path):
    write_config(tmp_path, "[limits]\nmin_free_bytes = 1\n")
    question = "這份文件對退款有什麼規定？"
    code, out, err = invoke(monkeypatch, ["ask"], cwd=tmp_path, stdin=question)
    assert code == EXIT_NO_COLLECTION
    assert question not in out
    assert question not in err
    assert "尚未有作用中集合" in err


def test_ask_argument_does_not_read_stdin(monkeypatch, tmp_path):
    write_config(tmp_path, "[limits]\nmin_free_bytes = 1\n")
    code, out, err = invoke(
        monkeypatch,
        ["ask", "--k", "8", "hello"],
        cwd=tmp_path,
        stdin=_StdinBoom(),
    )
    assert code == EXIT_NO_COLLECTION
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
    assert "--embed-only and --generator-only cannot be used together." in err
    assert not (tmp_path / "data").exists()


def test_delete_requires_exactly_one_selector(monkeypatch, tmp_path):
    write_config(tmp_path)
    neither, _out, err = invoke(monkeypatch, ["delete"], cwd=tmp_path)
    assert neither == EXIT_USAGE
    assert "rag delete requires --path or --doc-id, exactly one." in err
    both, _out, err = invoke(
        monkeypatch,
        ["delete", "--path", "a", "--doc-id", "b"],
        cwd=tmp_path,
    )
    assert both == EXIT_USAGE
    assert "rag delete requires --path or --doc-id, exactly one." in err


def test_use_requires_collection(monkeypatch, tmp_path):
    write_config(tmp_path)
    code, _out, err = invoke(monkeypatch, ["use"], cwd=tmp_path)
    assert code == EXIT_USAGE
    assert "rag use requires --collection NAME." in err


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


def _invalid(value: str, source: str) -> str:
    return EN_LANG_INVALID.format(value=value, source=source)


def _help_lang_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "t"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "help.lang"
    )


def test_default_help_is_english(monkeypatch, tmp_path):
    code, out, err = invoke(monkeypatch, ["--help"], cwd=tmp_path)
    assert code == EXIT_OK
    assert err == ""
    text = flat(out)
    assert "usage:" in out
    assert EN_LANG_HELP in text
    assert ZH_LANG_HELP not in text
    assert EN_DESCRIPTION in text
    assert ZH_DESCRIPTION not in text
    assert EN_INGEST_HELP in text
    assert ZH_INGEST_HELP not in text
    assert "command" in text
    assert "指令" not in text


def test_lang_zh_help_is_traditional_chinese(monkeypatch, tmp_path):
    code, out, err = invoke(monkeypatch, ["--lang", "zh", "--help"], cwd=tmp_path)
    assert code == EXIT_OK
    assert err == ""
    text = flat(out)
    assert "usage:" in out
    assert ZH_LANG_HELP in text
    assert EN_LANG_HELP not in text
    assert ZH_DESCRIPTION in text
    assert EN_DESCRIPTION not in text
    assert ZH_INGEST_HELP in text
    assert EN_INGEST_HELP not in text
    assert "指令" in text
    assert "command" not in text


def test_lang_flag_overrides_ui_language(monkeypatch, tmp_path):
    write_config(tmp_path, '[ui]\nlanguage = "en"\n')
    code, out, err = invoke(
        monkeypatch, ["--lang", "zh", "config", "show"], cwd=tmp_path
    )
    assert code == EXIT_OK, err
    assert shown(out)["ui.language"] == "zh"

    write_config(tmp_path, '[ui]\nlanguage = "zh"\n')
    code, out, err = invoke(
        monkeypatch, ["--lang", "en", "config", "show"], cwd=tmp_path
    )
    assert code == EXIT_OK, err
    assert shown(out)["ui.language"] == "en"
    code, out, err = invoke(
        monkeypatch, ["config", "show", "--lang", "en"], cwd=tmp_path
    )
    assert code == EXIT_OK, err
    assert shown(out)["ui.language"] == "en"
    code, out, err = invoke(monkeypatch, ["config", "show"], cwd=tmp_path)
    assert code == EXIT_OK, err
    assert shown(out)["ui.language"] == "zh"


def test_valid_lang_skips_illegal_ui_language(monkeypatch, tmp_path):
    write_config(tmp_path, '[ui]\nlanguage = "fr"\n')
    code, out, err = invoke(monkeypatch, ["--lang", "zh", "--help"], cwd=tmp_path)
    assert code == EXIT_OK
    assert err == ""
    assert ZH_LANG_HELP in flat(out)
    assert "Invalid language" not in out

    code, out, err = invoke(
        monkeypatch, ["--lang", "zh", "config", "show"], cwd=tmp_path
    )
    assert code == EXIT_OK, err
    assert shown(out)["ui.language"] == "zh"
    code, out, err = invoke(
        monkeypatch, ["--lang", "en", "config", "show"], cwd=tmp_path
    )
    assert code == EXIT_OK, err
    assert shown(out)["ui.language"] == "en"


@pytest.mark.parametrize("value", ["fr", "zh-Hant", "zh ", "ZH", ""])
def test_illegal_file_language_fails_commands_not_help(monkeypatch, tmp_path, value):
    write_config(tmp_path, f'[ui]\nlanguage = "{value}"\n')
    code, out, err = invoke(monkeypatch, ["--help"], cwd=tmp_path)
    assert code == EXIT_OK
    assert err == ""
    assert EN_LANG_HELP in flat(out)
    assert ZH_LANG_HELP not in flat(out)
    code, out, err = invoke(monkeypatch, ["config", "show"], cwd=tmp_path)
    assert code == EXIT_USAGE
    assert out == ""
    assert err.strip() == _invalid(value, "ui.language")
    assert EN_DESCRIPTION not in flat(err)
    assert ZH_DESCRIPTION not in flat(err)


@pytest.mark.parametrize("value", ["fr", "zh-Hant", "zh ", "ZH", "en-US"])
def test_invalid_lang_flag_does_not_adopt_the_file(monkeypatch, tmp_path, value):
    write_config(tmp_path, '[ui]\nlanguage = "zh"\n')
    code, out, err = invoke(
        monkeypatch, ["--lang", value, "config", "show"], cwd=tmp_path
    )
    assert code == EXIT_USAGE
    assert out == ""
    assert err.strip() == _invalid(value, "--lang")
    assert "語言值無效" not in err
    assert EN_DESCRIPTION not in flat(out + err)
    assert ZH_DESCRIPTION not in flat(out + err)


def test_lang_equals_form(monkeypatch, tmp_path):
    write_config(tmp_path, '[ui]\nlanguage = "en"\n')
    code, out, err = invoke(monkeypatch, ["--lang=zh", "config", "show"], cwd=tmp_path)
    assert code == EXIT_OK, err
    assert shown(out)["ui.language"] == "zh"
    code, out, err = invoke(monkeypatch, ["--lang=fr", "config", "show"], cwd=tmp_path)
    assert code == EXIT_USAGE
    assert out == ""
    assert err.strip() == _invalid("fr", "--lang")


@pytest.mark.parametrize(
    "argv",
    [
        ["--lang", "--help"],
        ["--lang"],
        ["--lang="],
        ["--lang", "--config"],
        ["config", "show", "--lang"],
    ],
)
def test_lang_flag_without_a_value_does_not_print_help(monkeypatch, tmp_path, argv):
    write_config(tmp_path, '[ui]\nlanguage = "zh"\n')
    code, out, err = invoke(monkeypatch, argv, cwd=tmp_path)
    assert code == EXIT_USAGE
    assert err.strip() == _invalid("", "--lang")
    blob = flat(out + err)
    assert EN_DESCRIPTION not in blob
    assert ZH_DESCRIPTION not in blob
    assert "usage:" not in blob
    assert ZH_LANG_HELP not in blob


@pytest.mark.parametrize(
    ("body", "label"),
    [
        ("[ui]\nlanguage = 1\n", "int"),
        ("[ui]\nlanguage = true\n", "bool"),
        ("[ui]\nlanguage = []\n", "list"),
    ],
)
def test_non_string_ui_language_uses_the_type_name(monkeypatch, tmp_path, body, label):
    write_config(tmp_path, body)
    code, out, err = invoke(monkeypatch, ["--help"], cwd=tmp_path)
    assert code == EXIT_OK
    assert EN_LANG_HELP in flat(out)
    code, out, err = invoke(monkeypatch, ["config", "show"], cwd=tmp_path)
    assert code == EXIT_USAGE
    assert out == ""
    assert err.strip() == _invalid(label, "ui.language")


@pytest.mark.parametrize(
    "argv",
    [
        ["--config"],
        ["--config="],
        ["--config", "--help"],
        ["config", "show", "--config"],
    ],
)
def test_config_flag_without_a_value_does_not_print_help(monkeypatch, tmp_path, argv):
    code, out, err = invoke(monkeypatch, argv, cwd=tmp_path)
    assert code == EXIT_USAGE
    assert err.strip() == EN_CONFIG_MISSING
    blob = flat(out + err)
    assert EN_DESCRIPTION not in blob
    assert ZH_DESCRIPTION not in blob
    assert "usage:" not in blob


def test_last_lang_on_config_show_wins(monkeypatch, tmp_path):
    write_config(tmp_path, '[ui]\nlanguage = "en"\n')
    code, out, err = invoke(
        monkeypatch,
        ["--lang", "en", "config", "show", "--lang", "zh"],
        cwd=tmp_path,
    )
    assert code == EXIT_OK, err
    assert shown(out)["ui.language"] == "zh"


def test_last_lang_flag_wins_on_nested_help(monkeypatch, tmp_path):
    code, out, err = invoke(
        monkeypatch,
        ["--lang", "en", "models", "pull", "--lang", "zh", "--help"],
        cwd=tmp_path,
    )
    assert code == EXIT_OK
    assert err == ""
    text = flat(out)
    assert ZH_LANG_HELP in text
    assert EN_LANG_HELP not in text
    assert "usage:" in out


def test_lang_after_double_dash_is_not_a_flag(monkeypatch, tmp_path):
    code, out, err = invoke(
        monkeypatch,
        ["--lang", "en", "--help", "--", "--lang", "zh"],
        cwd=tmp_path,
    )
    assert code == EXIT_OK
    assert err == ""
    text = flat(out)
    assert EN_LANG_HELP in text
    assert ZH_LANG_HELP not in text


def test_lang_abbreviation_is_not_a_language(monkeypatch, tmp_path):
    write_config(tmp_path, '[ui]\nlanguage = "zh"\n')
    for argv in (["--l", "zh", "config", "show"], ["--language", "zh", "config", "show"]):
        code, out, err = invoke(monkeypatch, argv, cwd=tmp_path)
        assert code == EXIT_USAGE
        assert "Invalid language" not in err
        assert "語言值無效" not in err
        assert "ui.language" not in out


def test_parent_rag_toml_is_seen_before_help(monkeypatch, tmp_path):
    write_config(tmp_path, '[ui]\nlanguage = "zh"\n')
    child = tmp_path / "nested" / "dir"
    child.mkdir(parents=True)
    code, out, err = invoke(monkeypatch, ["--help"], cwd=child)
    assert code == EXIT_OK
    assert err == ""
    assert ZH_LANG_HELP in flat(out)
    assert EN_LANG_HELP not in flat(out)
    assert ZH_DESCRIPTION in flat(out)
    assert EN_DESCRIPTION not in flat(out)
    assert "usage:" in out
    code, out, err = invoke(monkeypatch, ["config", "show"], cwd=child)
    assert code == EXIT_OK, err
    values = shown(out)
    assert values["ui.language"] == "zh"
    assert values["config_path"] == str((tmp_path / "rag.toml").resolve())


def test_prescan_resolves_config_like_load_config(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    write_config(home, '[ui]\nlanguage = "zh"\n')
    parent = tmp_path / "parent"
    parent.mkdir()
    write_config(parent, '[ui]\nlanguage = "en"\n')
    child = parent / "child"
    child.mkdir()

    code, out, err = invoke(
        monkeypatch, ["--config", "~/rag.toml", "--help"], cwd=child
    )
    assert code == EXIT_OK, err
    assert ZH_LANG_HELP in flat(out)
    assert EN_LANG_HELP not in flat(out)

    code, out, err = invoke(
        monkeypatch, ["--config", "~/rag.toml", "config", "show"], cwd=child
    )
    assert code == EXIT_OK, err
    values = shown(out)
    assert values["ui.language"] == "zh"
    assert values["config_path"] == str((home / "rag.toml").resolve())

    code, out, err = invoke(
        monkeypatch,
        ["--config", "../../home/rag.toml", "--help"],
        cwd=child,
    )
    assert code == EXIT_OK, err
    assert ZH_LANG_HELP in flat(out)
    assert EN_LANG_HELP not in flat(out)


def test_ui_language_does_not_change_ocr_languages(monkeypatch, tmp_path):
    write_config(
        tmp_path,
        '[ui]\nlanguage = "en"\n[ocr]\nlanguages = ["ja-JP", "en-US"]\n',
    )
    code, out, err = invoke(monkeypatch, ["config", "show"], cwd=tmp_path)
    assert code == EXIT_OK, err
    assert shown(out)["ui.language"] == "en"
    assert shown(out)["ocr.languages"] == "ja-JP,en-US"
    config = load_config(tmp_path / "rag.toml", cwd=tmp_path)
    assert config.ui_language == "en"
    assert config.ocr_languages == ("ja-JP", "en-US")

    code, out, err = invoke(
        monkeypatch, ["--lang", "zh", "config", "show"], cwd=tmp_path
    )
    assert code == EXIT_OK, err
    assert shown(out)["ui.language"] == "zh"
    assert shown(out)["ocr.languages"] == "ja-JP,en-US"
    config = load_config(
        tmp_path / "rag.toml", cwd=tmp_path, language_from_flag="zh"
    )
    assert config.ui_language == "zh"
    assert config.ocr_languages == ("ja-JP", "en-US")


def test_empty_ui_table_defaults_to_english(monkeypatch, tmp_path):
    write_config(tmp_path, "[ui]\n")
    code, out, err = invoke(monkeypatch, ["config", "show"], cwd=tmp_path)
    assert code == EXIT_OK, err
    assert shown(out)["ui.language"] == "en"


def test_unknown_section_does_not_block_translated_help(monkeypatch, tmp_path):
    write_config(tmp_path, '[theme]\ncolor = "x"\n[ui]\nlanguage = "zh"\n')
    code, out, err = invoke(monkeypatch, ["--help"], cwd=tmp_path)
    assert code == EXIT_OK
    assert err == ""
    assert ZH_LANG_HELP in flat(out)
    code, out, err = invoke(monkeypatch, ["config", "show"], cwd=tmp_path)
    assert code == EXIT_USAGE
    assert "theme" in err
    assert "不明區段" in err


def test_unknown_ui_field_is_still_rejected(monkeypatch, tmp_path):
    write_config(tmp_path, '[ui]\nlanguage = "zh"\ntheme = "dark"\n')
    code, out, err = invoke(monkeypatch, ["--help"], cwd=tmp_path)
    assert code == EXIT_OK
    assert ZH_LANG_HELP in flat(out)
    code, out, err = invoke(
        monkeypatch, ["--lang", "en", "config", "show"], cwd=tmp_path
    )
    assert code == EXIT_USAGE
    assert "theme" in err
    assert "unknown field" in err


def test_verbose_logs_language_source_without_a_path(monkeypatch, tmp_path):
    write_config(tmp_path)
    code, out, err = invoke(
        monkeypatch, ["--verbose", "config", "show"], cwd=tmp_path
    )
    assert code == EXIT_OK, err
    assert err == "DEBUG Interface language en from default.\n"
    assert "rag.toml" not in err
    assert str(tmp_path) not in err

    write_config(tmp_path, '[ui]\nlanguage = "zh"\n')
    code, _out, err = invoke(
        monkeypatch, ["--verbose", "config", "show"], cwd=tmp_path
    )
    assert code == EXIT_OK, err
    assert err == "DEBUG 介面語言 zh，來源 config。\n"
    assert str(tmp_path) not in err

    code, _out, err = invoke(
        monkeypatch,
        ["--verbose", "--lang", "en", "config", "show"],
        cwd=tmp_path,
    )
    assert code == EXIT_OK, err
    assert err == "DEBUG Interface language en from flag.\n"
    assert str(tmp_path) not in err


def test_peek_ui_language_keeps_raw_values(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    write_config(tmp_path, '[ui]\nlanguage = "zh "\n')
    assert peek_ui_language() == "zh "
    write_config(tmp_path, "[ui]\nlanguage = 1\n")
    assert peek_ui_language() == 1
    write_config(tmp_path, "[ui]\n")
    assert peek_ui_language() is None
    write_config(tmp_path, "[\n")
    assert peek_ui_language() is None
    assert peek_ui_language(tmp_path / "missing.toml") is None
    directory = tmp_path / "not-a-file"
    directory.mkdir()
    assert peek_ui_language(directory) is None
    write_config(tmp_path, '[theme]\ncolor = "x"\n[ui]\nlanguage = "zh"\n')
    assert peek_ui_language() == "zh"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ("absent", None)),
        (["--lang=zh"], ("value", "zh")),
        (["--lang", "zh"], ("value", "zh")),
        (["--lang", "en", "models", "pull", "--lang", "zh"], ("value", "zh")),
        (["ingest", "--lang", "zh"], ("value", "zh")),
        (["--lang"], ("missing", None)),
        (["--lang", "--help"], ("missing", None)),
        (["--lang="], ("missing", None)),
        (["--lang", "en", "--", "--lang", "zh"], ("value", "en")),
        (["--", "--lang", "zh"], ("absent", None)),
        (["--l", "zh"], ("absent", None)),
        (["--language", "zh"], ("absent", None)),
        (["--langzh"], ("absent", None)),
    ],
)
def test_peek_option_table(argv, expected):
    assert _peek_option(argv, "--lang") == expected


def test_prepare_language_prescan_table(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls: list[Path | None] = []

    def fake(config_path: Path | None = None, *, cwd: Path | None = None) -> str:
        calls.append(config_path)
        return "zh"

    monkeypatch.setattr("ragcli.cli.peek_ui_language", fake)
    assert _prepare_language(["--lang=zh"]) == ("zh", "flag")
    assert _prepare_language(["ingest", "--lang", "zh"]) == ("zh", "flag")
    assert _prepare_language(["--lang", "en", "models", "pull", "--lang", "zh"]) == (
        "zh",
        "flag",
    )
    assert _prepare_language(["--lang", "en", "--", "--lang", "zh"]) == ("en", "flag")
    assert calls == []

    assert _prepare_language(["--l", "zh"]) == ("zh", "config")
    assert _prepare_language(["--", "--lang", "zh"]) == ("zh", "config")
    assert _prepare_language(["--language", "en"]) == ("zh", "config")
    assert calls == [None, None, None]

    calls.clear()
    assert _prepare_language(["--config", "other/rag.toml"]) == ("zh", "config")
    assert calls == [Path("other/rag.toml")]

    calls.clear()

    def boom(config_path: Path | None = None, *, cwd: Path | None = None) -> str:
        raise AssertionError("config was read")

    monkeypatch.setattr("ragcli.cli.peek_ui_language", boom)
    with pytest.raises(RagError) as caught:
        _prepare_language(["--lang", "zh", "--config"])
    assert caught.value.code == EXIT_USAGE
    assert caught.value.message == ZH_CONFIG_MISSING
    assert current_language() == "zh"
    set_language("zh")
    with pytest.raises(RagError) as caught:
        _prepare_language(["--lang", "fr", "--config"])
    assert caught.value.message == _invalid("fr", "--lang")
    assert current_language() == "en"


def test_prepare_language_ignores_illegal_peek_value(monkeypatch):
    monkeypatch.setattr("ragcli.cli.peek_ui_language", lambda *_a, **_k: "zh ")
    assert _prepare_language(["config", "show"]) == ("en", "default")
    monkeypatch.setattr("ragcli.cli.peek_ui_language", lambda *_a, **_k: None)
    assert _prepare_language(["config", "show"]) == ("en", "default")
    monkeypatch.setattr("ragcli.cli.peek_ui_language", lambda *_a, **_k: 1)
    assert _prepare_language(["config", "show"]) == ("en", "default")


def test_lang_help_call_lives_in_add_globals():
    tree = ast.parse((REPO / "src" / "ragcli" / "cli.py").read_text(encoding="utf-8"))
    owners: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if any(_help_lang_call(child) for child in ast.walk(node)):
                owners.append(node.name)
            continue
        if any(_help_lang_call(child) for child in ast.walk(node)):
            owners.append("<module>")
    assert owners == ["_add_globals"]


def _message_ids(node: ast.AST) -> list[str]:
    found = []
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "t"
            and child.args
            and isinstance(child.args[0], ast.Constant)
            and isinstance(child.args[0].value, str)
        ):
            found.append(child.args[0].value)
    return found


def test_help_text_is_translated_inside_parser_builders():
    source = (REPO / "src" / "ragcli" / "cli.py").read_text(encoding="utf-8")
    assert not any("\u4e00" <= char <= "\u9fff" for char in source)
    tree = ast.parse(source)
    module_ids: list[str] = []
    by_function: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            by_function[node.name] = _message_ids(node)
        else:
            module_ids.extend(_message_ids(node))
    assert module_ids == []
    help_owners = {
        name
        for name, ids in by_function.items()
        if any(message_id.startswith("help.") for message_id in ids)
    }
    assert help_owners == {"build_parser", "_add_globals"}
    help_calls = [
        message_id
        for ids in by_function.values()
        for message_id in ids
        if message_id.startswith("help.")
    ]
    assert len(help_calls) == 46
    assert help_calls.count("help.metavar.action") == 2
    assert "help.lang" in by_function["_add_globals"]


def test_ui_language_log_uses_percent_s():
    tree = ast.parse((REPO / "src" / "ragcli" / "cli.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run"
    )
    debug_calls = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "debug":
            continue
        debug_calls.append(node)
    assert len(debug_calls) == 1
    call = debug_calls[0]
    assert len(call.args) == 2
    assert isinstance(call.args[0], ast.Constant) and call.args[0].value == "%s"
    rendered = call.args[1]
    assert isinstance(rendered, ast.Call)
    assert isinstance(rendered.func, ast.Name) and rendered.func.id == "t"
    assert rendered.args[0].value == "log.ui_language"
