"""Argument parsing and exit codes for the rag command.

`rag ingest` runs the seven ingest stages and writes the corpus.
`rag models pull` and `rag models unload` talk to loopback Ollama.
`rag reembed` rebuilds or switches the vector index. `rag use` moves the
active collection. `rag delete` removes one document. `rag status` reports
the local corpus and whether Ollama answers. `rag ask` validates the
invocation, then exits 1 until a later change implements it.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from ragcli.config import Config, format_config, load_config
from ragcli.errors import (
    EXIT_NOT_FOUND,
    EXIT_OK,
    EXIT_UNCLASSIFIED,
    EXIT_UNSUPPORTED_TYPE,
    EXIT_USAGE,
    RagError,
)
from ragcli.ingest import run_ingest
from ragcli.models_lock import ollama_lock
from ragcli.ollama import OllamaClient, model_pull_list, pull_models
from ragcli.reembed import run_delete, run_reembed, run_status, run_use

SUPPORTED_SUFFIXES = frozenset(
    {
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
)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    try:
        return _run(argv)
    except RagError as exc:
        print(exc.message, file=sys.stderr)
        return exc.code
    except SystemExit as exc:
        return _code_from_system_exit(exc)


def console_main() -> None:
    """Setuptools entry point. Turns the integer from main into a process status."""
    sys.exit(main())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="本機混合語言 RAG。文件留在這部 Mac。",
        allow_abbrev=False,
    )
    _add_globals(parser, root=True)
    commands = parser.add_subparsers(dest="command", required=True, metavar="指令")

    ingest = _add_parser(commands, "ingest", help="把一個檔案入庫")
    _add_globals(ingest, root=False)
    ingest.add_argument("file", nargs="?", help="檔案路徑。第一版不接受資料夾")
    ingest.add_argument("--force", action="store_true", help="重新處理這個檔案")
    ingest.add_argument(
        "--resume", action="store_true", help="繼續未完成的入庫，可以不帶路徑"
    )
    ingest.add_argument("--no-ocr", action="store_true", help="不做 OCR")
    ingest.add_argument("--no-vision", action="store_true", help="不做圖像描述")
    ingest.add_argument("--no-embed", action="store_true", help="不嵌入")
    ingest.add_argument(
        "--strict", action="store_true", help="任何圖像或區塊失敗都算失敗"
    )
    ingest.add_argument("--max-pages", type=_positive_int, help="頁數上限")
    ingest.add_argument("--max-bytes", type=_positive_int, help="檔案位元組上限")
    ingest.set_defaults(func=_cmd_ingest)

    ask = _add_parser(commands, "ask", help="問已入庫的文件")
    _add_globals(ask, root=False)
    ask.add_argument("question", nargs="?", help="問題。省略時由標準輸入讀取")
    ask.add_argument("--k", type=_k_value, help="取回區塊數，1 到 8")
    ask.add_argument("--model", help="優先使用的生成模型標籤")
    ask.add_argument(
        "--no-fallback", action="store_true", help="記憶體不足時不要改用 4B"
    )
    ask.add_argument("--show-chunks", action="store_true", help="在 stderr 印出命中區塊")
    ask.add_argument("--json", action="store_true", help="以 JSON 輸出")
    ask.set_defaults(func=_cmd_ask)

    reembed = _add_parser(commands, "reembed", help="按模型或維度重建向量")
    _add_globals(reembed, root=False)
    reembed.add_argument("--model", help="嵌入模型標籤")
    reembed.add_argument("--dimensions", type=_positive_int, help="目標維度")
    reembed.add_argument(
        "--native-dimensions",
        type=_positive_int,
        help="未知嵌入模型的原生寬度",
    )
    reembed.add_argument(
        "--no-activate", action="store_true", help="只建集合，不改作用中指標"
    )
    reembed.add_argument("--drop-old", action="store_true", help="刪除上一個集合")
    reembed.set_defaults(func=_cmd_reembed)

    use = _add_parser(commands, "use", help="改作用中集合")
    _add_globals(use, root=False)
    use.add_argument("--collection", help="作用中集合名稱")
    use.set_defaults(func=_cmd_use)

    delete = _add_parser(commands, "delete", help="刪除一份文件的區塊同向量")
    _add_globals(delete, root=False)
    delete.add_argument("--path", help="要刪除的原始檔路徑")
    delete.add_argument("--doc-id", help="要刪除的文件 id")
    delete.set_defaults(func=_cmd_delete)

    status = _add_parser(commands, "status", help="顯示本機狀態")
    _add_globals(status, root=False)
    status.set_defaults(func=_cmd_status)

    models = _add_parser(commands, "models", help="拉取或卸載 Ollama 模型")
    _add_globals(models, root=False)
    model_commands = models.add_subparsers(
        dest="models_command", required=True, metavar="動作"
    )
    pull = _add_parser(model_commands, "pull", help="按預設順序拉取模型")
    _add_globals(pull, root=False)
    pull.add_argument("--embed-only", action="store_true", help="只拉取嵌入模型")
    pull.add_argument("--generator-only", action="store_true", help="只拉取生成模型")
    pull.add_argument(
        "--no-fallback-model",
        action="store_true",
        help="不拉取 4B，未見 4B 前不自動後備",
    )
    pull.set_defaults(func=_cmd_models_pull)
    unload = _add_parser(model_commands, "unload", help="卸載已載入的模型")
    _add_globals(unload, root=False)
    unload.set_defaults(func=_cmd_models_unload)

    config = _add_parser(commands, "config", help="查看設定")
    _add_globals(config, root=False)
    config_commands = config.add_subparsers(
        dest="config_command", required=True, metavar="動作"
    )
    show = _add_parser(config_commands, "show", help="印出解析後的設定")
    _add_globals(show, root=False)
    show.set_defaults(func=_cmd_config_show)
    return parser


def _run(argv: list[str] | None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(bool(getattr(args, "verbose", False)))
    config = load_config(getattr(args, "config", None), getattr(args, "data_dir", None))
    func = getattr(args, "func", None)
    if func is None:
        parser.error("缺少子命令")
    return func(args, config)


def configure_logging(verbose: bool) -> None:
    """Send ragcli logs to stderr. Debug stays off unless --verbose is set."""
    logger = logging.getLogger("ragcli")
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False


def _cmd_config_show(_args: argparse.Namespace, config: Config) -> int:
    sys.stdout.write(format_config(config))
    return EXIT_OK


def _cmd_ingest(args: argparse.Namespace, config: Config) -> int:
    if args.file is None and not args.resume:
        raise RagError(EXIT_USAGE, "第一版要檔案路徑，不接受資料夾。")
    if args.file is not None:
        _require_ingest_file(args.file)
    return run_ingest(
        config,
        path=None if args.file is None else Path(args.file),
        force=bool(args.force),
        resume=bool(args.resume),
        no_ocr=bool(args.no_ocr),
        no_vision=bool(args.no_vision),
        no_embed=bool(args.no_embed),
        strict=bool(args.strict),
        max_pages=args.max_pages,
        max_bytes=args.max_bytes,
        lock_timeout=_lock_timeout(args),
    )


def _require_ingest_file(raw: str) -> None:
    path = Path(raw)
    if path.is_dir():
        raise RagError(EXIT_USAGE, "第一版要檔案路徑，不接受資料夾。")
    if not path.is_file() or not os.access(path, os.R_OK):
        raise RagError(EXIT_NOT_FOUND, f"檔案不存在或不可讀：{path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        label = suffix if suffix else "（沒有副檔名）"
        raise RagError(EXIT_UNSUPPORTED_TYPE, f"副檔名不在第一版：{label}")


def _cmd_ask(args: argparse.Namespace, _config: Config) -> int:
    _question_text(args.question)
    raise RagError(EXIT_UNCLASSIFIED, "rag ask 尚未實作。")


def _question_text(question: str | None) -> str:
    if question is not None and question.strip() != "":
        return question
    if sys.stdin is None or sys.stdin.isatty():
        raise RagError(EXIT_USAGE, "缺少問題。")
    text = sys.stdin.read()
    if text.strip() == "":
        raise RagError(EXIT_USAGE, "缺少問題。")
    return text


def _cmd_reembed(args: argparse.Namespace, config: Config) -> int:
    return run_reembed(
        config,
        model=args.model,
        dimensions=args.dimensions,
        native_dimensions=args.native_dimensions,
        activate=not args.no_activate,
        drop_old=bool(args.drop_old),
        lock_timeout=_lock_timeout(args),
    )


def _cmd_use(args: argparse.Namespace, config: Config) -> int:
    if not args.collection:
        raise RagError(EXIT_USAGE, "rag use 要 --collection NAME。")
    return run_use(config, args.collection)


def _cmd_delete(args: argparse.Namespace, config: Config) -> int:
    if (args.path is None) == (args.doc_id is None):
        raise RagError(EXIT_USAGE, "rag delete 要 --path 或 --doc-id，兩者擇一。")
    path = None if args.path is None else Path(args.path)
    return run_delete(config, path=path, doc_id=args.doc_id)


def _cmd_status(_args: argparse.Namespace, config: Config) -> int:
    return run_status(config)


def _cmd_models_pull(args: argparse.Namespace, config: Config) -> int:
    if args.embed_only and args.generator_only:
        raise RagError(EXIT_USAGE, "--embed-only 同 --generator-only 不能同時使用。")
    names = model_pull_list(
        config,
        embed_only=args.embed_only,
        generator_only=args.generator_only,
        no_fallback_model=args.no_fallback_model,
    )
    with ollama_lock(config.data_dir, timeout=_lock_timeout(args)):
        pull_models(OllamaClient(config.ollama_base_url), names)
    for name in names:
        print(f"已拉取 {name}")
    return EXIT_OK


def _cmd_models_unload(args: argparse.Namespace, config: Config) -> int:
    with ollama_lock(config.data_dir, timeout=_lock_timeout(args)):
        names = OllamaClient(config.ollama_base_url).unload_all()
    if not names:
        print("沒有已載入的模型。")
    for name in names:
        print(f"已卸載 {name}")
    return EXIT_OK


def _lock_timeout(args: argparse.Namespace) -> float | None:
    value = getattr(args, "lock_timeout", None)
    if value is None:
        return None
    return float(value)


def _unimplemented(name: str):
    def command(_args: argparse.Namespace, _config: Config) -> int:
        raise RagError(EXIT_UNCLASSIFIED, f"{name} 尚未實作。")

    return command


def _add_globals(parser: argparse.ArgumentParser, *, root: bool) -> None:
    absent = None if root else argparse.SUPPRESS
    verbose_default = False if root else argparse.SUPPRESS
    parser.add_argument(
        "--config",
        type=Path,
        default=absent,
        help="設定檔。預設由目前目錄向上尋找 rag.toml",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=absent,
        help="覆蓋設定檔的 data_dir。相對路徑由目前目錄解析",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=verbose_default,
        help="日誌改為 DEBUG。DEBUG 不印文件正文",
    )
    parser.add_argument(
        "--lock-timeout",
        type=_non_negative_int,
        default=absent,
        help="等待 Ollama 鎖的秒數。省略則一直等。0 表示不等",
    )


def _add_parser(subparsers, name: str, **kwargs) -> argparse.ArgumentParser:
    kwargs.setdefault("allow_abbrev", False)
    return subparsers.add_parser(name, **kwargs)


def _non_negative_int(raw: str) -> int:
    try:
        value = int(raw, 10)
    except ValueError:
        raise argparse.ArgumentTypeError("必須是零或正整數") from None
    if value < 0:
        raise argparse.ArgumentTypeError("必須是零或正整數")
    return value


def _positive_int(raw: str) -> int:
    try:
        value = int(raw, 10)
    except ValueError:
        raise argparse.ArgumentTypeError("必須是正整數") from None
    if value <= 0:
        raise argparse.ArgumentTypeError("必須是正整數")
    return value


def _k_value(raw: str) -> int:
    try:
        value = int(raw, 10)
    except ValueError:
        raise argparse.ArgumentTypeError("--k 必須是 1 到 8 的整數") from None
    if value < 1 or value > 8:
        raise argparse.ArgumentTypeError("--k 必須是 1 到 8 的整數")
    return value


def _code_from_system_exit(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return EXIT_OK
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return EXIT_USAGE


if __name__ == "__main__":
    console_main()
