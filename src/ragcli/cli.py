"""Argument parsing and exit codes for the rag command.

`rag ingest` runs the seven ingest stages and writes the corpus.
`rag models pull` and `rag models unload` talk to loopback Ollama.
`rag reembed` rebuilds or switches the vector index. `rag use` moves the
active collection. `rag delete` removes one document. `rag status` reports
the local corpus and whether Ollama answers. `rag ask` answers one question
from the active collection and does not embed pending chunks.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from ragcli.ask import run_ask
from ragcli.config import Config, format_config, load_config, peek_ui_language
from ragcli.errors import (
    EXIT_NOT_FOUND,
    EXIT_OK,
    EXIT_UNCLASSIFIED,
    EXIT_UNSUPPORTED_TYPE,
    EXIT_USAGE,
    RagError,
)
from ragcli.i18n import set_language, t
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
        description=t("help.description"),
        allow_abbrev=False,
    )
    _add_globals(parser, root=True)
    commands = parser.add_subparsers(
        dest="command", required=True, metavar=t("help.metavar.command")
    )

    ingest = _add_parser(commands, "ingest", help=t("help.ingest"))
    _add_globals(ingest, root=False)
    ingest.add_argument("file", nargs="?", help=t("help.ingest.file"))
    ingest.add_argument("--force", action="store_true", help=t("help.ingest.force"))
    ingest.add_argument(
        "--resume", action="store_true", help=t("help.ingest.resume")
    )
    ingest.add_argument("--no-ocr", action="store_true", help=t("help.ingest.no_ocr"))
    ingest.add_argument(
        "--no-vision", action="store_true", help=t("help.ingest.no_vision")
    )
    ingest.add_argument(
        "--no-embed", action="store_true", help=t("help.ingest.no_embed")
    )
    ingest.add_argument(
        "--strict", action="store_true", help=t("help.ingest.strict")
    )
    ingest.add_argument(
        "--max-pages", type=_positive_int, help=t("help.ingest.max_pages")
    )
    ingest.add_argument(
        "--max-bytes", type=_positive_int, help=t("help.ingest.max_bytes")
    )
    ingest.set_defaults(func=_cmd_ingest)

    ask = _add_parser(commands, "ask", help=t("help.ask"))
    _add_globals(ask, root=False)
    ask.add_argument("question", nargs="?", help=t("help.ask.question"))
    ask.add_argument("--k", type=_k_value, help=t("help.ask.k"))
    ask.add_argument("--model", help=t("help.ask.model"))
    ask.add_argument(
        "--no-fallback", action="store_true", help=t("help.ask.no_fallback")
    )
    ask.add_argument(
        "--show-chunks", action="store_true", help=t("help.ask.show_chunks")
    )
    ask.add_argument("--json", action="store_true", help=t("help.ask.json"))
    ask.set_defaults(func=_cmd_ask)

    reembed = _add_parser(commands, "reembed", help=t("help.reembed"))
    _add_globals(reembed, root=False)
    reembed.add_argument("--model", help=t("help.reembed.model"))
    reembed.add_argument(
        "--dimensions", type=_positive_int, help=t("help.reembed.dimensions")
    )
    reembed.add_argument(
        "--native-dimensions",
        type=_positive_int,
        help=t("help.reembed.native_dimensions"),
    )
    reembed.add_argument(
        "--no-activate", action="store_true", help=t("help.reembed.no_activate")
    )
    reembed.add_argument(
        "--drop-old", action="store_true", help=t("help.reembed.drop_old")
    )
    reembed.set_defaults(func=_cmd_reembed)

    use = _add_parser(commands, "use", help=t("help.use"))
    _add_globals(use, root=False)
    use.add_argument("--collection", help=t("help.use.collection"))
    use.set_defaults(func=_cmd_use)

    delete = _add_parser(commands, "delete", help=t("help.delete"))
    _add_globals(delete, root=False)
    delete.add_argument("--path", help=t("help.delete.path"))
    delete.add_argument("--doc-id", help=t("help.delete.doc_id"))
    delete.set_defaults(func=_cmd_delete)

    status = _add_parser(commands, "status", help=t("help.status"))
    _add_globals(status, root=False)
    status.set_defaults(func=_cmd_status)

    models = _add_parser(commands, "models", help=t("help.models"))
    _add_globals(models, root=False)
    model_commands = models.add_subparsers(
        dest="models_command", required=True, metavar=t("help.metavar.action")
    )
    pull = _add_parser(model_commands, "pull", help=t("help.models.pull"))
    _add_globals(pull, root=False)
    pull.add_argument(
        "--embed-only", action="store_true", help=t("help.models.pull.embed_only")
    )
    pull.add_argument(
        "--generator-only",
        action="store_true",
        help=t("help.models.pull.generator_only"),
    )
    pull.add_argument(
        "--no-fallback-model",
        action="store_true",
        help=t("help.models.pull.no_fallback_model"),
    )
    pull.set_defaults(func=_cmd_models_pull)
    unload = _add_parser(model_commands, "unload", help=t("help.models.unload"))
    _add_globals(unload, root=False)
    unload.set_defaults(func=_cmd_models_unload)

    config = _add_parser(commands, "config", help=t("help.config"))
    _add_globals(config, root=False)
    config_commands = config.add_subparsers(
        dest="config_command", required=True, metavar=t("help.metavar.action")
    )
    show = _add_parser(config_commands, "show", help=t("help.config.show"))
    _add_globals(show, root=False)
    show.set_defaults(func=_cmd_config_show)
    return parser


def _run(argv: list[str] | None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    language, source = _prepare_language(args_list)
    set_language(language)
    parser = build_parser()
    args = parser.parse_args(args_list)
    configure_logging(bool(getattr(args, "verbose", False)))
    logging.getLogger("ragcli").debug(
        "%s",
        t("log.ui_language", language=language, source=source),
    )
    config = load_config(
        getattr(args, "config", None),
        getattr(args, "data_dir", None),
        language_from_flag=language if source == "flag" else None,
    )
    func = getattr(args, "func", None)
    if func is None:
        parser.error(t("usage.subcommand_missing"))
    return func(args, config)


def _prepare_language(argv: list[str]) -> tuple[str, str]:
    """Choose `en` or `zh` before the parser is built.

    Returns the language and a source of `flag`, `config`, or `default`.
    A missing or illegal `--lang` is an English usage error and does not
    read `[ui] language`. A missing `--config` value stops here too. A
    valid `--lang` is applied first, so that error follows the flag.
    """
    lang_state, lang_value = _peek_option(argv, "--lang")
    config_state, config_value = _peek_option(argv, "--config")
    chosen: str | None = None
    if lang_state != "absent":
        chosen = "" if lang_value is None else lang_value
        if chosen not in {"en", "zh"}:
            set_language("en")
            raise RagError(
                EXIT_USAGE,
                t("usage.language_invalid", value=chosen, source="--lang"),
            )
    if config_state == "missing":
        set_language("en" if chosen is None else chosen)
        raise RagError(EXIT_USAGE, t("usage.config_missing"))
    if chosen is not None:
        return chosen, "flag"
    config_path = None if config_value is None else Path(config_value)
    peeked = peek_ui_language(config_path)
    if isinstance(peeked, str) and peeked in {"en", "zh"}:
        return peeked, "config"
    return "en", "default"


def _peek_option(argv: list[str], name: str) -> tuple[str, str | None]:
    """Return the last exact `name` in `argv`.

    The state is `absent`, `missing`, or `value`. A bare `--` stops the
    scan. `name=text` is a value, and an empty `name=` is missing. The
    following token is a value only when it exists and does not start
    with `-`. Abbreviations are not flags.
    """
    state = "absent"
    value: str | None = None
    index = 0
    limit = len(argv)
    prefix = f"{name}="
    while index < limit:
        token = argv[index]
        if token == "--":
            break
        if token == name:
            nxt = index + 1
            if nxt >= limit or argv[nxt].startswith("-"):
                state = "missing"
                value = None
                index += 1
                continue
            state = "value"
            value = argv[nxt]
            index += 2
            continue
        if token.startswith(prefix):
            text = token[len(prefix) :]
            if text == "":
                state = "missing"
                value = None
            else:
                state = "value"
                value = text
            index += 1
            continue
        index += 1
    return state, value


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
        raise RagError(EXIT_USAGE, t("usage.file_not_directory"))
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
        raise RagError(EXIT_USAGE, t("usage.file_not_directory"))
    if not path.is_file() or not os.access(path, os.R_OK):
        raise RagError(EXIT_NOT_FOUND, t("io.file_unreadable", path=path))
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise RagError(
            EXIT_UNSUPPORTED_TYPE,
            t(
                "usage.suffix_unsupported",
                label=suffix if suffix else t("usage.no_suffix"),
            ),
        )


def _cmd_ask(args: argparse.Namespace, config: Config) -> int:
    return run_ask(
        config,
        _question_text(args.question),
        k=args.k,
        model=args.model,
        allow_fallback=not bool(args.no_fallback),
        show_chunks=bool(args.show_chunks),
        as_json=bool(args.json),
        lock_timeout=_lock_timeout(args),
    )


def _question_text(question: str | None) -> str:
    if question is not None:
        text = question.strip()
        if text == "":
            raise RagError(EXIT_USAGE, t("ask.question_missing"))
        return text
    if sys.stdin is None or sys.stdin.isatty():
        raise RagError(EXIT_USAGE, t("ask.question_missing"))
    text = sys.stdin.read().strip()
    if text == "":
        raise RagError(EXIT_USAGE, t("ask.question_missing"))
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
        raise RagError(EXIT_USAGE, t("usage.use_needs_collection"))
    return run_use(config, args.collection)


def _cmd_delete(args: argparse.Namespace, config: Config) -> int:
    if (args.path is None) == (args.doc_id is None):
        raise RagError(EXIT_USAGE, t("usage.delete_needs_one"))
    path = None if args.path is None else Path(args.path)
    return run_delete(config, path=path, doc_id=args.doc_id)


def _cmd_status(_args: argparse.Namespace, config: Config) -> int:
    return run_status(config)


def _cmd_models_pull(args: argparse.Namespace, config: Config) -> int:
    if args.embed_only and args.generator_only:
        raise RagError(EXIT_USAGE, t("usage.flags_exclusive"))
    names = model_pull_list(
        config,
        embed_only=args.embed_only,
        generator_only=args.generator_only,
        no_fallback_model=args.no_fallback_model,
    )
    with ollama_lock(config.data_dir, timeout=_lock_timeout(args)):
        pull_models(OllamaClient(config.ollama_base_url), names)
    for name in names:
        print(t("status.pulled", name=name))
    return EXIT_OK


def _cmd_models_unload(args: argparse.Namespace, config: Config) -> int:
    with ollama_lock(config.data_dir, timeout=_lock_timeout(args)):
        names = OllamaClient(config.ollama_base_url).unload_all()
    if not names:
        print(t("status.no_loaded_models"))
    for name in names:
        print(t("status.unloaded", name=name))
    return EXIT_OK


def _lock_timeout(args: argparse.Namespace) -> float | None:
    value = getattr(args, "lock_timeout", None)
    if value is None:
        return None
    return float(value)


def _unimplemented(name: str):
    def command(_args: argparse.Namespace, _config: Config) -> int:
        raise RagError(EXIT_UNCLASSIFIED, t("usage.not_implemented", name=name))

    return command


def _add_globals(parser: argparse.ArgumentParser, *, root: bool) -> None:
    absent = None if root else argparse.SUPPRESS
    verbose_default = False if root else argparse.SUPPRESS
    parser.add_argument(
        "--config",
        type=Path,
        default=absent,
        help=t("help.flag.config"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=absent,
        help=t("help.flag.data_dir"),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=verbose_default,
        help=t("help.flag.verbose"),
    )
    parser.add_argument(
        "--lock-timeout",
        type=_non_negative_int,
        default=absent,
        help=t("help.flag.lock_timeout"),
    )
    parser.add_argument(
        "--lang",
        choices=("en", "zh"),
        default=absent,
        help=t("help.lang"),
    )


def _add_parser(subparsers, name: str, **kwargs) -> argparse.ArgumentParser:
    kwargs.setdefault("allow_abbrev", False)
    return subparsers.add_parser(name, **kwargs)


def _non_negative_int(raw: str) -> int:
    try:
        value = int(raw, 10)
    except ValueError:
        raise argparse.ArgumentTypeError(t("usage.non_negative_int")) from None
    if value < 0:
        raise argparse.ArgumentTypeError(t("usage.non_negative_int"))
    return value


def _positive_int(raw: str) -> int:
    try:
        value = int(raw, 10)
    except ValueError:
        raise argparse.ArgumentTypeError(t("usage.positive_int")) from None
    if value <= 0:
        raise argparse.ArgumentTypeError(t("usage.positive_int"))
    return value


def _k_value(raw: str) -> int:
    try:
        value = int(raw, 10)
    except ValueError:
        raise argparse.ArgumentTypeError(t("usage.k_range")) from None
    if value < 1 or value > 8:
        raise argparse.ArgumentTypeError(t("usage.k_range"))
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
