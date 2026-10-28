"""Load rag.toml and resolve paths.

`data_dir` in the file is relative to the directory that contains that file.
`--data-dir` is relative to the current working directory. Ollama URLs must
use the hosts 127.0.0.1, localhost, or ::1. There is no setting that turns
thinking on. `[ui] language` is optional and is only `en` or `zh`.
"""

from __future__ import annotations

import tomllib
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from ragcli.errors import EXIT_USAGE, RagError
from ragcli.i18n import t

_UI_LANGUAGES = frozenset({"en", "zh"})

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

_SECTIONS: dict[str, frozenset[str]] = {
    "paths": frozenset({"data_dir"}),
    "ollama": frozenset({"base_url"}),
    "embed": frozenset(
        {"model", "native_dimensions", "dimensions", "num_ctx", "batch_size"}
    ),
    "generate": frozenset(
        {
            "model",
            "fallback_model",
            "num_ctx",
            "vision_num_ctx",
            "num_predict",
            "vision_num_predict",
            "temperature",
            "vision_temperature",
        }
    ),
    "retrieve": frozenset({"k", "max_cosine_distance", "max_question_tokens"}),
    "chunk": frozenset({"target_tokens", "max_tokens", "overlap_tokens"}),
    "ocr": frozenset(
        {"languages", "recognition_level", "confidence_threshold", "rich_tokens"}
    ),
    "limits": frozenset(
        {
            "max_pages",
            "max_images",
            "max_bytes",
            "min_free_bytes",
            "max_pixels",
            "nine_b_min_bytes",
            "four_b_min_bytes",
        }
    ),
    "ui": frozenset({"language"}),
}


@dataclass(frozen=True)
class Config:
    config_path: Path
    ui_language: str
    data_dir: Path
    ollama_base_url: str
    embed_model: str
    embed_native_dimensions: int
    embed_dimensions: int
    embed_num_ctx: int
    embed_batch_size: int
    generate_model: str
    generate_fallback_model: str
    generate_num_ctx: int
    vision_num_ctx: int
    num_predict: int
    vision_num_predict: int
    temperature: float
    vision_temperature: float
    k: int
    max_cosine_distance: float
    max_question_tokens: int
    chunk_target_tokens: int
    chunk_max_tokens: int
    chunk_overlap_tokens: int
    ocr_languages: tuple[str, ...]
    ocr_recognition_level: str
    ocr_confidence_threshold: float
    ocr_rich_tokens: int
    max_pages: int
    max_images: int
    max_bytes: int
    min_free_bytes: int
    max_pixels: int
    nine_b_min_bytes: int
    four_b_min_bytes: int


def load_config(
    config_path: Path | None = None,
    data_dir: Path | None = None,
    *,
    cwd: Path | None = None,
    language_from_flag: str | None = None,
) -> Config:
    """Load and validate configuration. Does not create directories.

    `language_from_flag` is the language `--lang` already chose. When it is
    `en` or `zh`, an illegal `[ui] language` is left unused. This function
    does not change the active catalog language.
    """
    work_dir = (cwd or Path.cwd()).resolve()
    path = _locate_config(config_path, work_dir)
    raw = _read_toml(path)
    tables = _sections(raw)
    paths = tables["paths"]
    ollama = tables["ollama"]
    embed = tables["embed"]
    generate = tables["generate"]
    retrieve = tables["retrieve"]
    chunk = tables["chunk"]
    ocr = tables["ocr"]
    limits = tables["limits"]
    ui = tables["ui"]

    if data_dir is None:
        data_raw = _string(paths, "data_dir", "data", "paths.data_dir")
        data_base = path.parent
    else:
        data_raw = str(data_dir)
        data_base = work_dir

    k = _int(retrieve, "k", 5, "retrieve.k", minimum=1)
    if k > 8:
        raise RagError(EXIT_USAGE, t("config.retrieve_k"))

    level = _string(ocr, "recognition_level", "accurate", "ocr.recognition_level")
    if level != "accurate":
        raise RagError(EXIT_USAGE, t("ocr.recognition_level_invalid"))

    return Config(
        config_path=path,
        ui_language=_adopt_ui_language(ui, language_from_flag),
        data_dir=_resolve_dir(data_raw, data_base),
        ollama_base_url=normalize_base_url(
            _string(
                ollama,
                "base_url",
                "http://127.0.0.1:11434",
                "ollama.base_url",
            )
        ),
        embed_model=_string(embed, "model", "qwen3-embedding:0.6b", "embed.model"),
        embed_native_dimensions=_int(
            embed, "native_dimensions", 1024, "embed.native_dimensions", minimum=1
        ),
        embed_dimensions=_int(embed, "dimensions", 1024, "embed.dimensions", minimum=1),
        embed_num_ctx=_int(embed, "num_ctx", 2048, "embed.num_ctx", minimum=1),
        embed_batch_size=_int(embed, "batch_size", 8, "embed.batch_size", minimum=1),
        generate_model=_string(generate, "model", "qwen3.5:9b", "generate.model"),
        generate_fallback_model=_string(
            generate, "fallback_model", "qwen3.5:4b", "generate.fallback_model"
        ),
        generate_num_ctx=_int(generate, "num_ctx", 8192, "generate.num_ctx", minimum=1),
        vision_num_ctx=_int(
            generate, "vision_num_ctx", 4096, "generate.vision_num_ctx", minimum=1
        ),
        num_predict=_int(generate, "num_predict", 768, "generate.num_predict", minimum=1),
        vision_num_predict=_int(
            generate,
            "vision_num_predict",
            400,
            "generate.vision_num_predict",
            minimum=1,
        ),
        temperature=_float(generate, "temperature", 0.2, "generate.temperature"),
        vision_temperature=_float(
            generate, "vision_temperature", 0.1, "generate.vision_temperature"
        ),
        k=k,
        max_cosine_distance=_float(
            retrieve, "max_cosine_distance", 0.85, "retrieve.max_cosine_distance"
        ),
        max_question_tokens=_int(
            retrieve,
            "max_question_tokens",
            512,
            "retrieve.max_question_tokens",
            minimum=1,
        ),
        chunk_target_tokens=_int(
            chunk, "target_tokens", 480, "chunk.target_tokens", minimum=1
        ),
        chunk_max_tokens=_int(chunk, "max_tokens", 640, "chunk.max_tokens", minimum=1),
        chunk_overlap_tokens=_int(
            chunk, "overlap_tokens", 80, "chunk.overlap_tokens", minimum=1
        ),
        ocr_languages=_languages(ocr),
        ocr_recognition_level=level,
        ocr_confidence_threshold=_float(
            ocr, "confidence_threshold", 0.3, "ocr.confidence_threshold"
        ),
        ocr_rich_tokens=_int(ocr, "rich_tokens", 40, "ocr.rich_tokens", minimum=1),
        max_pages=_int(limits, "max_pages", 500, "limits.max_pages", minimum=1),
        max_images=_int(limits, "max_images", 300, "limits.max_images", minimum=1),
        max_bytes=_int(limits, "max_bytes", 209715200, "limits.max_bytes", minimum=1),
        min_free_bytes=_int(
            limits, "min_free_bytes", 2147483648, "limits.min_free_bytes", minimum=1
        ),
        max_pixels=_int(limits, "max_pixels", 20000000, "limits.max_pixels", minimum=1),
        nine_b_min_bytes=_int(
            limits, "nine_b_min_bytes", 9663676416, "limits.nine_b_min_bytes", minimum=1
        ),
        four_b_min_bytes=_int(
            limits,
            "four_b_min_bytes",
            6442450944,
            "limits.four_b_min_bytes",
            minimum=1,
        ),
    )


def format_config(config: Config) -> str:
    """Render the resolved configuration. The file has no secret fields."""
    lines = [
        f"config_path={config.config_path}",
        f"ui.language={config.ui_language}",
        f"data_dir={config.data_dir}",
        f"ollama.base_url={config.ollama_base_url}",
        f"embed.model={config.embed_model}",
        f"embed.native_dimensions={config.embed_native_dimensions}",
        f"embed.dimensions={config.embed_dimensions}",
        f"embed.num_ctx={config.embed_num_ctx}",
        f"embed.batch_size={config.embed_batch_size}",
        f"generate.model={config.generate_model}",
        f"generate.fallback_model={config.generate_fallback_model}",
        f"generate.num_ctx={config.generate_num_ctx}",
        f"generate.vision_num_ctx={config.vision_num_ctx}",
        f"generate.num_predict={config.num_predict}",
        f"generate.vision_num_predict={config.vision_num_predict}",
        f"generate.temperature={config.temperature}",
        f"generate.vision_temperature={config.vision_temperature}",
        f"retrieve.k={config.k}",
        f"retrieve.max_cosine_distance={config.max_cosine_distance}",
        f"retrieve.max_question_tokens={config.max_question_tokens}",
        f"chunk.target_tokens={config.chunk_target_tokens}",
        f"chunk.max_tokens={config.chunk_max_tokens}",
        f"chunk.overlap_tokens={config.chunk_overlap_tokens}",
        f"ocr.languages={','.join(config.ocr_languages)}",
        f"ocr.recognition_level={config.ocr_recognition_level}",
        f"ocr.confidence_threshold={config.ocr_confidence_threshold}",
        f"ocr.rich_tokens={config.ocr_rich_tokens}",
        f"limits.max_pages={config.max_pages}",
        f"limits.max_images={config.max_images}",
        f"limits.max_bytes={config.max_bytes}",
        f"limits.min_free_bytes={config.min_free_bytes}",
        f"limits.max_pixels={config.max_pixels}",
        f"limits.nine_b_min_bytes={config.nine_b_min_bytes}",
        f"limits.four_b_min_bytes={config.four_b_min_bytes}",
    ]
    return "\n".join(lines) + "\n"


def normalize_base_url(url: str) -> str:
    """Accept only loopback Ollama URLs and return one without a trailing slash."""
    if not isinstance(url, str):
        raise RagError(
            EXIT_USAGE, t("config.must_be_string", label="ollama.base_url")
        )
    raw = url.strip()
    if raw == "":
        raise RagError(
            EXIT_USAGE, t("config.nonempty_string", label="ollama.base_url")
        )
    parts = urllib.parse.urlsplit(raw)
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise RagError(EXIT_USAGE, t("config.url_userinfo"))
    host = parts.hostname
    if parts.scheme.lower() not in {"http", "https"} or host is None:
        raise RagError(EXIT_USAGE, _reject_url(raw))
    if host.lower() not in _LOOPBACK_HOSTS:
        raise RagError(EXIT_USAGE, _reject_url(raw))
    if parts.path not in {"", "/"} or parts.query or parts.fragment:
        raise RagError(EXIT_USAGE, t("config.url_path", url=raw))
    shown = host.lower()
    if ":" in shown:
        shown = f"[{shown}]"
    port = "" if parts.port is None else f":{parts.port}"
    return f"{parts.scheme.lower()}://{shown}{port}"


def _reject_url(url: str) -> str:
    return t("config.url_loopback", url=url)


def peek_ui_language(
    config_path: Path | None = None,
    *,
    cwd: Path | None = None,
) -> object:
    """Return the raw `[ui] language` value, or None when it is absent.

    Lookup matches `load_config`: `--config` uses `_resolve_user_path`,
    and otherwise `_find_config` walks parents for `rag.toml`. A missing
    file, a path that is not a file, an unreadable file, or invalid TOML is
    an absent value. Unknown sections are ignored. The value is not stripped.
    """
    work_dir = (cwd or Path.cwd()).resolve()
    try:
        path = _locate_config(config_path, work_dir)
    except RagError:
        return None
    try:
        text = path.read_text(encoding="utf-8")
        raw = tomllib.loads(text)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    ui = raw.get("ui")
    if not isinstance(ui, dict) or "language" not in ui:
        return None
    return ui["language"]


def _adopt_ui_language(table: dict, language_from_flag: str | None) -> str:
    if language_from_flag is not None:
        if language_from_flag not in _UI_LANGUAGES:
            raise RagError(
                EXIT_USAGE,
                t(
                    "usage.language_invalid",
                    value=language_from_flag,
                    source="--lang",
                ),
            )
        return language_from_flag
    if "language" not in table:
        return "en"
    raw = table["language"]
    if isinstance(raw, str) and raw in _UI_LANGUAGES:
        return raw
    label = raw if isinstance(raw, str) else type(raw).__name__
    raise RagError(
        EXIT_USAGE,
        t("usage.language_invalid", value=label, source="ui.language"),
    )


def _locate_config(config_path: Path | None, work_dir: Path) -> Path:
    if config_path is None:
        return _find_config(work_dir)
    path = _resolve_user_path(config_path, work_dir)
    if not path.is_file():
        raise RagError(EXIT_USAGE, t("config.file_missing", path=path))
    return path


def _find_config(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        candidate = directory / "rag.toml"
        if candidate.is_file():
            return candidate
    raise RagError(EXIT_USAGE, t("config.toml_missing"))


def _read_toml(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RagError(EXIT_USAGE, t("config.unreadable", path=path)) from exc
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RagError(
            EXIT_USAGE, t("config.toml_invalid", path=path, error=str(exc))
        ) from exc
    if not isinstance(raw, dict):
        raise RagError(EXIT_USAGE, t("config.format_invalid", path=path))
    return raw


def _sections(raw: dict) -> dict[str, dict]:
    extra = set(raw) - set(_SECTIONS)
    if extra:
        names = t("config.list_joiner").join(sorted(str(name) for name in extra))
        raise RagError(EXIT_USAGE, t("config.unknown_section", names=names))
    found: dict[str, dict] = {}
    for name, allowed in _SECTIONS.items():
        table = raw.get(name, {})
        if not isinstance(table, dict):
            raise RagError(EXIT_USAGE, t("config.must_be_table", name=name))
        unknown = set(table) - allowed
        if unknown:
            fields = t("config.list_joiner").join(
                sorted(str(field) for field in unknown)
            )
            raise RagError(
                EXIT_USAGE, t("config.unknown_field", name=name, fields=fields)
            )
        found[name] = table
    return found


def _resolve_user_path(path: Path, base: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (base / expanded).resolve()


def _resolve_dir(raw: str, base: Path) -> Path:
    return _resolve_user_path(Path(raw), base)


def _string(table: dict, key: str, default: str, label: str) -> str:
    if key not in table:
        return default
    value = table[key]
    if not isinstance(value, str) or value == "" or value != value.strip():
        raise RagError(EXIT_USAGE, t("config.nonempty_string", label=label))
    return value


def _int(table: dict, key: str, default: int, label: str, *, minimum: int) -> int:
    if key not in table:
        value = default
    else:
        raw = table[key]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise RagError(EXIT_USAGE, t("config.must_be_int", label=label))
        value = raw
    if value < minimum:
        raise RagError(
            EXIT_USAGE, t("config.minimum", label=label, minimum=minimum)
        )
    return value


def _float(table: dict, key: str, default: float, label: str) -> float:
    if key not in table:
        value = default
    else:
        raw = table[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RagError(EXIT_USAGE, t("config.must_be_number", label=label))
        value = float(raw)
    if value < 0:
        raise RagError(EXIT_USAGE, t("config.not_negative", label=label))
    return value


def _languages(table: dict) -> tuple[str, ...]:
    if "languages" not in table:
        return ("zh-Hant", "zh-Hans", "en-US")
    value = table["languages"]
    if not isinstance(value, list) or not value:
        raise RagError(EXIT_USAGE, t("ocr.languages_invalid"))
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or item == "" or item != item.strip():
            raise RagError(EXIT_USAGE, t("ocr.languages_invalid"))
        items.append(item)
    return tuple(items)
