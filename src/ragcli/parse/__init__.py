"""Document parsers.

Each parser returns a document record, text chunks, and image records. None
of them write SQLite or ``data/images``, and none of them call OCR or Ollama.
"""

from __future__ import annotations

from pathlib import Path

from ragcli.errors import EXIT_NOT_FOUND, EXIT_UNSUPPORTED_TYPE, EXIT_USAGE, RagError
from ragcli.i18n import t
from ragcli.parse.docx import parse_docx
from ragcli.parse.image import parse_image
from ragcli.parse.model import PARSER_VERSION, ParseLimits, ParseResult
from ragcli.parse.pdf import parse_pdf
from ragcli.parse.pptx import parse_pptx
from ragcli.parse.text import parse_text

_PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".pptx": parse_pptx,
    ".txt": parse_text,
    ".md": parse_text,
    ".png": parse_image,
    ".jpg": parse_image,
    ".jpeg": parse_image,
    ".webp": parse_image,
    ".tif": parse_image,
    ".tiff": parse_image,
    ".gif": parse_image,
    ".bmp": parse_image,
    ".heic": parse_image,
    ".heif": parse_image,
}


def parse_file(path: Path, limits: ParseLimits | None = None) -> ParseResult:
    """Parse one supported file. The suffix chooses the parser.

    Directories are exit 2 and missing files are exit 4, before the suffix
    is considered. The parser then applies the byte cap itself.
    """
    limits = limits or ParseLimits()
    try:
        resolved = Path(path).resolve(strict=True)
    except FileNotFoundError as exc:
        raise RagError(EXIT_NOT_FOUND, t("io.file_unreadable", path=path)) from exc
    except OSError as exc:
        raise RagError(EXIT_NOT_FOUND, t("io.file_unreadable", path=path)) from exc
    if resolved.is_dir():
        raise RagError(EXIT_USAGE, t("usage.file_not_directory"))
    if not resolved.is_file():
        raise RagError(EXIT_NOT_FOUND, t("io.file_unreadable", path=path))
    suffix = resolved.suffix.lower()
    parser = _PARSERS.get(suffix)
    if parser is None:
        label = suffix if suffix else t("usage.no_suffix")
        raise RagError(EXIT_UNSUPPORTED_TYPE, t("usage.suffix_unsupported", label=label))
    return parser(resolved, limits)


__all__ = [
    "PARSER_VERSION",
    "ParseLimits",
    "ParseResult",
    "parse_file",
]
