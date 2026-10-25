"""Plain text and Markdown.

The whole file is decoded, then chunked once. Encoding order is
``utf-8-sig``, ``utf-8``, then Big5 and GB18030. When both legacy codecs
accept the bytes, the less noisy decoding is kept, with Big5 winning a tie.
``charset-normalizer`` runs only after those four fail. There is no OCR
and no image record.
"""

from __future__ import annotations

from pathlib import Path

from ragcli.errors import EXIT_DOCUMENT, RagError
from ragcli.parse.model import ParseLimits, ParseResult, TextPiece, finish, load_bytes


def parse_text(path: Path, limits: ParseLimits | None = None) -> ParseResult:
    """Parse one ``.txt`` or ``.md`` file."""
    limits = limits or ParseLimits()
    resolved, data = load_bytes(path, limits.max_bytes)
    text = decode_text(data)
    segments = [TextPiece(text, None, None)] if text.strip() else []
    return finish(
        resolved=resolved,
        data=data,
        media_type="text",
        parser="text",
        title=None,
        page_count=None,
        segments=segments,
        limits=limits,
    )


def decode_text(data: bytes) -> str:
    """Decode bytes in the design order. Do not assume Mac Roman.

    ``utf-8-sig`` and ``utf-8`` win when they decode. Big5 and GB18030 are
    both tried, because Python's Big5 codec accepts many GB18030 sequences
    and the first success would turn simplified Chinese into the wrong
    characters. When both accept the bytes, the less noisy decoding wins.
    A tie keeps Big5, then GB18030. ``charset-normalizer`` runs only when
    none of those four codecs accept the bytes.
    """
    if data == b"":
        return ""
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    legacy: list[str] = []
    for encoding in ("big5", "gb18030"):
        try:
            legacy.append(data.decode(encoding))
        except UnicodeDecodeError:
            continue
    if len(legacy) == 1:
        return legacy[0]
    if len(legacy) == 2:
        return _prefer_legacy(legacy)
    return _charset_normalizer(data)


def _prefer_legacy(decoded: list[str]) -> str:
    """Return the Big5 or GB18030 decoding that looks less like mojibake."""
    best = decoded[0]
    best_cost = _legacy_cost(best)
    for text in decoded[1:]:
        cost = _legacy_cost(text)
        if cost < best_cost:
            best = text
            best_cost = cost
    return best


def _legacy_cost(text: str) -> tuple[int, int]:
    """Count kana, hangul, private-use, then uncommon CJK. Lower is better."""
    from charset_normalizer.utils import (
        is_cjk,
        is_cjk_uncommon,
        is_hangul,
        is_hiragana,
        is_katakana,
    )

    foreign = 0
    uncommon = 0
    for char in text:
        if is_hiragana(char) or is_katakana(char) or is_hangul(char):
            foreign += 1
        elif 0xE000 <= ord(char) <= 0xF8FF:
            foreign += 1
        elif is_cjk(char) and is_cjk_uncommon(char):
            uncommon += 1
    return foreign, uncommon


def _charset_normalizer(data: bytes) -> str:
    from charset_normalizer import from_bytes

    best = from_bytes(data).best()
    if best is None:
        raise RagError(EXIT_DOCUMENT, "無法判斷文字編碼。")
    return str(best)
