"""Split text into overlapping chunks.

This module is pure: it does not read the corpus, call Ollama, or write
``data/``. Token estimates use the design formula, which counts only the
CJK Unified Ideographs block (U+4E00–U+9FFF) as one token each. Hiragana,
hangul, and punctuation stay on the four-characters-per-token side. The
trailing ``+ 1`` is part of that formula.

Separators are tried from coarsest to finest: blank lines, newlines,
``。！？；``, then ``.!?``. A piece that still fits under the hard maximum
stays whole when it has no separator. Hard cuts run only after that, so a
long sentence is not sliced at the target when it still fits under the
maximum. CRLF and a lone CR become LF before splitting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_TOKENS = 480
MAX_TOKENS = 640
OVERLAP_TOKENS = 80

_BLANK_LINE = re.compile(r"\n(?:[ \t]*\n)+")
_NEWLINE = re.compile(r"\n")
_CJK_PUNCT = re.compile(r"[。！？；]")
# A period between digits stays inside the number. Sentence punctuation must
# be followed by whitespace or the end of the text, so "3.14" and "..." do
# not break on the first mark.
_EN_SENTENCE = re.compile(r"(?<!\d)\.(?=\s|$)|(?<=\d)\.(?!\d)(?=\s|$)|[!?](?=\s|$)")
_SEPARATORS = (_BLANK_LINE, _NEWLINE, _CJK_PUNCT, _EN_SENTENCE)


@dataclass(frozen=True, slots=True)
class TextChunk:
    """One slice of source text. ``image_id`` is set only for OCR pieces."""

    text: str
    token_estimate: int
    image_id: str | None = None


def estimate_tokens(text: str) -> int:
    """Return the conservative token estimate from the design."""
    if not isinstance(text, str):
        raise TypeError("text 必須是字串。")
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    return cjk + (len(text) - cjk) // 4 + 1


def chunk_text(
    text: str,
    *,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[TextChunk]:
    """Split ``text`` into chunks near ``target_tokens`` and never over ``max_tokens``.

    Adjacent chunks share a suffix of about ``overlap_tokens`` when that
    suffix still fits under the hard maximum. Surrounding whitespace is
    removed. An empty result means there was no text to keep.
    """
    if not isinstance(text, str):
        raise TypeError("text 必須是字串。")
    _require_budget(target_tokens, max_tokens, overlap_tokens)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    units = _units(normalized, target_tokens, max_tokens)
    packed = _pack(units, target_tokens, max_tokens, overlap_tokens)
    return [_make(piece) for piece in packed]


def chunk_ocr_body(
    body: str,
    *,
    header: str,
    image_id: str,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[TextChunk]:
    """Split OCR body text and repeat ``header`` plus ``image_id`` on every piece.

    Only the body is cut. Each piece is ``header``, a newline, then that body
    slice. The header's own estimate is subtracted from the body budget, so
    the combined chunk stays within ``max_tokens``. An empty body yields no
    chunks.
    """
    if not isinstance(body, str):
        raise TypeError("body 必須是字串。")
    if not isinstance(header, str):
        raise TypeError("header 必須是字串。")
    if not isinstance(image_id, str):
        raise TypeError("image_id 必須是字串。")
    _require_budget(target_tokens, max_tokens, overlap_tokens)
    if "\n" in header or "\r" in header:
        raise ValueError("OCR 標頭必須是單行。")
    header_line = header.strip()
    if header_line == "":
        raise ValueError("OCR 標頭必須是非空字串。")
    if image_id.strip() == "" or any(char.isspace() for char in image_id):
        raise ValueError("image_id 必須是不含空白的非空字串。")

    prefix = header_line + "\n"
    prefix_tokens = estimate_tokens(prefix)
    if prefix_tokens >= max_tokens:
        raise ValueError("OCR 標頭本身已達到硬上限。")
    sample = body.strip()[:1]
    if sample and estimate_tokens(prefix + sample) > max_tokens:
        raise ValueError("硬上限放不進標頭加一個字元。")

    # Concatenation adds at most the two estimates, so the body budget is the
    # room left under each limit after the header. See estimate_tokens.
    body_max = max_tokens - prefix_tokens
    body_target = min(max(1, target_tokens - prefix_tokens), body_max)
    body_overlap = min(overlap_tokens, max(0, body_target - 1))
    pieces = chunk_text(
        body,
        target_tokens=body_target,
        max_tokens=body_max,
        overlap_tokens=body_overlap,
    )
    chunks: list[TextChunk] = []
    for piece in pieces:
        combined = prefix + piece.text
        estimate = estimate_tokens(combined)
        if estimate > max_tokens:
            raise ValueError("帶標頭的區塊超過硬上限。")
        chunks.append(
            TextChunk(text=combined, token_estimate=estimate, image_id=image_id)
        )
    return chunks


def _make(text: str) -> TextChunk:
    return TextChunk(text=text, token_estimate=estimate_tokens(text))


def _require_budget(target: int, maximum: int, overlap: int) -> None:
    for name, value in (
        ("target_tokens", target),
        ("max_tokens", maximum),
        ("overlap_tokens", overlap),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} 必須是整數。")
    if target < 1:
        raise ValueError("target_tokens 必須至少是 1。")
    if maximum < target:
        raise ValueError("max_tokens 不可小於 target_tokens。")
    if overlap < 0:
        raise ValueError("overlap_tokens 不可是負數。")
    if overlap >= target:
        raise ValueError("overlap_tokens 必須小於 target_tokens。")


def _units(text: str, target: int, maximum: int, sep_index: int = 0) -> list[str]:
    if text == "":
        return []
    if estimate_tokens(text) <= target:
        return [text]
    if sep_index >= len(_SEPARATORS):
        if estimate_tokens(text) <= maximum:
            return [text]
        return _hard_cut(text, target, maximum)

    parts = _split_keep_left(text, _SEPARATORS[sep_index])
    if len(parts) <= 1:
        return _units(text, target, maximum, sep_index + 1)

    units: list[str] = []
    for part in parts:
        units.extend(_units(part, target, maximum, sep_index + 1))
    if "".join(units) != text:
        raise ValueError("internal: units did not cover the text")
    return units


def _split_keep_left(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Split on ``pattern`` and keep each match at the end of the left piece."""
    matches = list(pattern.finditer(text))
    if not matches:
        return [text]
    parts: list[str] = []
    last = 0
    for match in matches:
        if match.end() <= last:
            continue
        parts.append(text[last : match.end()])
        last = match.end()
    if last < len(text):
        parts.append(text[last:])
    parts = [part for part in parts if part]
    if len(parts) <= 1 or "".join(parts) != text:
        return [text]
    return parts


def _hard_cut(text: str, target: int, maximum: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = _walk_end(text, start, target)
        if end <= start:
            end = _walk_end(text, start, maximum)
        if end <= start:
            raise ValueError("max_tokens 放不進一個字元。")
        pieces.append(text[start:end])
        start = end
    if "".join(pieces) != text:
        raise ValueError("internal: hard cut did not cover the text")
    return pieces


def _walk_end(text: str, start: int, budget: int) -> int:
    """Largest end index at or after ``start`` whose span stays within ``budget``."""
    if budget < 1:
        return start
    cjk = 0
    best = start
    for end in range(start + 1, len(text) + 1):
        char = text[end - 1]
        if "\u4e00" <= char <= "\u9fff":
            cjk += 1
        span = end - start
        estimate = cjk + (span - cjk) // 4 + 1
        if estimate > budget:
            break
        best = end
    return best


def _pack(units: list[str], target: int, maximum: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current_units: list[str] = []
    overlap_prefix = ""

    def current_text() -> str:
        return overlap_prefix + "".join(current_units)

    for unit in units:
        if unit == "":
            continue
        if estimate_tokens(unit) > maximum:
            raise ValueError("internal: unit exceeds max_tokens")
        if not current_units:
            overlap_prefix = ""
            current_units = [unit]
            continue
        merged = current_text() + unit
        if estimate_tokens(current_text()) <= target and estimate_tokens(merged) <= target:
            current_units.append(unit)
            continue
        chunks.append(current_text())
        overlap_prefix = _fitting_suffix(
            _suffix_for_tokens(current_text(), overlap),
            unit,
            maximum,
        )
        current_units = [unit]

    if current_units:
        chunks.append(current_text())
    packed = [chunk for chunk in chunks if chunk]
    for chunk in packed:
        if estimate_tokens(chunk) > maximum:
            raise ValueError("internal: chunk exceeds max_tokens")
    return packed


def _suffix_for_tokens(text: str, budget: int) -> str:
    """Longest suffix whose own estimate is within ``budget``."""
    if budget <= 0 or text == "":
        return ""
    low = 1
    high = len(text)
    best = 0
    while low <= high:
        mid = (low + high) // 2
        if estimate_tokens(text[-mid:]) <= budget:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    if best == 0:
        return ""
    return text[-best:]


def _fitting_suffix(candidate: str, upcoming: str, maximum: int) -> str:
    """Longest suffix of ``candidate`` that still fits in front of ``upcoming``."""
    if candidate == "":
        return ""
    low = 0
    high = len(candidate)
    best = 0
    while low <= high:
        mid = (low + high) // 2
        piece = candidate[-mid:] if mid else ""
        if estimate_tokens(piece + upcoming) <= maximum:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    if best == 0:
        return ""
    return candidate[-best:]
