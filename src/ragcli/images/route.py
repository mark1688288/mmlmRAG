"""Route OCR text and build the synthetic chunk text.

Pure functions. Nothing is written to SQLite. An empty OCR string is 0
tokens: ``estimate_tokens`` would report 1 because of its trailing ``+ 1``,
and that would send a blank image down the caption path incorrectly.

``text_rich`` splits only the OCR body and repeats one header.
``label_and_caption`` and ``caption_only`` are one block per image. The
description line is omitted until a caption exists, which leaves the block
``blocked``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragcli.chunking import (
    MAX_TOKENS,
    OVERLAP_TOKENS,
    TARGET_TOKENS,
    chunk_ocr_body,
    estimate_tokens,
)

ROUTE_TEXT_RICH = "text_rich"
ROUTE_LABEL_AND_CAPTION = "label_and_caption"
ROUTE_CAPTION_ONLY = "caption_only"
RICH_TOKENS = 40
CONTEXT_CHARS = 120

_ROUTES = frozenset(
    {ROUTE_TEXT_RICH, ROUTE_LABEL_AND_CAPTION, ROUTE_CAPTION_ONLY}
)


@dataclass(frozen=True, slots=True)
class SyntheticChunk:
    """Text a later ingest step can store. This step does not store it."""

    text: str
    kind: str
    embed_state: str
    token_estimate: int
    image_id: str
    anchor_chunk_id: str | None = None


@dataclass(frozen=True, slots=True)
class RoutedImage:
    """The route and the chunks that carry it. Skipped images have none."""

    route: str
    chunks: tuple[SyntheticChunk, ...]


def ocr_tokens(text: str) -> int:
    """Token estimate of OCR text. Blank text is 0, not 1."""
    if not isinstance(text, str):
        raise TypeError("text 必須是字串。")
    body = text.strip()
    if body == "":
        return 0
    return estimate_tokens(body)


def route_for(text: str, *, rich_tokens: int = RICH_TOKENS) -> str:
    """Map an OCR estimate to ``text_rich``, ``label_and_caption``, or ``caption_only``."""
    if isinstance(rich_tokens, bool) or not isinstance(rich_tokens, int) or rich_tokens < 1:
        raise ValueError("rich_tokens 必須是正整數。")
    count = ocr_tokens(text)
    if count <= 0:
        return ROUTE_CAPTION_ONLY
    if count >= rich_tokens:
        return ROUTE_TEXT_RICH
    return ROUTE_LABEL_AND_CAPTION


def needs_vision(route: str) -> bool:
    """True when the route still needs a vision caption."""
    if route == ROUTE_TEXT_RICH:
        return False
    if route in {ROUTE_LABEL_AND_CAPTION, ROUTE_CAPTION_ONLY}:
        return True
    if isinstance(route, str) and route.startswith("skipped_"):
        return False
    raise ValueError("route 不是已知的圖像路由。")


def source_header(filename: str, page: int | None, *, ocr: bool) -> str:
    """One-line source header. A missing page omits the page clause."""
    name = _filename(filename)
    label = "圖像 OCR" if ocr else "圖像"
    if page is None:
        return f"[{label}] {name}"
    if isinstance(page, bool) or type(page) is not int or page < 1:
        raise ValueError("page 必須是正整數或 None。")
    return f"[{label}] {name} 第{page}頁"


def compose(
    route: str,
    ocr_text: str,
    *,
    filename: str,
    page: int | None,
    image_id: str,
    before: str = "",
    after: str = "",
    anchor_chunk_id: str | None = None,
    caption: str | None = None,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> RoutedImage:
    """Build synthetic chunks for one image. The route argument is authoritative.

    ``caption`` None leaves the description line out and marks the block
    ``blocked``. A blank caption is the same as None. ``text_rich`` ignores
    both the caption and the surrounding context.
    """
    if route not in _ROUTES:
        raise ValueError("route 不是已知的圖像路由。")
    if not isinstance(ocr_text, str):
        raise TypeError("text 必須是字串。")
    identity = _image_id(image_id)
    anchor = _anchor(anchor_chunk_id)
    if route == ROUTE_TEXT_RICH:
        chunks = _ocr_chunks(
            ocr_text,
            filename=filename,
            page=page,
            image_id=identity,
            anchor=anchor,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        return RoutedImage(route, chunks)
    text = _context_text(
        route,
        ocr_text,
        filename=filename,
        page=page,
        before=before,
        after=after,
        caption=caption,
    )
    estimate = estimate_tokens(text)
    if estimate > max_tokens:
        raise ValueError("圖像區塊超過硬上限。")
    paragraph = _paragraph(caption)
    kind = "figure" if route == ROUTE_LABEL_AND_CAPTION else "caption"
    state = "pending" if paragraph is not None else "blocked"
    chunk = SyntheticChunk(
        text=text,
        kind=kind,
        embed_state=state,
        token_estimate=estimate,
        image_id=identity,
        anchor_chunk_id=anchor,
    )
    return RoutedImage(route, (chunk,))


def add_caption_line(text: str, caption: str) -> str:
    """Insert ``描述：`` before the template's final ``下文：`` line."""
    if not isinstance(text, str):
        raise TypeError("text 必須是字串。")
    paragraph = _paragraph(caption)
    if paragraph is None:
        raise ValueError("描述必須是非空字串。")
    lines = text.split("\n")
    if any(line.startswith("描述：") for line in lines):
        raise ValueError("描述行已經存在。")
    inserted = f"描述：{paragraph}"
    indexes = [index for index, line in enumerate(lines) if line.startswith("下文：")]
    if indexes:
        lines.insert(indexes[-1], inserted)
    else:
        lines.append(inserted)
    return "\n".join(lines)


def _ocr_chunks(
    ocr_text: str,
    *,
    filename: str,
    page: int | None,
    image_id: str,
    anchor: str | None,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> tuple[SyntheticChunk, ...]:
    header = source_header(filename, page, ocr=True)
    pieces = chunk_ocr_body(
        ocr_text,
        header=header,
        image_id=image_id,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    return tuple(
        SyntheticChunk(
            text=piece.text,
            kind="ocr",
            embed_state="pending",
            token_estimate=piece.token_estimate,
            image_id=image_id,
            anchor_chunk_id=anchor,
        )
        for piece in pieces
    )


def _context_text(
    route: str,
    ocr_text: str,
    *,
    filename: str,
    page: int | None,
    before: str,
    after: str,
    caption: str | None,
) -> str:
    lines = [
        source_header(filename, page, ocr=False),
        f"上文：{_clip(before, tail=True)}",
    ]
    if route == ROUTE_LABEL_AND_CAPTION:
        lines.append(f"圖中文字：{ocr_text.strip()}")
    paragraph = _paragraph(caption)
    if paragraph is not None:
        lines.append(f"描述：{paragraph}")
    lines.append(f"下文：{_clip(after, tail=False)}")
    return "\n".join(lines)


def _clip(text: str, *, tail: bool) -> str:
    if not isinstance(text, str):
        raise TypeError("上文同下文必須是字串。")
    stripped = text.strip()
    if len(stripped) <= CONTEXT_CHARS:
        return stripped
    if tail:
        return stripped[-CONTEXT_CHARS:]
    return stripped[:CONTEXT_CHARS]


def _paragraph(caption: str | None) -> str | None:
    if caption is None:
        return None
    if not isinstance(caption, str):
        raise TypeError("描述必須是字串。")
    paragraph = " ".join(caption.split())
    if paragraph == "":
        return None
    return paragraph


def _filename(filename: str) -> str:
    if not isinstance(filename, str):
        raise TypeError("filename 必須是字串。")
    if "\n" in filename or "\r" in filename:
        raise ValueError("filename 必須是單行檔名。")
    cleaned = filename.strip()
    if cleaned == "" or "/" in cleaned or "\\" in cleaned:
        raise ValueError("filename 必須是單行檔名。")
    return cleaned


def _image_id(image_id: str) -> str:
    if (
        not isinstance(image_id, str)
        or image_id.strip() == ""
        or any(char.isspace() for char in image_id)
    ):
        raise ValueError("image_id 必須是不含空白的非空字串。")
    return image_id


def _anchor(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or value.strip() == ""
        or any(char.isspace() for char in value)
    ):
        raise ValueError("anchor_chunk_id 必須是不含空白的非空字串。")
    return value
