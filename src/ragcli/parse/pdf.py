"""PDF parser.

Uses PyMuPDF (``pymupdf``, not ``fitz``). Returns records only. Encrypted
files are rejected without asking for a password. A scanned page plans its
pixel size before ``get_pixmap``. An embedded image whose header is over
``max_pixels`` is ``skipped_bomb`` and is not passed to ``extract_image``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pymupdf

from ragcli.errors import EXIT_DOCUMENT, RagError
from ragcli.i18n import t
from ragcli.parse.model import (
    BombImage,
    EncodedImage,
    ParseLimits,
    ParseResult,
    RasterImage,
    Segment,
    TextPiece,
    finish,
    load_bytes,
)
from ragcli.parse.pixels import is_decorative, sniff_header

SCAN_TEXT_CHARS = 20
VECTOR_DRAWING_MIN = 40
_START_DPI = 200
_FLOOR_DPI = 72
_TEXT_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES


def parse_pdf(path: Path, limits: ParseLimits | None = None) -> ParseResult:
    """Parse one PDF into a document, text chunks, and image records."""
    limits = limits or ParseLimits()
    resolved, data = load_bytes(path, limits.max_bytes)
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise RagError(EXIT_DOCUMENT, t("parse.corrupt")) from exc
    try:
        _reject_password(document)
        if document.page_count > limits.max_pages:
            raise RagError(EXIT_DOCUMENT, t("parse.page_limit"))
        segments: list[Segment] = []
        for index in range(document.page_count):
            _read_page(document, document.load_page(index), index + 1, segments, limits)
        return finish(
            resolved=resolved,
            data=data,
            media_type="pdf",
            parser="pymupdf",
            title=_title(document),
            page_count=document.page_count,
            segments=segments,
            limits=limits,
        )
    finally:
        document.close()


def plan_raster(
    width_pt: float,
    height_pt: float,
    *,
    max_pixels: int,
    max_long_side: int = 3000,
    start_dpi: int = _START_DPI,
    floor_dpi: int = _FLOOR_DPI,
) -> tuple[int, int, int]:
    """Return ``(dpi, pixel_width, pixel_height)`` before any pixmap is made.

    Starts at 200 DPI. Lowers DPI when the pixel count exceeds ``max_pixels``
    or the long side exceeds 3000. 3000 is an output cap, not the bomb check.
    Exit 5 when 72 DPI is still over ``max_pixels``.
    """
    if width_pt <= 0 or height_pt <= 0:
        raise RagError(EXIT_DOCUMENT, t("parse.corrupt"))

    def pixels(dpi: int) -> tuple[int, int]:
        width = max(1, math.ceil(width_pt * dpi / 72))
        height = max(1, math.ceil(height_pt * dpi / 72))
        return width, height

    def fits(dpi: int) -> bool:
        width, height = pixels(dpi)
        return width * height <= max_pixels and max(width, height) <= max_long_side

    if fits(start_dpi):
        width, height = pixels(start_dpi)
        return start_dpi, width, height
    chosen = None
    low = floor_dpi
    high = start_dpi - 1
    while low <= high:
        mid = (low + high) // 2
        if fits(mid):
            chosen = mid
            low = mid + 1
        else:
            high = mid - 1
    if chosen is not None:
        width, height = pixels(chosen)
        return chosen, width, height
    width, height = pixels(floor_dpi)
    if width * height > max_pixels:
        raise RagError(EXIT_DOCUMENT, t("parse.page_pixels"))
    return floor_dpi, width, height


def _reject_password(document: pymupdf.Document) -> None:
    if document.needs_pass and not document.authenticate(""):
        raise RagError(EXIT_DOCUMENT, t("parse.pdf_encrypted"))


def _title(document: pymupdf.Document) -> str | None:
    meta = document.metadata or {}
    title = meta.get("title")
    if not isinstance(title, str):
        return None
    return title


def _read_page(
    document: pymupdf.Document,
    page: pymupdf.Page,
    page_no: int,
    segments: list[Segment],
    limits: ParseLimits,
) -> None:
    raw_text = page.get_text("text")
    if _non_ws(raw_text) < SCAN_TEXT_CHARS:
        if raw_text.strip():
            segments.append(TextPiece(raw_text.strip(), page_no, None))
        segments.append(_render(page, page_no, "page_render", limits))
        return
    text = _page_text(page)
    if text:
        segments.append(TextPiece(text, page_no, None))
    ocr_candidate = False
    for info in page.get_images(full=True):
        if _embedded(document, info, page_no, segments, limits):
            ocr_candidate = True
    if not ocr_candidate and len(page.get_drawings()) > VECTOR_DRAWING_MIN:
        segments.append(_render(page, page_no, "vector_render", limits))


def _embedded(
    document: pymupdf.Document,
    info: tuple,
    page_no: int,
    segments: list[Segment],
    limits: ParseLimits,
) -> bool:
    """Append one embedded image. Return True when it still needs OCR.

    Width and height come from ``get_images`` and are checked before
    ``extract_image``.
    """
    xref = int(info[0])
    width = int(info[2])
    height = int(info[3])
    if width < 1 or height < 1:
        raise RagError(EXIT_DOCUMENT, t("parse.corrupt"))
    if width * height > limits.max_pixels:
        segments.append(
            BombImage(width, height, page_no, "embedded", f"pdf:{page_no}:{xref}")
        )
        return False
    extracted = document.extract_image(xref)
    raw = extracted.get("image") if isinstance(extracted, dict) else None
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        raise RagError(EXIT_DOCUMENT, t("parse.corrupt"))
    raw = bytes(raw)
    sniffed = sniff_header(raw)
    if sniffed is not None and sniffed[0] * sniffed[1] > limits.max_pixels:
        segments.append(
            BombImage(sniffed[0], sniffed[1], page_no, "embedded", f"pdf:{page_no}:{xref}")
        )
        return False
    decoded_w = int(extracted.get("width") or width)
    decoded_h = int(extracted.get("height") or height)
    if decoded_w * decoded_h > limits.max_pixels:
        segments.append(
            BombImage(decoded_w, decoded_h, page_no, "embedded", f"pdf:{page_no}:{xref}")
        )
        return False
    use_w, use_h = sniffed if sniffed is not None else (decoded_w, decoded_h)
    segments.append(EncodedImage(raw, use_w, use_h, page_no, "embedded"))
    return not is_decorative(use_w, use_h, len(raw))


def _render(
    page: pymupdf.Page, page_no: int, kind: str, limits: ParseLimits
) -> RasterImage:
    _dpi, width, height = plan_raster(
        float(page.rect.width),
        float(page.rect.height),
        max_pixels=limits.max_pixels,
    )
    if width * height > limits.max_pixels:
        raise RagError(EXIT_DOCUMENT, t("parse.page_pixels"))
    pixmap = page.get_pixmap(dpi=_dpi, alpha=False, colorspace=pymupdf.csRGB)
    if pixmap.width * pixmap.height > limits.max_pixels:
        raise RagError(EXIT_DOCUMENT, t("parse.page_pixels"))
    return RasterImage(
        png=pixmap.tobytes("png"),
        width=pixmap.width,
        height=pixmap.height,
        page=page_no,
        kind=kind,
    )


def _page_text(page: pymupdf.Page) -> str:
    tables = _tables(page)
    try:
        data = page.get_text("dict", flags=_TEXT_FLAGS)
    except Exception as exc:
        raise RagError(EXIT_DOCUMENT, t("parse.corrupt")) from exc
    used: set[int] = set()
    parts: list[str] = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = block.get("bbox")
        table = _table_at(bbox, tables)
        if table is not None:
            marker = id(table)
            if marker not in used:
                used.add(marker)
                markdown = _markdown(table)
                if markdown:
                    parts.append(markdown)
            continue
        text = _block_text(block).strip()
        if text:
            parts.append(text)
    for table in tables:
        if id(table) in used:
            continue
        markdown = _markdown(table)
        if markdown:
            parts.append(markdown)
    return "\n".join(parts).strip()


def _tables(page: pymupdf.Page) -> list:
    try:
        finder = page.find_tables()
        return list(finder.tables)
    except Exception:
        return []


def _table_at(bbox: object, tables: list):
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = (float(value) for value in bbox)
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    for table in tables:
        box = table.bbox
        if box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3]:
            return table
    return None


def _markdown(table) -> str:
    try:
        rows = table.extract()
    except Exception:
        return ""
    cleaned: list[list[str]] = []
    for row in rows or []:
        cells = []
        for cell in row:
            text = "" if cell is None else str(cell)
            cells.append(text.replace("\n", " ").replace("|", "\\|").strip())
        cleaned.append(cells)
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    lines = [_markdown_line(cleaned[0], width), _markdown_rule(width)]
    lines.extend(_markdown_line(row, width) for row in cleaned[1:])
    return "\n".join(lines)


def _markdown_line(cells: list[str], width: int) -> str:
    padded = cells + [""] * (width - len(cells))
    return "| " + " | ".join(padded) + " |"


def _markdown_rule(width: int) -> str:
    return "| " + " | ".join("---" for _ in range(width)) + " |"


def _block_text(block: dict) -> str:
    lines: list[str] = []
    for line in block.get("lines", []):
        lines.append("".join(span.get("text", "") for span in line.get("spans", [])))
    return "\n".join(lines)


def _non_ws(text: str) -> int:
    return sum(1 for char in text if not char.isspace())
