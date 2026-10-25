"""DOCX parser.

Walks the body in document order. Table cells are joined with tab characters.
Pictures are the ``a:blip`` / ``r:embed`` parts inside that paragraph or
table, not the relationship dictionary order. The parser does not write files.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from ragcli.errors import EXIT_DOCUMENT, RagError
from ragcli.parse.model import (
    BombImage,
    EncodedImage,
    ParseLimits,
    ParseResult,
    Segment,
    TextPiece,
    finish,
    load_bytes,
)
from ragcli.parse.pixels import sniff_header

_NS_P = qn("w:p")
_NS_TBL = qn("w:tbl")
_NS_T = qn("w:t")
_NS_TAB = qn("w:tab")
_NS_BR = qn("w:br")
_NS_CR = qn("w:cr")
_NS_BLIP = qn("a:blip")
_NS_EMBED = qn("r:embed")


def parse_docx(path: Path, limits: ParseLimits | None = None) -> ParseResult:
    """Parse one DOCX into a document, text chunks, and image records."""
    limits = limits or ParseLimits()
    resolved, data = load_bytes(path, limits.max_bytes)
    try:
        document = Document(str(resolved))
    except Exception as exc:
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。") from exc
    segments: list[Segment] = []
    buffer: list[str] = []
    section: str | None = None

    def flush() -> None:
        text = "\n".join(part for part in buffer if part.strip()).strip()
        buffer.clear()
        if text:
            segments.append(TextPiece(text, None, section))

    for child in document.element.body.iterchildren():
        if child.tag == _NS_P:
            paragraph = Paragraph(child, document)
            heading = _heading(paragraph)
            if heading is not None:
                flush()
                section = heading
            for kind, value in _paragraph_events(paragraph, document):
                if kind == "text":
                    buffer.append(value)
                    continue
                flush()
                segments.append(_classify(value, page=None, limits=limits, locator="docx"))
        elif child.tag == _NS_TBL:
            flush()
            table = Table(child, document)
            text = _table_text(table)
            if text:
                buffer.append(text)
            flush()
            for blob in _blobs(child, document):
                segments.append(_classify(blob, page=None, limits=limits, locator="docx"))
    flush()
    return finish(
        resolved=resolved,
        data=data,
        media_type="docx",
        parser="python-docx",
        title=None,
        page_count=None,
        segments=segments,
        limits=limits,
    )


def _heading(paragraph: Paragraph) -> str | None:
    style = paragraph.style
    name = "" if style is None or not style.name else style.name
    if not (name.startswith("Heading") or name.startswith("標題")):
        return None
    text = paragraph.text.strip()
    return text or None


def _paragraph_events(paragraph: Paragraph, document: Document) -> list[tuple[str, str | bytes]]:
    events: list[tuple[str, str | bytes]] = []
    texts: list[str] = []
    seen: set[str] = set()

    def flush_text() -> None:
        text = "".join(texts).strip()
        texts.clear()
        if text:
            events.append(("text", text))

    for node in paragraph._element.iter():
        if node.tag == _NS_T and node.text:
            texts.append(node.text)
        elif node.tag == _NS_TAB:
            texts.append("\t")
        elif node.tag in (_NS_BR, _NS_CR):
            texts.append("\n")
        elif node.tag == _NS_BLIP:
            embed = node.get(_NS_EMBED)
            if not embed or embed in seen:
                continue
            part = document.part.related_parts.get(embed)
            if part is None:
                raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
            seen.add(embed)
            flush_text()
            events.append(("image", part.blob))
    flush_text()
    return events


def _table_text(table: Table) -> str:
    rows: list[str] = []
    seen: set[int] = set()
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            marker = id(cell._tc)
            if marker in seen:
                continue
            seen.add(marker)
            cells.append(cell.text.replace("\t", " ").replace("\n", " ").strip())
        if cells:
            rows.append("\t".join(cells))
    return "\n".join(rows).strip()


def _blobs(element, document: Document) -> list[bytes]:
    blobs: list[bytes] = []
    seen: set[str] = set()
    for node in element.iter():
        if node.tag != _NS_BLIP:
            continue
        embed = node.get(_NS_EMBED)
        if not embed or embed in seen:
            continue
        part = document.part.related_parts.get(embed)
        if part is None:
            raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
        seen.add(embed)
        blobs.append(part.blob)
    return blobs


def _classify(raw: bytes, *, page: int | None, limits: ParseLimits, locator: str):
    sniffed = sniff_header(raw)
    if sniffed is None:
        raise RagError(EXIT_DOCUMENT, "圖像格式無法在解碼前讀到尺寸。")
    width, height = sniffed
    if width * height > limits.max_pixels:
        return BombImage(width, height, page, "embedded", locator)
    return EncodedImage(raw, width, height, page, "embedded")
