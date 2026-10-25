"""Records the parsers return.

Parsers do not write SQLite and do not write ``data/images``. ``png_bytes``
stays in memory so a later ingest step can store the file. Text chunks are
only the text layer. OCR, figure, and caption rows are not created here.

``chunk_id`` and ``image_id`` use the same spelling as the corpus store.
This module does not import the store: the parser package depends on the
CLI shell and the chunker only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

from ragcli.chunking import chunk_text
from ragcli.errors import EXIT_DOCUMENT, EXIT_NOT_FOUND, EXIT_USAGE, RagError
from ragcli.parse.pixels import (
    PixelOverflow,
    is_decorative,
    sniff_header,
    to_png,
)

PARSER_VERSION = "1"

MEDIA_TYPES = frozenset({"pdf", "docx", "pptx", "text", "image"})
PARSER_NAMES = frozenset({"pymupdf", "python-docx", "python-pptx", "text", "image"})


@dataclass(frozen=True, slots=True)
class ParseLimits:
    """Hard caps from the design. Callers may pass the CLI overrides."""

    max_pages: int = 500
    max_bytes: int = 209_715_200
    max_pixels: int = 20_000_000
    max_images: int = 300

    def __post_init__(self) -> None:
        for name in ("max_pages", "max_bytes", "max_pixels", "max_images"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise RagError(EXIT_USAGE, f"{name} 必須是正整數。")


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """One source file. Status and ingest time belong to the ingest step."""

    doc_id: str
    source_path: str
    filename: str
    media_type: str
    byte_size: int
    sha256: str
    parser: str
    parser_version: str
    title: str | None = None
    page_count: int | None = None


@dataclass(frozen=True, slots=True)
class ParsedChunk:
    """A text-layer chunk. ``kind`` is ``text`` and ``embed_state`` is ``pending``."""

    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    kind: str
    token_estimate: int
    embed_state: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedImage:
    """An image record. ``png_bytes`` is set only when the file should be stored.

    ``preceding_chunk_count`` is how many text chunks sit before this image
    in reading order. Ingest uses that to attach the previous chunk as the
    anchor. Decorative, duplicate, and bomb images have no PNG payload.
    """

    image_id: str
    doc_id: str
    storage_relpath: str
    width: int
    height: int
    byte_size: int
    sha256: str
    kind: str
    status: str
    preceding_chunk_count: int
    page: int | None = None
    route: str | None = None
    duplicate_count: int = 0
    png_bytes: bytes | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Document, text chunks, and image records. Nothing has been written."""

    document: ParsedDocument
    chunks: tuple[ParsedChunk, ...]
    images: tuple[ParsedImage, ...]


@dataclass(frozen=True, slots=True)
class TextPiece:
    """Text in reading order, before it is split into chunks."""

    text: str
    page: int | None
    section: str | None


@dataclass(frozen=True, slots=True)
class EncodedImage:
    """Encoded bytes whose header is already under the pixel cap."""

    raw: bytes
    width: int
    height: int
    page: int | None
    kind: str


@dataclass(frozen=True, slots=True)
class RasterImage:
    """A page pixmap that was planned, then rendered. Pixels are already checked."""

    png: bytes
    width: int
    height: int
    page: int | None
    kind: str


@dataclass(frozen=True, slots=True)
class BombImage:
    """Header dimensions over the cap. The pixels were not decoded."""

    width: int
    height: int
    page: int | None
    kind: str
    locator: str


Segment = TextPiece | EncodedImage | RasterImage | BombImage


def load_bytes(path: Path, max_bytes: int) -> tuple[Path, bytes]:
    """Read one regular file. Directories are a usage error. Oversize is exit 5."""
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RagError(EXIT_NOT_FOUND, f"檔案不存在或不可讀：{path}") from exc
    except OSError as exc:
        raise RagError(EXIT_NOT_FOUND, f"檔案不存在或不可讀：{path}") from exc
    if resolved.is_dir():
        raise RagError(EXIT_USAGE, "第一版要檔案路徑，不接受資料夾。")
    if not resolved.is_file():
        raise RagError(EXIT_NOT_FOUND, f"檔案不存在或不可讀：{path}")
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise RagError(EXIT_NOT_FOUND, f"檔案不存在或不可讀：{path}") from exc
    if size > max_bytes:
        raise RagError(EXIT_DOCUMENT, "檔案超過位元組上限。")
    try:
        data = resolved.read_bytes()
    except OSError as exc:
        raise RagError(EXIT_NOT_FOUND, f"檔案不存在或不可讀：{path}") from exc
    if len(data) > max_bytes:
        raise RagError(EXIT_DOCUMENT, "檔案超過位元組上限。")
    return resolved, data


def chunk_id_for(doc_id: str, ordinal: int) -> str:
    """Return ``{first 12 hex}:{ordinal:06d}``."""
    return f"{doc_id[:12]}:{ordinal:06d}"


def image_id_for(doc_id: str, ordinal: int) -> str:
    """Return ``{first 12 hex}:img:{ordinal:04d}``."""
    return f"{doc_id[:12]}:img:{ordinal:04d}"


def storage_relpath(doc_id: str, image_id: str) -> str:
    """Return the path relative to the data directory. The file is not created."""
    return f"images/{doc_id}/{image_id}.png"


def finish(
    *,
    resolved: Path,
    data: bytes,
    media_type: str,
    parser: str,
    title: str | None,
    page_count: int | None,
    segments: list[Segment],
    limits: ParseLimits,
) -> ParseResult:
    """Turn reading-order segments into chunks and image records."""
    if media_type not in MEDIA_TYPES:
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
    if parser not in PARSER_NAMES:
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
    digest = hashlib.sha256(data).hexdigest()
    document = ParsedDocument(
        doc_id=digest,
        source_path=str(resolved),
        filename=resolved.name,
        media_type=media_type,
        byte_size=len(data),
        sha256=digest,
        parser=parser,
        parser_version=PARSER_VERSION,
        title=_clean_title(title),
        page_count=page_count,
    )
    chunks: list[ParsedChunk] = []
    images: list[ParsedImage] = []
    kept: dict[str, int] = {}
    ordinal = 0
    image_ordinal = 0
    for segment in segments:
        if isinstance(segment, TextPiece):
            ordinal = _append_text(document.doc_id, segment, chunks, ordinal)
            continue
        if image_ordinal >= limits.max_images:
            raise RagError(EXIT_DOCUMENT, "圖像張數超過上限。")
        record = _image_record(
            document.doc_id,
            image_ordinal,
            segment,
            preceding_chunk_count=ordinal,
            kept=kept,
            images=images,
            limits=limits,
        )
        images.append(record)
        image_ordinal += 1
    return ParseResult(document=document, chunks=tuple(chunks), images=tuple(images))


def _append_text(
    doc_id: str, piece: TextPiece, chunks: list[ParsedChunk], ordinal: int
) -> int:
    for part in chunk_text(piece.text):
        chunks.append(
            ParsedChunk(
                chunk_id=chunk_id_for(doc_id, ordinal),
                doc_id=doc_id,
                ordinal=ordinal,
                text=part.text,
                kind="text",
                token_estimate=part.token_estimate,
                embed_state="pending",
                page_start=piece.page,
                page_end=piece.page,
                section=piece.section,
            )
        )
        ordinal += 1
    return ordinal


def _image_record(
    doc_id: str,
    ordinal: int,
    segment: EncodedImage | RasterImage | BombImage,
    *,
    preceding_chunk_count: int,
    kept: dict[str, int],
    images: list[ParsedImage],
    limits: ParseLimits,
) -> ParsedImage:
    image_id = image_id_for(doc_id, ordinal)
    relpath = storage_relpath(doc_id, image_id)
    if isinstance(segment, BombImage):
        return _bomb(doc_id, image_id, relpath, segment, preceding_chunk_count)
    if isinstance(segment, RasterImage):
        raw = segment.png
        width, height = segment.width, segment.height
        kind = segment.kind
        page = segment.page
        png, width, height = _maybe_shrink(raw, width, height, limits)
        if is_decorative(width, height, len(png)):
            return _skipped(
                doc_id,
                image_id,
                relpath,
                width=width,
                height=height,
                raw=png,
                kind=kind,
                page=page,
                route="skipped_small",
                status="skipped",
                preceding_chunk_count=preceding_chunk_count,
            )
    else:
        width, height = _confirmed_size(segment)
        kind = segment.kind
        page = segment.page
        raw = segment.raw
        if width * height > limits.max_pixels:
            if kind == "standalone":
                raise RagError(EXIT_DOCUMENT, "圖像像素超過上限。")
            return _bomb(
                doc_id,
                image_id,
                relpath,
                BombImage(width, height, page, kind, "header"),
                preceding_chunk_count,
            )
        if is_decorative(width, height, len(raw)):
            return _skipped(
                doc_id,
                image_id,
                relpath,
                width=width,
                height=height,
                raw=raw,
                kind=kind,
                page=page,
                route="skipped_small",
                status="skipped",
                preceding_chunk_count=preceding_chunk_count,
            )
        try:
            png, width, height = to_png(raw, max_pixels=limits.max_pixels)
        except PixelOverflow:
            if kind == "standalone":
                raise RagError(EXIT_DOCUMENT, "圖像像素超過上限。") from None
            return _bomb(
                doc_id,
                image_id,
                relpath,
                BombImage(width, height, page, kind, "decode"),
                preceding_chunk_count,
            )
        if is_decorative(width, height, len(raw)):
            return _skipped(
                doc_id,
                image_id,
                relpath,
                width=width,
                height=height,
                raw=raw,
                kind=kind,
                page=page,
                route="skipped_small",
                status="skipped",
                preceding_chunk_count=preceding_chunk_count,
            )
    digest = hashlib.sha256(raw).hexdigest()
    previous = kept.get(digest)
    if previous is not None:
        images[previous] = replace(
            images[previous], duplicate_count=images[previous].duplicate_count + 1
        )
        return _skipped(
            doc_id,
            image_id,
            relpath,
            width=width,
            height=height,
            raw=raw,
            kind=kind,
            page=page,
            route="skipped_duplicate",
            status="skipped",
            preceding_chunk_count=preceding_chunk_count,
            digest=digest,
        )
    record = ParsedImage(
        image_id=image_id,
        doc_id=doc_id,
        storage_relpath=relpath,
        width=width,
        height=height,
        byte_size=len(png),
        sha256=digest,
        kind=kind,
        status="pending_ocr",
        preceding_chunk_count=preceding_chunk_count,
        page=page,
        png_bytes=png,
    )
    kept[digest] = len(images)
    return record


def _confirmed_size(segment: EncodedImage) -> tuple[int, int]:
    """Read the header again before opening the image."""
    sniffed = sniff_header(segment.raw)
    width, height = sniffed if sniffed is not None else (segment.width, segment.height)
    if width < 1 or height < 1:
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
    return width, height


def _maybe_shrink(
    png: bytes, width: int, height: int, limits: ParseLimits
) -> tuple[bytes, int, int]:
    long_side = max(width, height)
    if long_side <= 3000 and width * height <= limits.max_pixels:
        return png, width, height
    try:
        return to_png(png, max_pixels=limits.max_pixels)
    except PixelOverflow as exc:
        raise RagError(EXIT_DOCUMENT, "頁面在 72 DPI 仍超過像素上限。") from exc


def _bomb(
    doc_id: str,
    image_id: str,
    relpath: str,
    segment: BombImage,
    preceding_chunk_count: int,
) -> ParsedImage:
    token = f"skipped-bomb\n{segment.locator}\n{segment.width}\n{segment.height}"
    return ParsedImage(
        image_id=image_id,
        doc_id=doc_id,
        storage_relpath=relpath,
        width=segment.width,
        height=segment.height,
        byte_size=0,
        sha256=hashlib.sha256(token.encode()).hexdigest(),
        kind=segment.kind,
        status="skipped_bomb",
        preceding_chunk_count=preceding_chunk_count,
        page=segment.page,
        route="skipped_bomb",
        png_bytes=None,
        error="像素超過上限，未解碼。",
    )


def _skipped(
    doc_id: str,
    image_id: str,
    relpath: str,
    *,
    width: int,
    height: int,
    raw: bytes,
    kind: str,
    page: int | None,
    route: str,
    status: str,
    preceding_chunk_count: int,
    digest: str | None = None,
) -> ParsedImage:
    return ParsedImage(
        image_id=image_id,
        doc_id=doc_id,
        storage_relpath=relpath,
        width=width,
        height=height,
        byte_size=len(raw),
        sha256=digest if digest is not None else hashlib.sha256(raw).hexdigest(),
        kind=kind,
        status=status,
        preceding_chunk_count=preceding_chunk_count,
        page=page,
        route=route,
        png_bytes=None,
    )


def _clean_title(title: str | None) -> str | None:
    if title is None:
        return None
    stripped = title.strip()
    if stripped == "" or "\x00" in stripped:
        return None
    return stripped
