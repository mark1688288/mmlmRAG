"""Fixtures for parser tests. Files are generated in the test, not committed."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import pymupdf
from PIL import Image

from ragcli.chunking import MAX_TOKENS, estimate_tokens
from ragcli.parse.model import ParseResult

FONT = Path("/Library/Fonts/Arial Unicode.ttf")


def solid_image(width: int, height: int, color: tuple[int, int, int], fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format=fmt)
    return buf.getvalue()


def write_image(path: Path, width: int, height: int, color: tuple[int, int, int], fmt: str) -> Path:
    path.write_bytes(solid_image(width, height, color, fmt))
    return path


def bomb_png(width: int, height: int) -> bytes:
    """PNG signature and IHDR only. Opening it would not be a safe decode."""
    payload = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return _png_chunk(b"IHDR", payload)


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    import zlib

    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def heif_ispe(width: int, height: int) -> bytes:
    """Minimal HEIF box tree whose largest ispe box has the given canvas."""

    def box(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", 8 + len(payload)) + tag + payload

    ispe = box(b"ispe", b"\x00\x00\x00\x00" + struct.pack(">II", width, height))
    ipco = box(b"ipco", ispe)
    iprp = box(b"iprp", ipco)
    meta = box(b"meta", b"\x00\x00\x00\x00" + iprp)
    ftyp = box(b"ftyp", b"heic" + b"\x00\x00\x00\x00" + b"heic")
    return ftyp + meta


def new_pdf() -> pymupdf.Document:
    return pymupdf.open()


def use_cjk(page: pymupdf.Page) -> None:
    page.insert_font(fontname="uni", fontfile=str(FONT))


def put_text(page: pymupdf.Page, point: tuple[float, float], text: str, size: int = 12) -> None:
    page.insert_text(point, text, fontname="uni", fontsize=size)


def assert_text_chunks(result: ParseResult) -> None:
    for chunk in result.chunks:
        assert chunk.kind == "text"
        assert chunk.embed_state == "pending"
        assert chunk.token_estimate == estimate_tokens(chunk.text)
        assert chunk.token_estimate <= MAX_TOKENS
        assert chunk.chunk_id == f"{result.document.doc_id[:12]}:{chunk.ordinal:06d}"
        assert chunk.doc_id == result.document.doc_id
