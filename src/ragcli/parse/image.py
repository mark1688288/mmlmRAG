"""Standalone image files.

Header width and height are read before ``Image.open``. Over ``max_pixels``
is exit 5, and no image record is returned. GIF keeps the first frame. HEIC
is opened only after ``pillow-heif`` is registered, which happens inside the
PNG conversion after the header check.
"""

from __future__ import annotations

from pathlib import Path

from ragcli.errors import EXIT_DOCUMENT, RagError
from ragcli.parse.model import EncodedImage, ParseLimits, ParseResult, finish, load_bytes
from ragcli.parse.pixels import header_size

_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".gif", ".bmp", ".heic", ".heif"}
)


def parse_image(path: Path, limits: ParseLimits | None = None) -> ParseResult:
    """Parse one standalone image. There is no surrounding text."""
    limits = limits or ParseLimits()
    resolved, data = load_bytes(path, limits.max_bytes)
    suffix = resolved.suffix.lower()
    if suffix not in _SUFFIXES:
        raise RagError(EXIT_DOCUMENT, "圖像格式無法在解碼前讀到尺寸。")
    width, height = header_size(data, suffix)
    if width * height > limits.max_pixels:
        raise RagError(EXIT_DOCUMENT, "圖像像素超過上限。")
    return finish(
        resolved=resolved,
        data=data,
        media_type="image",
        parser="image",
        title=None,
        page_count=None,
        segments=[EncodedImage(data, width, height, None, "standalone")],
        limits=limits,
    )
