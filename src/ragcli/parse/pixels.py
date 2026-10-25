"""Header sizes and PNG conversion.

Pixel caps are decided from the file header before ``Image.open``,
``extract_image``, or ``Pixmap``. Pillow's ``MAX_IMAGE_PIXELS`` is only the
second check. Decorative images are dropped from this decision and are not
decoded when the header already proves they are small.
"""

from __future__ import annotations

import io
import math

from PIL import Image, UnidentifiedImageError

from ragcli.errors import EXIT_DOCUMENT, RagError

DECOR_MIN_SIDE = 32
DECOR_BOTH_UNDER = 96
DECOR_MAX_BYTES = 8192
MAX_LONG_SIDE = 3000

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_JPEG_SOF = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
)
_HEIF_BRANDS = frozenset(
    {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"heif", b"mif1", b"msf1"}
)
_CONTAINERS = frozenset(
    {b"meta", b"iprp", b"ipco", b"moov", b"trak", b"mdia", b"minf", b"stbl"}
)

_heif_ready = False


class PixelOverflow(Exception):
    """The header or the decoder says the raster is over the cap."""


def is_decorative(width: int, height: int, byte_size: int) -> bool:
    """True when the image is too small to OCR or describe."""
    if width < DECOR_MIN_SIDE or height < DECOR_MIN_SIDE:
        return True
    return (
        width < DECOR_BOTH_UNDER
        and height < DECOR_BOTH_UNDER
        and byte_size < DECOR_MAX_BYTES
    )


def header_size(data: bytes, suffix: str) -> tuple[int, int]:
    """Read width and height for one standalone suffix. Does not decode pixels."""
    reader = _READERS.get(suffix)
    if reader is None:
        raise RagError(EXIT_DOCUMENT, "圖像格式無法在解碼前讀到尺寸。")
    return reader(data)


def sniff_header(data: bytes) -> tuple[int, int] | None:
    """Return header dimensions, or None when the bytes are not a known image."""
    if data.startswith(_PNG_SIG):
        return png_size(data)
    if data.startswith(b"\xff\xd8"):
        return jpeg_size(data)
    if data.startswith((b"GIF87a", b"GIF89a")):
        return gif_size(data)
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return webp_size(data)
    if data.startswith(b"BM"):
        return bmp_size(data)
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return tiff_size(data)
    if len(data) >= 12 and data[4:8] == b"ftyp" and _heif_brand(data):
        return heif_size(data)
    return None


def to_png(raw: bytes, *, max_pixels: int) -> tuple[bytes, int, int]:
    """Decode ``raw`` and return a PNG whose long side is at most 3000.

    The caller has already accepted the header size. GIF and HEIC contribute
    only the first frame. CMYK becomes RGB.
    """
    _register_heif()
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        try:
            with Image.open(io.BytesIO(raw)) as image:
                image.seek(0)
                working = _working_copy(image)
                working.load()
        except Image.DecompressionBombError as exc:
            raise PixelOverflow from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous
    width, height = working.size
    if width < 1 or height < 1 or width * height > max_pixels:
        raise PixelOverflow
    width, height, working = _shrink(working, width, height)
    out = io.BytesIO()
    working.save(out, format="PNG")
    return out.getvalue(), width, height


def png_size(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or not data.startswith(_PNG_SIG) or data[12:16] != b"IHDR":
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
    return _positive(data[16:20], data[20:24], "big")


def jpeg_size(data: bytes) -> tuple[int, int]:
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
    index = 2
    length = len(data)
    while index < length:
        if data[index] != 0xFF:
            raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
        index += 1
        while index < length and data[index] == 0xFF:
            index += 1
        if index >= length:
            break
        marker = data[index]
        index += 1
        if marker in (0xD8, 0xD9) or marker == 0x01 or 0xD0 <= marker <= 0xD7:
            if marker == 0xD9:
                break
            continue
        if index + 2 > length:
            break
        segment = int.from_bytes(data[index : index + 2], "big")
        if segment < 2 or index + segment > length:
            raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
        if marker == 0xDA:
            break
        if marker in _JPEG_SOF:
            if segment < 7:
                raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
            height = int.from_bytes(data[index + 3 : index + 5], "big")
            width = int.from_bytes(data[index + 5 : index + 7], "big")
            if width < 1 or height < 1:
                raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
            return width, height
        index += segment
    raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")


def gif_size(data: bytes) -> tuple[int, int]:
    """Logical screen size. Only the first frame is decoded later."""
    if len(data) < 10 or data[:6] not in (b"GIF87a", b"GIF89a"):
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
    return _positive(data[6:8], data[8:10], "little")


def webp_size(data: bytes) -> tuple[int, int]:
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
    chunk = data[12:16]
    if chunk == b"VP8X":
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
    elif chunk == b"VP8 ":
        if data[23:26] != b"\x9d\x01\x2a":
            raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
    elif chunk == b"VP8L":
        if data[20] != 0x2F:
            raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
        bits = int.from_bytes(data[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
    else:
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
    if width < 1 or height < 1:
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
    return width, height


def bmp_size(data: bytes) -> tuple[int, int]:
    if len(data) < 26 or data[:2] != b"BM":
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
    header = int.from_bytes(data[14:18], "little")
    if header == 12:
        width = int.from_bytes(data[18:20], "little")
        height = int.from_bytes(data[20:22], "little")
    else:
        width = int.from_bytes(data[18:22], "little", signed=True)
        height = abs(int.from_bytes(data[22:26], "little", signed=True))
    if width < 1 or height < 1:
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
    return width, height


def tiff_size(data: bytes) -> tuple[int, int]:
    """Width and height from the first IFD. Later frames are ignored."""
    if len(data) < 8:
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
    if data[:2] == b"II":
        order = "little"
    elif data[:2] == b"MM":
        order = "big"
    else:
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
    if int.from_bytes(data[2:4], order) != 42:
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
    offset = int.from_bytes(data[4:8], order)
    if offset < 0 or offset + 2 > len(data):
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
    count = int.from_bytes(data[offset : offset + 2], order)
    width = None
    height = None
    entry = offset + 2
    for _ in range(count):
        if entry + 12 > len(data):
            break
        tag = int.from_bytes(data[entry : entry + 2], order)
        kind = int.from_bytes(data[entry + 2 : entry + 4], order)
        number = int.from_bytes(data[entry + 4 : entry + 8], order)
        field = data[entry + 8 : entry + 12]
        if tag in (256, 257) and number >= 1:
            value = _tiff_inline(field, kind, order)
            if tag == 256:
                width = value
            else:
                height = value
        entry += 12
    if width is None or height is None or width < 1 or height < 1:
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
    return width, height


def heif_size(data: bytes) -> tuple[int, int]:
    """Largest ``ispe`` box. The largest canvas is the bomb check."""
    if len(data) < 12 or data[4:8] != b"ftyp" or not _heif_brand(data):
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
    found = _largest_ispe(data, 0, len(data))
    if found is None:
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
    return found


def _largest_ispe(data: bytes, start: int, end: int) -> tuple[int, int] | None:
    index = start
    best: tuple[int, int] | None = None
    best_area = -1
    while index + 8 <= end:
        size = int.from_bytes(data[index : index + 4], "big")
        box_type = data[index + 4 : index + 8]
        header = 8
        if size == 1:
            if index + 16 > end:
                break
            size = int.from_bytes(data[index + 8 : index + 16], "big")
            header = 16
        elif size == 0:
            size = end - index
        if size < header or index + size > end:
            break
        if box_type == b"ispe" and index + header + 12 <= index + size:
            payload = index + header
            width = int.from_bytes(data[payload + 4 : payload + 8], "big")
            height = int.from_bytes(data[payload + 8 : payload + 12], "big")
            area = width * height
            if width > 0 and height > 0 and area > best_area:
                best = (width, height)
                best_area = area
        elif box_type in _CONTAINERS:
            child = index + header + (4 if box_type == b"meta" else 0)
            inner = _largest_ispe(data, child, index + size)
            if inner is not None and inner[0] * inner[1] > best_area:
                best = inner
                best_area = inner[0] * inner[1]
        index += size
    return best


def _heif_brand(data: bytes) -> bool:
    brand = data[8:12]
    if brand in _HEIF_BRANDS:
        return True
    # Compatible brands follow the minor version.
    rest = data[16 : 16 + int.from_bytes(data[0:4], "big") - 16]
    return any(rest[i : i + 4] in _HEIF_BRANDS for i in range(0, len(rest) - 3, 4))


def _tiff_inline(field: bytes, kind: int, order: str) -> int:
    if kind == 3:
        return int.from_bytes(field[:2], order)
    if kind == 4:
        return int.from_bytes(field[:4], order)
    raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")


def _positive(width_bytes: bytes, height_bytes: bytes, order: str) -> tuple[int, int]:
    width = int.from_bytes(width_bytes, order)
    height = int.from_bytes(height_bytes, order)
    if width < 1 or height < 1:
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
    return width, height


def _working_copy(image: Image.Image) -> Image.Image:
    if image.mode == "CMYK":
        return image.convert("RGB")
    if image.mode in ("RGB", "RGBA", "L"):
        return image.copy()
    return image.convert("RGB")


def _shrink(image: Image.Image, width: int, height: int) -> tuple[int, int, Image.Image]:
    long_side = max(width, height)
    if long_side <= MAX_LONG_SIDE:
        return width, height, image
    scale = MAX_LONG_SIDE / long_side
    resized = (
        max(1, math.floor(width * scale)),
        max(1, math.floor(height * scale)),
    )
    if max(resized) > MAX_LONG_SIDE:
        raise RagError(EXIT_DOCUMENT, "檔案已損壞，無法解析。")
    return resized[0], resized[1], image.resize(resized, Image.Resampling.LANCZOS)


def _register_heif() -> None:
    global _heif_ready
    if _heif_ready:
        return
    import pillow_heif

    pillow_heif.register_heif_opener()
    _heif_ready = True


_READERS = {
    ".png": png_size,
    ".jpg": jpeg_size,
    ".jpeg": jpeg_size,
    ".gif": gif_size,
    ".webp": webp_size,
    ".bmp": bmp_size,
    ".tif": tiff_size,
    ".tiff": tiff_size,
    ".heic": heif_size,
    ".heif": heif_size,
}
