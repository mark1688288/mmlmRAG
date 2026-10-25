"""Apple Vision OCR through ocrmac.

Returns text and a mean confidence. Does not write SQLite, does not write
``data/``, and does not call Ollama. Vision's bounding box origin is the
bottom left, so observations are ordered by ``y`` descending and then by
``x``. Chinese recognition stays on ``accurate``; ``fast`` cannot be
selected. A failure here is exit 1, including a pixel cap, so it is not the
parse rejection that rolls a document back.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from ragcli.errors import EXIT_UNCLASSIFIED, EXIT_USAGE, RagError
from ragcli.parse.pixels import sniff_header

LANGUAGES = ("zh-Hant", "zh-Hans", "en-US")
RECOGNITION_LEVEL = "accurate"
CONFIDENCE_THRESHOLD = 0.3
MAX_PIXELS = 20_000_000

_heif_ready = False


@dataclass(frozen=True, slots=True)
class OcrObservation:
    """One recognized string and the bottom-left origin of its box."""

    text: str
    confidence: float
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class OcrResult:
    """Joined text. ``confidence`` is None when nothing was kept."""

    text: str
    confidence: float | None


def recognize(
    image: bytes | bytearray | Image.Image,
    *,
    languages: Sequence[str] = LANGUAGES,
    recognition_level: str = RECOGNITION_LEVEL,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    max_pixels: int = MAX_PIXELS,
    engine: Callable[..., Sequence[object]] | None = None,
) -> OcrResult:
    """Read one image. ``engine`` defaults to ocrmac Vision and is for tests."""
    level = _level(recognition_level)
    chosen = _languages(languages)
    threshold = _threshold(confidence_threshold)
    _pixels(max_pixels)
    owns_raster = not isinstance(image, Image.Image)
    raster = image if isinstance(image, Image.Image) else _open_raster(image, max_pixels)
    caller = _ocrmac_engine if engine is None else engine
    try:
        try:
            raw = caller(
                raster,
                languages=chosen,
                recognition_level=level,
                confidence_threshold=threshold,
            )
        except RagError:
            raise
        except Exception as exc:
            raise RagError(EXIT_UNCLASSIFIED, "OCR 失敗。") from exc
    finally:
        if owns_raster:
            raster.close()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise RagError(EXIT_UNCLASSIFIED, "OCR 結果格式不正確。")
    kept: list[OcrObservation] = []
    for item in raw:
        observation = _observation(item, threshold)
        if observation is not None:
            kept.append(observation)
    return _join(kept)


def _ocrmac_engine(
    image: Image.Image,
    *,
    languages: Sequence[str],
    recognition_level: str,
    confidence_threshold: float,
) -> Sequence[object]:
    from ocrmac.ocrmac import OCR

    tool = OCR(
        image,
        framework="vision",
        recognition_level=recognition_level,
        language_preference=list(languages),
        confidence_threshold=confidence_threshold,
        detail=True,
    )
    return tool.recognize()


def _join(items: Sequence[OcrObservation]) -> OcrResult:
    ordered = tuple(sorted(items, key=lambda item: (-item.y, item.x)))
    if not ordered:
        return OcrResult("", None)
    mean = sum(item.confidence for item in ordered) / len(ordered)
    return OcrResult("\n".join(item.text for item in ordered), mean)


def _observation(item: object, threshold: float) -> OcrObservation | None:
    if not isinstance(item, (tuple, list)) or len(item) < 3:
        raise RagError(EXIT_UNCLASSIFIED, "OCR 結果格式不正確。")
    text, confidence, bbox = item[0], item[1], item[2]
    if not isinstance(text, str):
        raise RagError(EXIT_UNCLASSIFIED, "OCR 結果格式不正確。")
    score = _number(confidence)
    if score is None:
        raise RagError(EXIT_UNCLASSIFIED, "OCR 結果格式不正確。")
    if score < threshold:
        return None
    if not isinstance(bbox, (tuple, list)) or len(bbox) < 2:
        raise RagError(EXIT_UNCLASSIFIED, "OCR 結果格式不正確。")
    x = _number(bbox[0])
    y = _number(bbox[1])
    if x is None or y is None:
        raise RagError(EXIT_UNCLASSIFIED, "OCR 結果格式不正確。")
    cleaned = " ".join(text.split())
    if cleaned == "":
        return None
    return OcrObservation(cleaned, score, x, y)


def _number(value: object) -> float | None:
    """Finite float. Booleans are rejected. A non-number is a format error."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _open_raster(image: bytes | bytearray, max_pixels: int) -> Image.Image:
    if not isinstance(image, (bytes, bytearray)) or len(image) == 0:
        raise RagError(EXIT_UNCLASSIFIED, "OCR 圖像無法讀取。")
    raw = bytes(image)
    sniffed = _sniff(raw)
    if sniffed is not None and sniffed[0] * sniffed[1] > max_pixels:
        raise RagError(EXIT_UNCLASSIFIED, "圖像超過像素上限。")
    _register_heif()
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        try:
            with Image.open(io.BytesIO(raw)) as opened:
                opened.seek(0)
                opened.load()
                frame = opened.copy()
        except Image.DecompressionBombError as exc:
            raise RagError(EXIT_UNCLASSIFIED, "圖像超過像素上限。") from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise RagError(EXIT_UNCLASSIFIED, "OCR 圖像無法讀取。") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous
    if frame.size[0] < 1 or frame.size[1] < 1 or frame.size[0] * frame.size[1] > max_pixels:
        frame.close()
        raise RagError(EXIT_UNCLASSIFIED, "圖像超過像素上限。")
    return frame


def _sniff(raw: bytes) -> tuple[int, int] | None:
    try:
        return sniff_header(raw)
    except RagError:
        return None


def _register_heif() -> None:
    global _heif_ready
    if _heif_ready:
        return
    import pillow_heif

    pillow_heif.register_heif_opener()
    _heif_ready = True


def _level(value: str) -> str:
    if value != RECOGNITION_LEVEL:
        raise RagError(EXIT_USAGE, "ocr.recognition_level 第一版只可以是 accurate。")
    return value


def _languages(languages: Sequence[str]) -> tuple[str, ...]:
    if isinstance(languages, str) or not isinstance(languages, Sequence) or not languages:
        raise RagError(EXIT_USAGE, "ocr.languages 必須是非空字串清單。")
    items: list[str] = []
    for item in languages:
        if not isinstance(item, str) or item == "" or item != item.strip():
            raise RagError(EXIT_USAGE, "ocr.languages 必須是非空字串清單。")
        items.append(item)
    return tuple(items)


def _threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RagError(EXIT_USAGE, "ocr.confidence_threshold 必須是數字。")
    if value < 0 or value > 1:
        raise RagError(EXIT_USAGE, "ocr.confidence_threshold 必須在 0 到 1。")
    return float(value)


def _pixels(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RagError(EXIT_USAGE, "max_pixels 必須是正整數。")
