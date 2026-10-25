"""Vision captions for images whose OCR was not enough.

Returns text. Does not write SQLite and does not write ``data/images``.
The caller holds the Ollama lock for the stage. This function does not take
``ollama.lock``, because a nested flock would release the caller's lock.

Other models are unloaded first. A generator whose CONTEXT is not 4096 is
unloaded and loaded again by the chat request. Each image uses ``keep_alive``
30m. ``finally`` unloads the generator once for the whole batch, including
when a request fails. An empty ``content`` is retried once. Thinking text is
never used as the caption. The JPEG long side is 1280, which is the output
size sent to Ollama, not the pixel bomb cap. Smaller images are not enlarged.
"""

from __future__ import annotations

import base64
import io
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from ragcli.config import Config
from ragcli.errors import EXIT_UNCLASSIFIED, EXIT_USAGE, RagError
from ragcli.ollama import (
    KEEP_ALIVE_BATCH,
    VISION_READ_TIMEOUT_SECONDS,
    ChatMessage,
    available_bytes,
    choose_generator,
)
from ragcli.parse.pixels import sniff_header

logger = logging.getLogger("ragcli.images.caption")

SYSTEM_PROMPT = (
    "你是圖像描述器。只根據圖像內容用繁體中文寫一段描述，120字以內。"
    "如果圖入面有讀得到的文字，照實抄低。不要估計圖入面沒有出現的事實。不要輸出思考過程。"
)
USER_PROMPT = "描述這張圖，並保留圖中文字。"
VISION_LONG_SIDE = 1280
VISION_JPEG_QUALITY = 85
MAX_PIXELS = 20_000_000
CAPTION_DONE = "caption_done"
CAPTION_FAILED = "caption_failed"

_heif_ready = False


@dataclass(frozen=True, slots=True)
class VisionSettings:
    """Knobs for one vision batch. Defaults are the design's 9B/4B gate."""

    preferred: str = "qwen3.5:9b"
    fallback: str = "qwen3.5:4b"
    num_ctx: int = 4096
    num_predict: int = 400
    temperature: float = 0.1
    nine_b_min_bytes: int = 9_663_676_416
    four_b_min_bytes: int = 6_442_450_944
    allow_fallback: bool = True

    @staticmethod
    def from_config(config: Config, *, allow_fallback: bool = True) -> VisionSettings:
        """Copy the generate and memory fields. ``allow_fallback`` is the CLI flag."""
        if not isinstance(allow_fallback, bool):
            raise TypeError("allow_fallback 必須是布林值。")
        return VisionSettings(
            preferred=config.generate_model,
            fallback=config.generate_fallback_model,
            num_ctx=config.vision_num_ctx,
            num_predict=config.vision_num_predict,
            temperature=config.vision_temperature,
            nine_b_min_bytes=config.nine_b_min_bytes,
            four_b_min_bytes=config.four_b_min_bytes,
            allow_fallback=allow_fallback,
        )


@dataclass(frozen=True, slots=True)
class VisionImage:
    """PNG bytes already accepted by the parser. OCR keeps this resolution."""

    image_id: str
    png: bytes


@dataclass(frozen=True, slots=True)
class CaptionResult:
    """One image. ``text`` is None when there is nothing to store as a caption.

    ``model`` is None when the image never reached Ollama.
    """

    image_id: str
    status: str
    text: str | None
    model: str | None


def jpeg_for_vision(
    png: bytes,
    *,
    long_side: int = VISION_LONG_SIDE,
    quality: int = VISION_JPEG_QUALITY,
    max_pixels: int = MAX_PIXELS,
) -> bytes:
    """Return a JPEG whose long side is at most ``long_side``. Does not enlarge."""
    if isinstance(long_side, bool) or not isinstance(long_side, int) or long_side < 1:
        raise RagError(EXIT_USAGE, "視覺長邊必須是正整數。")
    if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 95:
        raise RagError(EXIT_USAGE, "JPEG 質素必須是 1 到 95。")
    frame = _open_raster(png, max_pixels)
    try:
        rgb = _to_rgb(frame)
        width, height = _fitted_size(rgb.size[0], rgb.size[1], long_side)
        target = rgb
        if (width, height) != rgb.size:
            target = rgb.resize((width, height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        target.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()
    finally:
        frame.close()


def describe_batch(
    images: Sequence[VisionImage],
    client,
    settings: VisionSettings | None = None,
    *,
    available: int | None = None,
) -> tuple[CaptionResult, ...]:
    """Caption each image with one generator, then unload that generator once.

    A decode failure becomes ``caption_failed`` for that image and does not
    load a model when every image fails. Connection and gate errors still
    propagate after the ``finally`` unload. The caller holds the Ollama lock.
    """
    chosen = settings if settings is not None else VisionSettings()
    if not isinstance(chosen, VisionSettings):
        raise TypeError("settings 必須是 VisionSettings。")
    batch = _validate(images)
    if not batch:
        return ()
    prepared = [_prepare(image) for image in batch]
    if all(item.jpeg is None for item in prepared):
        return tuple(item.failure for item in prepared if item.failure is not None)
    installed = client.tags()
    free = available_bytes() if available is None else available
    model = choose_generator(
        preferred=chosen.preferred,
        fallback=chosen.fallback,
        tags=installed,
        available=free,
        nine_b_min_bytes=chosen.nine_b_min_bytes,
        four_b_min_bytes=chosen.four_b_min_bytes,
        allow_fallback=chosen.allow_fallback,
    )
    logger.info("視覺描述 %s 張，模型 %s", sum(item.jpeg is not None for item in prepared), model)
    results: list[CaptionResult | None] = [None] * len(prepared)
    for index, item in enumerate(prepared):
        if item.failure is not None:
            results[index] = item.failure
    body_error: BaseException | None = None
    try:
        for index, item in enumerate(prepared):
            if item.jpeg is None or item.failure is not None:
                continue
            results[index] = _describe_one(client, model, chosen, item.image_id, item.jpeg)
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        _release(client, model, body_error)
    finished: list[CaptionResult] = []
    for result in results:
        if result is None:
            raise RagError(EXIT_UNCLASSIFIED, "視覺描述沒有完成。")
        finished.append(result)
    return tuple(finished)


@dataclass(frozen=True, slots=True)
class _Prepared:
    image_id: str
    jpeg: bytes | None
    failure: CaptionResult | None


def _prepare(image: VisionImage) -> _Prepared:
    try:
        jpeg = jpeg_for_vision(image.png)
    except RagError:
        logger.warning("圖像 %s 無法縮小。", image.image_id)
        return _Prepared(
            image.image_id,
            None,
            CaptionResult(image.image_id, CAPTION_FAILED, None, None),
        )
    return _Prepared(image.image_id, jpeg, None)


def _describe_one(
    client,
    model: str,
    settings: VisionSettings,
    image_id: str,
    jpeg: bytes,
) -> CaptionResult:
    encoded = base64.b64encode(jpeg).decode("ascii")
    messages = (
        ChatMessage("system", SYSTEM_PROMPT),
        ChatMessage("user", USER_PROMPT, (encoded,)),
    )
    for _attempt in range(2):
        result = client.chat(
            model,
            messages,
            num_ctx=settings.num_ctx,
            num_predict=settings.num_predict,
            temperature=settings.temperature,
            keep_alive=KEEP_ALIVE_BATCH,
            read_timeout=VISION_READ_TIMEOUT_SECONDS,
        )
        paragraph = " ".join(result.content.split())
        if paragraph:
            return CaptionResult(image_id, CAPTION_DONE, paragraph, model)
    logger.warning("圖像 %s 描述失敗。", image_id)
    return CaptionResult(image_id, CAPTION_FAILED, None, model)


def _release(client, model: str, body_error: BaseException | None) -> None:
    try:
        client.release_stage(model)
    except RagError:
        logger.error("卸載 %s 失敗。", model)
        if body_error is None:
            raise


def _validate(images: Sequence[VisionImage]) -> tuple[VisionImage, ...]:
    if isinstance(images, (str, bytes)) or not isinstance(images, Sequence):
        raise TypeError("images 必須是圖像清單。")
    checked: list[VisionImage] = []
    seen: set[str] = set()
    for image in images:
        if not isinstance(image, VisionImage):
            raise TypeError("images 必須是圖像清單。")
        image_id = image.image_id
        if (
            not isinstance(image_id, str)
            or image_id.strip() == ""
            or any(char.isspace() for char in image_id)
        ):
            raise ValueError("image_id 必須是不含空白的非空字串。")
        if image_id in seen:
            raise ValueError("image_id 重複。")
        seen.add(image_id)
        if not isinstance(image.png, (bytes, bytearray)):
            raise TypeError("圖像必須是位元組。")
        checked.append(image)
    return tuple(checked)


def _fitted_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    longest = max(width, height)
    if longest <= long_side:
        return width, height
    if width >= height:
        return long_side, max(1, height * long_side // width)
    return max(1, width * long_side // height), long_side


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


def _open_raster(image: bytes | bytearray, max_pixels: int) -> Image.Image:
    if isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or max_pixels < 1:
        raise RagError(EXIT_USAGE, "max_pixels 必須是正整數。")
    if not isinstance(image, (bytes, bytearray)) or len(image) == 0:
        raise RagError(EXIT_UNCLASSIFIED, "圖像無法讀取。")
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
            raise RagError(EXIT_UNCLASSIFIED, "圖像無法讀取。") from exc
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
