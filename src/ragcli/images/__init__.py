"""OCR, routing, and vision captions.

These modules return text and routes. They do not write SQLite and they do
not write ``data/images``.
"""

from ragcli.images.caption import (
    CAPTION_DONE,
    CAPTION_FAILED,
    SYSTEM_PROMPT,
    USER_PROMPT,
    VISION_JPEG_QUALITY,
    VISION_LONG_SIDE,
    CaptionResult,
    VisionImage,
    VisionSettings,
    describe_batch,
    jpeg_for_vision,
)
from ragcli.images.ocr import (
    CONFIDENCE_THRESHOLD,
    LANGUAGES,
    RECOGNITION_LEVEL,
    OcrResult,
    recognize,
)
from ragcli.images.route import (
    CONTEXT_CHARS,
    RICH_TOKENS,
    ROUTE_CAPTION_ONLY,
    ROUTE_LABEL_AND_CAPTION,
    ROUTE_TEXT_RICH,
    RoutedImage,
    SyntheticChunk,
    add_caption_line,
    compose,
    needs_vision,
    ocr_tokens,
    route_for,
    source_header,
)

__all__ = [
    "CAPTION_DONE",
    "CAPTION_FAILED",
    "CONFIDENCE_THRESHOLD",
    "CONTEXT_CHARS",
    "CaptionResult",
    "LANGUAGES",
    "OcrResult",
    "RECOGNITION_LEVEL",
    "RICH_TOKENS",
    "ROUTE_CAPTION_ONLY",
    "ROUTE_LABEL_AND_CAPTION",
    "ROUTE_TEXT_RICH",
    "RoutedImage",
    "SYSTEM_PROMPT",
    "SyntheticChunk",
    "USER_PROMPT",
    "VISION_JPEG_QUALITY",
    "VISION_LONG_SIDE",
    "VisionImage",
    "VisionSettings",
    "add_caption_line",
    "compose",
    "describe_batch",
    "jpeg_for_vision",
    "needs_vision",
    "ocr_tokens",
    "recognize",
    "route_for",
    "source_header",
]
