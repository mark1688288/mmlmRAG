"""Vision OCR ordering, language, and the accurate level. No database and no Ollama."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from ragcli.config import load_config
from ragcli.errors import EXIT_UNCLASSIFIED, EXIT_USAGE, RagError
from ragcli.images.ocr import (
    CONFIDENCE_THRESHOLD,
    LANGUAGES,
    MAX_PIXELS,
    RECOGNITION_LEVEL,
    recognize,
)

REPO = Path(__file__).resolve().parents[1]


def solid_png(width: int = 32, height: int = 32) -> bytes:
    image = Image.new("RGB", (width, height), (255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def png_ihdr(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )


def test_defaults_match_rag_toml():
    config = load_config(REPO / "rag.toml")
    assert LANGUAGES == config.ocr_languages == ("zh-Hant", "zh-Hans", "en-US")
    assert RECOGNITION_LEVEL == config.ocr_recognition_level == "accurate"
    assert CONFIDENCE_THRESHOLD == config.ocr_confidence_threshold == 0.3
    assert MAX_PIXELS == config.max_pixels == 20_000_000


def test_lines_are_ordered_from_the_top_then_left():
    seen = {}

    def engine(image, *, languages, recognition_level, confidence_threshold):
        seen["languages"] = languages
        seen["level"] = recognition_level
        seen["threshold"] = confidence_threshold
        seen["size"] = image.size
        return [
            ("右下", 0.4, [0.6, 0.1, 0.2, 0.1]),
            ("右上", 0.6, [0.7, 0.8, 0.2, 0.1]),
            ("左上", 0.8, [0.1, 0.8, 0.2, 0.1]),
            ("略過", 0.2, [0.0, 0.9, 0.1, 0.1]),
        ]

    result = recognize(solid_png(), engine=engine)
    assert result.text == "左上\n右上\n右下"
    assert result.confidence == pytest.approx((0.8 + 0.6 + 0.4) / 3)
    assert seen["languages"] == ("zh-Hant", "zh-Hans", "en-US")
    assert seen["level"] == "accurate"
    assert seen["threshold"] == 0.3
    assert seen["size"] == (32, 32)


def test_same_row_is_left_to_right_and_the_threshold_keeps_an_equal_score():
    def engine(image, *, languages, recognition_level, confidence_threshold):
        return [
            ("右", 0.9, [0.4, 0.5, 0.1, 0.1]),
            ("左", 0.3, [0.1, 0.5, 0.1, 0.1]),
        ]

    result = recognize(solid_png(), engine=engine, confidence_threshold=0.3)
    assert result.text == "左\n右"
    assert result.confidence == pytest.approx(0.6)


def test_blank_observations_are_empty_ocr():
    def engine(image, *, languages, recognition_level, confidence_threshold):
        return [("   ", 0.9, [0.1, 0.2, 0.1, 0.1]), ("", 0.4, [0.2, 0.2, 0.1, 0.1])]

    result = recognize(solid_png(), engine=engine)
    assert result.text == ""
    assert result.confidence is None


def test_fast_level_does_not_call_the_engine():
    def engine(image, *, languages, recognition_level, confidence_threshold):
        raise AssertionError("engine")

    with pytest.raises(RagError) as exc:
        recognize(solid_png(), recognition_level="fast", engine=engine)
    assert exc.value.code == EXIT_USAGE


def test_engine_failure_is_exit_1_and_hides_the_detail():
    def engine(image, *, languages, recognition_level, confidence_threshold):
        raise RuntimeError("SECRET_OCR_IMAGE")

    with pytest.raises(RagError) as exc:
        recognize(solid_png(), engine=engine)
    assert exc.value.code == EXIT_UNCLASSIFIED
    assert exc.value.message == "OCR 失敗。"
    assert "SECRET_OCR_IMAGE" not in exc.value.message


def test_over_pixel_header_is_not_decoded(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("decoded")

    monkeypatch.setattr("ragcli.images.ocr.Image.open", boom)
    with pytest.raises(RagError) as exc:
        recognize(png_ihdr(20_000, 20_000), engine=boom)
    assert exc.value.code == EXIT_UNCLASSIFIED
    assert exc.value.message == "圖像超過像素上限。"


def test_truncated_png_stays_exit_1():
    with pytest.raises(RagError) as exc:
        recognize(b"\x89PNG\r\n\x1a\ntruncated")
    assert exc.value.code == EXIT_UNCLASSIFIED
    assert exc.value.message == "OCR 圖像無法讀取。"


def test_default_engine_uses_vision_accurate(monkeypatch):
    captured = {}

    class FakeOCR:
        def __init__(
            self,
            image,
            framework="vision",
            recognition_level="accurate",
            language_preference=None,
            confidence_threshold=0.0,
            detail=True,
            unit="token",
        ):
            captured["framework"] = framework
            captured["recognition_level"] = recognition_level
            captured["language_preference"] = language_preference
            captured["confidence_threshold"] = confidence_threshold
            captured["detail"] = detail

        def recognize(self, px=False):
            captured["px"] = px
            return [("營業額", 0.91, [0.2, 0.7, 0.3, 0.1])]

    import ocrmac.ocrmac as ocrmac_module

    monkeypatch.setattr(ocrmac_module, "OCR", FakeOCR)
    result = recognize(solid_png())
    assert captured["framework"] == "vision"
    assert captured["recognition_level"] == "accurate"
    assert captured["language_preference"] == ["zh-Hant", "zh-Hans", "en-US"]
    assert captured["confidence_threshold"] == 0.3
    assert captured["detail"] is True
    assert captured["px"] is False
    assert result.text == "營業額"
    assert result.confidence == pytest.approx(0.91)


def test_ocr_module_does_not_touch_storage_or_ollama():
    source = Path(recognize.__code__.co_filename).read_text(encoding="utf-8")
    for banned in ("sqlite", "ollama", "httpx", "chromadb", "data/images"):
        assert banned not in source
