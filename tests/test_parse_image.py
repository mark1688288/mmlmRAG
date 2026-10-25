"""Standalone images. Header checks happen before Image.open."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from ragcli.errors import EXIT_DOCUMENT, RagError
from ragcli.parse import parse_file
from ragcli.parse.pixels import header_size, sniff_header
from tests.parse_support import bomb_png, heif_ispe, solid_image, write_image


def _forbid_open(*_args, **_kwargs):
    raise AssertionError("decoded")


def test_png_is_a_standalone_image(tmp_path: Path) -> None:
    path = write_image(tmp_path / "shot.png", 180, 120, (10, 20, 30), "PNG")
    result = parse_file(path)
    assert result.document.media_type == "image"
    assert result.document.parser == "image"
    assert result.chunks == ()
    assert len(result.images) == 1
    image = result.images[0]
    assert image.kind == "standalone"
    assert image.status == "pending_ocr"
    assert image.page is None
    assert image.preceding_chunk_count == 0
    assert image.png_bytes is not None
    assert Image.open(io.BytesIO(image.png_bytes)).size == (180, 120)
    assert list(tmp_path.iterdir()) == [path]


def test_long_side_over_3000_is_shrunk(tmp_path: Path) -> None:
    path = write_image(tmp_path / "wide.png", 4000, 1000, (2, 3, 4), "PNG")
    image = parse_file(path).images[0]
    assert image.png_bytes is not None
    assert image.width == 3000
    assert image.height == 750
    assert max(image.width, image.height) <= 3000


def test_header_over_max_pixels_is_exit_5_and_does_not_decode(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "bomb.png"
    path.write_bytes(bomb_png(8000, 5000))
    monkeypatch.setattr("ragcli.parse.pixels.Image.open", _forbid_open)
    before = set(tmp_path.iterdir())
    with pytest.raises(RagError) as exc:
        parse_file(path)
    assert exc.value.code == EXIT_DOCUMENT
    assert "像素" in exc.value.message
    assert set(tmp_path.iterdir()) == before


def test_decorative_png_is_skipped_without_opening(tmp_path: Path, monkeypatch) -> None:
    path = write_image(tmp_path / "icon.png", 8, 8, (9, 9, 9), "PNG")
    monkeypatch.setattr("ragcli.parse.pixels.Image.open", _forbid_open)
    result = parse_file(path)
    assert result.chunks == ()
    image = result.images[0]
    assert image.route == "skipped_small"
    assert image.status == "skipped"
    assert image.png_bytes is None
    assert image.width == 8
    assert image.height == 8


def test_gif_keeps_the_first_frame(tmp_path: Path) -> None:
    path = tmp_path / "anim.gif"
    frames = [
        Image.new("RGB", (120, 80), (255, 0, 0)),
        Image.new("RGB", (120, 80), (0, 0, 255)),
    ]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    image = parse_file(path).images[0]
    assert image.png_bytes is not None
    pixel = Image.open(io.BytesIO(image.png_bytes)).getpixel((0, 0))
    assert pixel[0] > 200
    assert pixel[2] < 30


def test_headers_match_the_encoded_canvas(tmp_path: Path) -> None:
    samples = {
        ".jpg": "JPEG",
        ".webp": "WEBP",
        ".bmp": "BMP",
        ".tif": "TIFF",
        ".gif": "GIF",
    }
    for suffix, fmt in samples.items():
        path = write_image(tmp_path / f"sample{suffix}", 17, 13, (8, 7, 6), fmt)
        assert header_size(path.read_bytes(), suffix) == (17, 13)


def test_webp_lossless_and_alpha_headers() -> None:
    lossless = io.BytesIO()
    Image.new("RGB", (17, 13), (1, 2, 3)).save(lossless, format="WEBP", lossless=True)
    alpha = io.BytesIO()
    Image.new("RGBA", (17, 13), (1, 2, 3, 4)).save(alpha, format="WEBP")
    assert header_size(lossless.getvalue(), ".webp") == (17, 13)
    assert header_size(alpha.getvalue(), ".webp") == (17, 13)


def test_heic_round_trip_uses_ispe_before_decode(tmp_path: Path) -> None:
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    path = tmp_path / "photo.heic"
    Image.new("RGB", (200, 100), (4, 5, 6)).save(path, format="HEIF")
    assert header_size(path.read_bytes(), ".heic") == (200, 100)
    image = parse_file(path).images[0]
    assert image.kind == "standalone"
    assert image.status == "pending_ocr"
    assert image.png_bytes is not None
    assert Image.open(io.BytesIO(image.png_bytes)).size == (200, 100)


def test_heic_ispe_over_the_cap_does_not_decode(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "bomb.heic"
    path.write_bytes(heif_ispe(8000, 5000))
    monkeypatch.setattr("ragcli.parse.pixels.Image.open", _forbid_open)
    with pytest.raises(RagError) as exc:
        parse_file(path)
    assert exc.value.code == EXIT_DOCUMENT
    assert sniff_header(path.read_bytes()) == (8000, 5000)


def test_too_many_images_is_exit_5(tmp_path: Path) -> None:
    import pymupdf

    from ragcli.parse import ParseLimits
    from tests.parse_support import put_text, use_cjk

    path = tmp_path / "many.pdf"
    png = solid_image(40, 40, (3, 3, 3))
    document = pymupdf.open()
    page = document.new_page()
    use_cjk(page)
    put_text(page, (72, 40), "這段文字夠長，所以這一頁不是掃描頁，請保留。")
    page.insert_image(pymupdf.Rect(72, 80, 120, 120), stream=png)
    page.insert_image(pymupdf.Rect(160, 80, 210, 120), stream=solid_image(40, 40, (9, 9, 9)))
    document.save(path)
    document.close()
    with pytest.raises(RagError) as exc:
        parse_file(path, ParseLimits(max_images=1))
    assert exc.value.code == EXIT_DOCUMENT