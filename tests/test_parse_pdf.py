"""PDF parser. Fixtures are built in the test with PyMuPDF."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from ragcli.errors import EXIT_DOCUMENT, RagError
from ragcli.parse import PARSER_VERSION, ParseLimits, parse_file
from ragcli.parse.pdf import plan_raster
from tests.parse_support import FONT, assert_text_chunks, put_text, solid_image, use_cjk

pytestmark = pytest.mark.skipif(not FONT.is_file(), reason="CJK font is not installed")


def _save(doc: pymupdf.Document, path: Path) -> None:
    doc.save(path)
    doc.close()


def test_plan_raster_keeps_200_dpi_for_a4() -> None:
    dpi, width, height = plan_raster(595, 842, max_pixels=20_000_000)
    assert dpi == 200
    assert width == 1653
    assert height == 2339
    assert width * height <= 20_000_000


def test_plan_raster_lowers_dpi_until_the_long_side_fits() -> None:
    dpi, width, height = plan_raster(2000, 2000, max_pixels=20_000_000)
    assert dpi == 108
    assert max(width, height) <= 3000
    assert width * height <= 20_000_000


def test_plan_raster_rejects_a_page_that_is_still_over_at_72_dpi() -> None:
    with pytest.raises(RagError) as exc:
        plan_raster(6000, 6000, max_pixels=20_000_000)
    assert exc.value.code == EXIT_DOCUMENT


def test_text_pdf_chunks_chinese_and_english(tmp_path: Path) -> None:
    path = tmp_path / "mixed.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    use_cjk(page)
    put_text(page, (72, 80), "營業額在香港同 London 都要保留原文內容。")
    doc.set_metadata({"title": " 季度報告 "})
    _save(doc, path)

    result = parse_file(path)
    assert result.document.media_type == "pdf"
    assert result.document.parser == "pymupdf"
    assert result.document.parser_version == PARSER_VERSION
    assert result.document.title == "季度報告"
    assert result.document.page_count == 1
    assert result.document.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.document.doc_id == result.document.sha256
    assert result.images == ()
    assert_text_chunks(result)
    assert any("營業額在香港同 London" in chunk.text for chunk in result.chunks)
    assert result.chunks[0].page_start == 1
    assert result.chunks[0].page_end == 1


def test_table_becomes_markdown_and_cells_are_not_repeated(tmp_path: Path) -> None:
    path = tmp_path / "table.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    use_cjk(page)
    xs = [72, 180, 300, 420]
    ys = [100, 140, 180, 220]
    shape = page.new_shape()
    for x in xs:
        shape.draw_line((x, ys[0]), (x, ys[-1]))
    for y in ys:
        shape.draw_line((xs[0], y), (xs[-1], y))
    shape.finish(color=(0, 0, 0), width=0.6)
    shape.commit()
    put_text(page, (72, 70), "表格之前的句子。")
    cells = [["名稱", "金額", "備註"], ["茶", "12", "熱"], ["咖啡", "30", "冰"]]
    for row, y in enumerate(ys[:-1]):
        for col, x in enumerate(xs[:-1]):
            put_text(page, (x + 4, y + 22), cells[row][col], size=11)
    put_text(page, (72, 260), "表格之後。")
    _save(doc, path)

    result = parse_file(path)
    text = "\n".join(chunk.text for chunk in result.chunks)
    assert "表格之前的句子。" in text
    assert "表格之後。" in text
    assert "| 茶 | 12 | 熱 |" in text
    assert "茶12熱" not in text
    assert text.count("茶") == 1
    assert result.images == ()


def test_embedded_image_is_pending_ocr_png(tmp_path: Path) -> None:
    path = tmp_path / "figure.pdf"
    png = solid_image(180, 120, (20, 40, 200))
    doc = pymupdf.open()
    page = doc.new_page()
    use_cjk(page)
    put_text(page, (72, 60), "這段文字夠長，所以這一頁不是掃描頁，請保留。")
    page.insert_image(pymupdf.Rect(72, 100, 252, 220), stream=png)
    _save(doc, path)

    result = parse_file(path)
    assert len(result.images) == 1
    image = result.images[0]
    assert image.kind == "embedded"
    assert image.status == "pending_ocr"
    assert image.route is None
    assert image.png_bytes is not None
    assert image.page == 1
    assert image.preceding_chunk_count == len(result.chunks)
    opened = Image.open(io.BytesIO(image.png_bytes))
    assert opened.mode == "RGB"
    assert opened.size == (180, 120)
    assert image.storage_relpath == f"images/{result.document.doc_id}/{image.image_id}.png"
    assert list(tmp_path.iterdir()) == [path]


def test_cmyk_embedded_image_is_stored_as_rgb(tmp_path: Path) -> None:
    path = tmp_path / "cmyk.pdf"
    buf = io.BytesIO()
    Image.new("CMYK", (160, 120), (10, 20, 30, 0)).save(buf, format="JPEG")
    doc = pymupdf.open()
    page = doc.new_page()
    use_cjk(page)
    put_text(page, (72, 60), "這段文字夠長，所以這一頁不是掃描頁，請保留。")
    page.insert_image(pymupdf.Rect(72, 90, 200, 180), stream=buf.getvalue())
    _save(doc, path)

    result = parse_file(path)
    assert len(result.images) == 1
    opened = Image.open(io.BytesIO(result.images[0].png_bytes or b""))
    assert opened.mode == "RGB"


def test_duplicate_embedded_images_keep_the_first(tmp_path: Path) -> None:
    path = tmp_path / "dup.pdf"
    png = solid_image(140, 100, (200, 10, 10))
    doc = pymupdf.open()
    page = doc.new_page()
    use_cjk(page)
    put_text(page, (72, 50), "這段文字夠長，所以這一頁不是掃描頁，請保留。")
    page.insert_image(pymupdf.Rect(72, 80, 212, 180), stream=png)
    page.insert_image(pymupdf.Rect(240, 80, 380, 180), stream=png)
    _save(doc, path)

    result = parse_file(path)
    assert len(result.images) == 2
    first, second = result.images
    assert first.status == "pending_ocr"
    assert first.duplicate_count == 1
    assert first.png_bytes is not None
    assert second.route == "skipped_duplicate"
    assert second.status == "skipped"
    assert second.png_bytes is None
    assert second.sha256 == first.sha256


def test_decorative_embedded_image_is_skipped_without_opening(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "icon.pdf"
    png = solid_image(8, 8, (1, 1, 1))
    doc = pymupdf.open()
    page = doc.new_page()
    use_cjk(page)
    put_text(page, (72, 60), "這段文字夠長，所以這一頁不是掃描頁，請保留。")
    page.insert_image(pymupdf.Rect(72, 90, 90, 108), stream=png)
    _save(doc, path)
    monkeypatch.setattr(
        "ragcli.parse.pixels.Image.open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("decoded")),
    )

    result = parse_file(path)
    assert len(result.images) == 1
    image = result.images[0]
    assert image.route == "skipped_small"
    assert image.status == "skipped"
    assert image.png_bytes is None
    assert result.chunks


def test_header_over_max_pixels_is_skipped_bomb_without_decode(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "bomb.pdf"
    png = solid_image(40, 30, (1, 2, 3))
    doc = pymupdf.open()
    page = doc.new_page()
    use_cjk(page)
    put_text(page, (72, 60), "這段文字夠長，所以這一頁不是掃描頁，請保留。")
    page.insert_image(pymupdf.Rect(72, 90, 140, 140), stream=png)
    xref = page.get_images(full=True)[0][0]
    doc.xref_set_key(xref, "Width", "8000")
    doc.xref_set_key(xref, "Height", "8000")
    _save(doc, path)

    def forbid(*_args, **_kwargs):
        raise AssertionError("decoded")

    monkeypatch.setattr(pymupdf.Document, "extract_image", forbid)
    monkeypatch.setattr("ragcli.parse.pixels.Image.open", forbid)

    result = parse_file(path)
    assert len(result.images) == 1
    image = result.images[0]
    assert image.width == 8000
    assert image.height == 8000
    assert image.route == "skipped_bomb"
    assert image.status == "skipped_bomb"
    assert image.png_bytes is None
    assert image.byte_size == 0
    assert result.chunks
    assert "掃描" not in "".join(chunk.text for chunk in result.chunks) or result.chunks


def test_scanned_page_is_one_page_render_and_not_an_embedding_model(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "scan.pdf"
    png = solid_image(400, 300, (240, 240, 240))
    doc = pymupdf.open()
    page = doc.new_page()
    use_cjk(page)
    put_text(page, (72, 40), "第1頁")
    page.insert_image(pymupdf.Rect(72, 80, 472, 380), stream=png)
    _save(doc, path)
    monkeypatch.setattr(
        pymupdf.Document,
        "extract_image",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("decoded")),
    )

    result = parse_file(path)
    assert [image.kind for image in result.images] == ["page_render"]
    image = result.images[0]
    assert image.status == "pending_ocr"
    assert image.png_bytes is not None
    assert image.width * image.height <= 20_000_000
    assert max(image.width, image.height) <= 3000
    opened = Image.open(io.BytesIO(image.png_bytes))
    assert opened.mode == "RGB"


def test_huge_page_does_not_call_get_pixmap(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "huge.pdf"
    doc = pymupdf.open()
    doc.new_page(width=6000, height=6000)
    _save(doc, path)
    monkeypatch.setattr(
        pymupdf.Page,
        "get_pixmap",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("rasterized")),
    )

    with pytest.raises(RagError) as exc:
        parse_file(path)
    assert exc.value.code == EXIT_DOCUMENT
    assert "72 DPI" in exc.value.message


def test_encrypted_pdf_is_exit_5(tmp_path: Path) -> None:
    path = tmp_path / "secret.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "secret")
    doc.save(
        path,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="secret",
    )
    doc.close()

    with pytest.raises(RagError) as exc:
        parse_file(path)
    assert exc.value.code == EXIT_DOCUMENT
    assert "encrypted" in exc.value.message


def test_page_limit_is_exit_5(tmp_path: Path) -> None:
    path = tmp_path / "pages.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    _save(doc, path)

    with pytest.raises(RagError) as exc:
        parse_file(path, ParseLimits(max_pages=1))
    assert exc.value.code == EXIT_DOCUMENT


def test_vector_page_renders_once_when_there_is_no_large_raster(tmp_path: Path) -> None:
    path = tmp_path / "vector.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    use_cjk(page)
    put_text(page, (72, 40), "這是一頁向量圖，用來測試合成頁面的規則。")
    for index in range(41):
        shape = page.new_shape()
        shape.draw_circle((80 + (index % 20) * 12, 80 + (index // 20) * 16), 3)
        shape.finish(color=(0, 0, 0), fill=(0, 0, 0))
        shape.commit()
    _save(doc, path)

    result = parse_file(path)
    assert [image.kind for image in result.images] == ["vector_render"]
    assert result.images[0].png_bytes is not None
    assert result.chunks


def test_vector_rule_does_not_fire_when_a_raster_needs_ocr(tmp_path: Path) -> None:
    path = tmp_path / "both.pdf"
    png = solid_image(120, 80, (0, 128, 0))
    doc = pymupdf.open()
    page = doc.new_page()
    use_cjk(page)
    put_text(page, (72, 40), "這是一頁向量圖，用來測試合成頁面的規則。")
    page.insert_image(pymupdf.Rect(72, 300, 220, 400), stream=png)
    for index in range(41):
        shape = page.new_shape()
        shape.draw_circle((80 + (index % 20) * 12, 80 + (index // 20) * 16), 3)
        shape.finish(color=(0, 0, 0), fill=(0, 0, 0))
        shape.commit()
    _save(doc, path)

    result = parse_file(path)
    assert [image.kind for image in result.images] == ["embedded"]


def test_corrupt_pdf_is_exit_5(tmp_path: Path) -> None:
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF-1.7\nnot a pdf")
    with pytest.raises(RagError) as exc:
        parse_file(path)
    assert exc.value.code == EXIT_DOCUMENT
