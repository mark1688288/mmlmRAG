"""Text and Markdown decoding. No images and no OCR."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragcli.errors import (
    EXIT_DOCUMENT,
    EXIT_NOT_FOUND,
    EXIT_UNSUPPORTED_TYPE,
    EXIT_USAGE,
    RagError,
)
from ragcli.parse import parse_file
from ragcli.parse.text import decode_text
from tests.parse_support import assert_text_chunks


def test_utf8_mixed_text_is_chunked(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("營業額在香港。\n\nRevenue stayed in London.\n", encoding="utf-8")
    result = parse_file(path)
    assert result.document.media_type == "text"
    assert result.document.parser == "text"
    assert result.document.page_count is None
    assert result.images == ()
    assert_text_chunks(result)
    joined = "\n".join(chunk.text for chunk in result.chunks)
    assert "營業額在香港。" in joined
    assert "Revenue stayed in London." in joined
    assert all(chunk.page_start is None for chunk in result.chunks)


def test_utf8_bom_is_stripped(tmp_path: Path) -> None:
    path = tmp_path / "bom.txt"
    path.write_bytes("\ufeff營業額".encode("utf-8"))
    result = parse_file(path)
    assert result.chunks[0].text == "營業額"


def test_big5_and_gb18030(tmp_path: Path) -> None:
    traditional = tmp_path / "big5.txt"
    traditional.write_bytes("繁體中文與 English".encode("big5"))
    simplified = tmp_path / "gb.txt"
    simplified.write_bytes("简体中文".encode("gb18030"))
    assert "繁體中文" in parse_file(traditional).chunks[0].text
    assert "English" in parse_file(traditional).chunks[0].text
    assert parse_file(simplified).chunks[0].text == "简体中文"


def test_empty_file_has_no_chunks(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    result = parse_file(path)
    assert result.chunks == ()
    assert result.images == ()
    assert result.document.byte_size == 0


def test_decode_order_prefers_utf8_over_later_encodings() -> None:
    assert decode_text("營業額".encode("utf-8")) == "營業額"


def test_undecodable_bytes_are_exit_5() -> None:
    with pytest.raises(RagError) as exc:
        decode_text(bytes(range(128, 256)) + b"\xff\xfe\x80\x81")
    assert exc.value.code == EXIT_DOCUMENT


def test_directory_missing_and_suffix(tmp_path: Path) -> None:
    with pytest.raises(RagError) as exc:
        parse_file(tmp_path)
    assert exc.value.code == EXIT_USAGE
    missing = tmp_path / "gone.pdf"
    with pytest.raises(RagError) as exc:
        parse_file(missing)
    assert exc.value.code == EXIT_NOT_FOUND
    html = tmp_path / "page.html"
    html.write_text("<p>no</p>", encoding="utf-8")
    with pytest.raises(RagError) as exc:
        parse_file(html)
    assert exc.value.code == EXIT_UNSUPPORTED_TYPE


def test_parser_sources_do_not_call_ocr_ollama_or_the_store() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "ragcli" / "parse"
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import ollama" not in text
        assert "ocrmac" not in text
        assert "chromadb" not in text
        assert "ragcli.store" not in text


def test_over_max_bytes_is_exit_5(tmp_path: Path) -> None:
    path = tmp_path / "big.txt"
    path.write_text("營業額", encoding="utf-8")
    from ragcli.parse import ParseLimits

    with pytest.raises(RagError) as exc:
        parse_file(path, ParseLimits(max_bytes=1))
    assert exc.value.code == EXIT_DOCUMENT
