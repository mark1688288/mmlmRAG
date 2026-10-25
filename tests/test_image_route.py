"""OCR routes and the synthetic text templates. No disk and no Ollama."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragcli.chunking import chunk_ocr_body
from ragcli.config import load_config
from ragcli.images.route import (
    CONTEXT_CHARS,
    RICH_TOKENS,
    ROUTE_CAPTION_ONLY,
    ROUTE_LABEL_AND_CAPTION,
    ROUTE_TEXT_RICH,
    add_caption_line,
    compose,
    needs_vision,
    ocr_tokens,
    route_for,
    source_header,
)

REPO = Path(__file__).resolve().parents[1]
IMAGE = "abcdefabcdef:img:0001"
ANCHOR = "abcdefabcdef:000003"


def test_threshold_matches_rag_toml():
    config = load_config(REPO / "rag.toml")
    assert RICH_TOKENS == config.ocr_rich_tokens == 40
    assert CONTEXT_CHARS == 120


def test_zero_ten_and_forty_tokens_pick_the_design_routes():
    assert ocr_tokens("") == 0
    assert ocr_tokens(" \n\t") == 0
    assert route_for("") == ROUTE_CAPTION_ONLY
    assert route_for(" \n") == ROUTE_CAPTION_ONLY
    ten = "測" * 9
    assert ocr_tokens(ten) == 10
    assert route_for(ten) == ROUTE_LABEL_AND_CAPTION
    assert ocr_tokens("測" * 38) == 39
    assert route_for("測" * 38) == ROUTE_LABEL_AND_CAPTION
    forty = "測" * 39
    assert ocr_tokens(forty) == 40
    assert route_for(forty) == ROUTE_TEXT_RICH
    assert route_for("測" * 40) == ROUTE_TEXT_RICH
    assert ocr_tokens("a") == 1
    assert route_for("a") == ROUTE_LABEL_AND_CAPTION


def test_text_rich_repeats_the_ocr_header_and_is_pending():
    body = "".join(f"掃描列{index:03d}文字清楚。" for index in range(12))
    routed = compose(
        ROUTE_TEXT_RICH,
        body,
        filename="報告.pdf",
        page=2,
        image_id=IMAGE,
        before="這段不該進 OCR 區塊",
        after="這段也不該",
        anchor_chunk_id=ANCHOR,
        caption="不使用",
        target_tokens=36,
        max_tokens=52,
        overlap_tokens=6,
    )
    header = "[圖像 OCR] 報告.pdf 第2頁"
    direct = chunk_ocr_body(
        body,
        header=header,
        image_id=IMAGE,
        target_tokens=36,
        max_tokens=52,
        overlap_tokens=6,
    )
    assert routed.route == ROUTE_TEXT_RICH
    assert [chunk.text for chunk in routed.chunks] == [chunk.text for chunk in direct]
    assert len(routed.chunks) > 1
    assert needs_vision(ROUTE_TEXT_RICH) is False
    for chunk in routed.chunks:
        assert chunk.text.startswith(header + "\n")
        assert "上文：" not in chunk.text
        assert "描述：" not in chunk.text
        assert chunk.kind == "ocr"
        assert chunk.embed_state == "pending"
        assert chunk.image_id == IMAGE
        assert chunk.anchor_chunk_id == ANCHOR
        assert chunk.token_estimate <= 52


def test_figure_placeholder_and_caption_use_the_template():
    placeholder = compose(
        ROUTE_LABEL_AND_CAPTION,
        "營業額",
        filename="報告.pdf",
        page=2,
        image_id=IMAGE,
        before="去年同期較低",
        after="下一季回升",
        anchor_chunk_id=ANCHOR,
    )
    assert needs_vision(ROUTE_LABEL_AND_CAPTION) is True
    assert len(placeholder.chunks) == 1
    chunk = placeholder.chunks[0]
    assert chunk.kind == "figure"
    assert chunk.embed_state == "blocked"
    assert chunk.anchor_chunk_id == ANCHOR
    assert chunk.text == "\n".join(
        [
            "[圖像] 報告.pdf 第2頁",
            "上文：去年同期較低",
            "圖中文字：營業額",
            "下文：下一季回升",
        ]
    )
    finished = compose(
        ROUTE_LABEL_AND_CAPTION,
        "營業額",
        filename="報告.pdf",
        page=2,
        image_id=IMAGE,
        before="去年同期較低",
        after="下一季回升",
        anchor_chunk_id=ANCHOR,
        caption="一張長條圖",
    )
    assert finished.chunks[0].embed_state == "pending"
    assert finished.chunks[0].text == add_caption_line(chunk.text, "一張長條圖")
    assert finished.chunks[0].text == "\n".join(
        [
            "[圖像] 報告.pdf 第2頁",
            "上文：去年同期較低",
            "圖中文字：營業額",
            "描述：一張長條圖",
            "下文：下一季回升",
        ]
    )


def test_caption_only_has_no_ocr_line_until_a_description_exists():
    placeholder = compose(
        ROUTE_CAPTION_ONLY,
        "不該出現的文字",
        filename="報告.pdf",
        page=2,
        image_id=IMAGE,
        before="去年",
        after="成長",
    )
    assert needs_vision(ROUTE_CAPTION_ONLY) is True
    chunk = placeholder.chunks[0]
    assert chunk.kind == "caption"
    assert chunk.embed_state == "blocked"
    assert chunk.text == "\n".join(
        [
            "[圖像] 報告.pdf 第2頁",
            "上文：去年",
            "下文：成長",
        ]
    )
    assert "圖中文字" not in chunk.text
    assert "不該出現" not in chunk.text
    finished = compose(
        ROUTE_CAPTION_ONLY,
        "",
        filename="報告.pdf",
        page=2,
        image_id=IMAGE,
        before="去年",
        after="成長",
        caption="一張沒有文字的照片",
    )
    assert finished.chunks[0].embed_state == "pending"
    assert finished.chunks[0].text == "\n".join(
        [
            "[圖像] 報告.pdf 第2頁",
            "上文：去年",
            "描述：一張沒有文字的照片",
            "下文：成長",
        ]
    )


def test_context_is_120_characters_and_a_missing_page_is_omitted():
    before = ("A" * 50) + ("B" * 70) + ("C" * 80)
    after = ("D" * 80) + ("E" * 120)
    routed = compose(
        ROUTE_LABEL_AND_CAPTION,
        "甲",
        filename="scan.png",
        page=None,
        image_id=IMAGE,
        before=before,
        after=after,
    )
    lines = routed.chunks[0].text.split("\n")
    assert lines[0] == "[圖像] scan.png"
    assert lines[1] == "上文：" + ("B" * 40) + ("C" * 80)
    assert lines[-1] == "下文：" + ("D" * 80) + ("E" * 40)
    assert source_header("scan.png", None, ocr=True) == "[圖像 OCR] scan.png"
    assert source_header("報告.pdf", 2, ocr=True) == "[圖像 OCR] 報告.pdf 第2頁"


def test_blank_caption_stays_a_placeholder_and_newlines_stay_one_line():
    blank = compose(
        ROUTE_LABEL_AND_CAPTION,
        "甲",
        filename="a.png",
        page=1,
        image_id=IMAGE,
        caption=" \n ",
    )
    assert "描述：" not in blank.chunks[0].text
    assert blank.chunks[0].embed_state == "blocked"
    updated = add_caption_line(blank.chunks[0].text, "長條\n圖")
    assert "描述：長條 圖" in updated


def test_caption_line_is_inserted_before_the_template_footer():
    routed = compose(
        ROUTE_LABEL_AND_CAPTION,
        "甲\n下文：假",
        filename="a.png",
        page=1,
        image_id=IMAGE,
        before="上",
        after="成長",
    )
    updated = add_caption_line(routed.chunks[0].text, "說明")
    assert updated.index("下文：假") < updated.index("描述：說明")
    assert updated.endswith("描述：說明\n下文：成長")


def test_skipped_routes_do_not_need_vision_or_chunks():
    for route in ("skipped_small", "skipped_duplicate", "skipped_bomb"):
        assert needs_vision(route) is False
        with pytest.raises(ValueError, match="路由"):
            compose(route, "甲", filename="a.png", page=1, image_id=IMAGE)


def test_arguments_are_rejected():
    with pytest.raises(ValueError, match="page"):
        compose(ROUTE_CAPTION_ONLY, "", page=0, filename="a.png", image_id=IMAGE)
    with pytest.raises(ValueError, match="image_id"):
        compose(ROUTE_CAPTION_ONLY, "", filename="a.png", page=1, image_id="has space")
    with pytest.raises(ValueError, match="單行"):
        compose(ROUTE_CAPTION_ONLY, "", filename="a/b.png", page=1, image_id=IMAGE)
    with pytest.raises(ValueError, match="描述行已經存在"):
        add_caption_line("描述：已有\n下文：尾", "再加")
    with pytest.raises(TypeError, match="text"):
        route_for(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rich_tokens"):
        route_for("甲", rich_tokens=0)


def test_route_module_does_not_touch_storage_or_ollama():
    source = Path(compose.__code__.co_filename).read_text(encoding="utf-8")
    for banned in ("sqlite", "ollama", "httpx", "ocrmac", "chromadb"):
        assert banned not in source
