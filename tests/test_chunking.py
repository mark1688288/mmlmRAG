"""Chunk boundaries, token estimates, and OCR headers. No disk and no Ollama."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragcli.chunking import (
    MAX_TOKENS,
    OVERLAP_TOKENS,
    TARGET_TOKENS,
    TextChunk,
    chunk_ocr_body,
    chunk_text,
    estimate_tokens,
)
from ragcli.config import load_config

REPO = Path(__file__).resolve().parents[1]


def cover(original: str, chunks: list[TextChunk]) -> None:
    """Every chunk is a contiguous slice, in order, with no gap and no tail left.

    Repeated sentences can occur more than once. Accept the earliest slice that
    overlaps the cursor and still consumes new characters.
    """
    cursor = 0
    for chunk in chunks:
        search_from = 0
        placed = False
        while search_from <= len(original):
            start = original.find(chunk.text, search_from)
            if start == -1:
                break
            end = start + len(chunk.text)
            if start <= cursor < end:
                cursor = end
                placed = True
                break
            search_from = start + 1
        assert placed
    assert cursor == len(original)


def overlap_within(previous: str, nxt: str, budget: int) -> str:
    """Longest suffix of ``previous`` that starts ``nxt`` and fits in ``budget``."""
    limit = min(len(previous), len(nxt))
    best = 0
    right_hash = 0
    left_hash = 0
    power = 1
    base = 911
    mask = (1 << 64) - 1
    for size in range(1, limit + 1):
        right_hash = (right_hash * base + ord(nxt[size - 1])) & mask
        left_hash = (left_hash + ord(previous[-size]) * power) & mask
        power = (power * base) & mask
        if right_hash == left_hash and previous[-size:] == nxt[:size]:
            if estimate_tokens(nxt[:size]) <= budget:
                best = size
    return nxt[:best]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 1),
        ("a", 1),
        ("abc", 1),
        ("abcd", 2),
        ("abcde", 2),
        ("中", 2),
        ("中文", 3),
        ("中a", 2),
        ("。", 1),
        ("あ", 1),
        ("한", 1),
    ],
)
def test_estimate_tokens_follows_the_design_formula(text: str, expected: int):
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    assert estimate_tokens(text) == cjk + (len(text) - cjk) // 4 + 1 == expected


def test_defaults_match_the_design_and_config():
    config = load_config(REPO / "rag.toml")
    assert TARGET_TOKENS == config.chunk_target_tokens == 480
    assert MAX_TOKENS == config.chunk_max_tokens == 640
    assert OVERLAP_TOKENS == config.chunk_overlap_tokens == 80


def test_empty_and_whitespace_make_no_chunks():
    assert chunk_text("") == []
    assert chunk_text(" \n\t\r\n ") == []
    assert chunk_ocr_body(" \n\t", header="[圖像 OCR] a.png 第1頁", image_id="img1") == []


def test_short_text_stays_one_chunk():
    text = "退款期限是三十日。The refund window is thirty days."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].token_estimate == estimate_tokens(text)
    assert chunks[0].image_id is None
    assert chunks[0].token_estimate <= TARGET_TOKENS


def test_blank_line_is_preferred_over_sentence_punctuation():
    first = "本段說明退款期限。" * 3
    second = "本段說明換貨期限。" * 3
    opening = first + "\n\n"
    target = estimate_tokens(opening)
    both = estimate_tokens(opening + second)
    assert both > target
    chunks = chunk_text(
        opening + second,
        target_tokens=target,
        max_tokens=both,
        overlap_tokens=4,
    )
    assert len(chunks) >= 2
    assert "退款" in chunks[0].text
    assert "換貨" not in chunks[0].text
    assert chunks[0].text.count("。") == 3
    assert "換貨" in chunks[-1].text
    cover(opening + second, chunks)


def test_chinese_punctuation_packs_near_the_target():
    sentence = "編號一句內容清楚。"
    target = estimate_tokens(sentence * 2)
    text = sentence * 6
    chunks = chunk_text(
        text,
        target_tokens=target,
        max_tokens=estimate_tokens(sentence * 4),
        overlap_tokens=4,
    )
    assert len(chunks) > 1
    assert all(chunk.text.endswith("。") for chunk in chunks)
    assert all(chunk.token_estimate <= estimate_tokens(sentence * 4) for chunk in chunks)
    cover(text, chunks)
    for previous, nxt in zip(chunks, chunks[1:]):
        shared = overlap_within(previous.text, nxt.text, 4)
        assert shared != ""
        assert estimate_tokens(shared) <= 4


def test_english_punctuation_and_newlines():
    sentence = "Rule 12 states the refund window clearly. "
    text = sentence * 5
    target = estimate_tokens(sentence * 2)
    chunks = chunk_text(
        text.strip(),
        target_tokens=target,
        max_tokens=estimate_tokens(text),
        overlap_tokens=3,
    )
    assert len(chunks) > 1
    assert all(chunk.text.rstrip().endswith(".") for chunk in chunks)
    assert all("?" not in chunk.text for chunk in chunks)
    cover(text.strip(), chunks)

    shouted = "Is the window open? It is open! Done."
    shouted_chunks = chunk_text(
        shouted,
        target_tokens=estimate_tokens("Is the window open?"),
        max_tokens=estimate_tokens(shouted),
        overlap_tokens=2,
    )
    assert len(shouted_chunks) > 1
    assert shouted_chunks[0].text.endswith("?")
    cover(shouted, shouted_chunks)


def test_fullwidth_semicolon_splits_before_a_hard_cut():
    clause = "第一條件必須成立；"
    text = clause * 4
    target = estimate_tokens(clause * 2)
    chunks = chunk_text(
        text,
        target_tokens=target,
        max_tokens=estimate_tokens(text),
        overlap_tokens=3,
    )
    assert len(chunks) > 1
    assert all(chunk.text.endswith("；") for chunk in chunks)
    cover(text, chunks)


def test_decimal_point_stays_inside_the_number():
    first = "比率維持在 3.14 左右就好。"
    second = "下一句開始講別的事情。"
    text = first + second
    chunks = chunk_text(
        text,
        target_tokens=estimate_tokens(first),
        max_tokens=estimate_tokens(text),
        overlap_tokens=2,
    )
    assert any("3.14" in chunk.text for chunk in chunks)
    assert all("3.14" in chunk.text or "3." not in chunk.text for chunk in chunks)
    cover(text, chunks)


def test_mixed_language_uses_chinese_punctuation_before_english():
    chinese = "中文條款說明完畢。"
    english = "UNIQUEENGLISHWORD stays whole. "
    text = (chinese + english) * 4
    target = estimate_tokens(chinese + english)
    chunks = chunk_text(
        text.strip(),
        target_tokens=target,
        max_tokens=estimate_tokens(text),
        overlap_tokens=3,
    )
    assert len(chunks) > 1
    for chunk in chunks:
        if "UNIQUEENGLISHWORD" in chunk.text:
            assert "UNIQUEENGLISHWORD stays" in chunk.text
    cover(text.strip(), chunks)


def test_hard_cut_keeps_the_cap_when_overlap_is_tight():
    text = "".join(f"{index:03d}" for index in range(200))
    chunks = chunk_text(text, target_tokens=40, max_tokens=50, overlap_tokens=20)
    assert len(chunks) > 1
    assert all(chunk.token_estimate <= 50 for chunk in chunks)
    cover(text, chunks)
    for previous, nxt in zip(chunks, chunks[1:]):
        assert overlap_within(previous.text, nxt.text, 20) != ""


def test_zero_overlap_still_covers_the_text():
    text = "".join(f"{index:03d}" for index in range(80))
    chunks = chunk_text(text, target_tokens=30, max_tokens=40, overlap_tokens=0)
    assert len(chunks) > 1
    assert all(chunk.token_estimate <= 40 for chunk in chunks)
    cover(text, chunks)
    for previous, nxt in zip(chunks, chunks[1:]):
        assert overlap_within(previous.text, nxt.text, 1) == ""


def test_newlines_are_normalized_before_splitting():
    text = "第一段說明。\r\n\r\n第二段說明。"
    normalized = "第一段說明。\n\n第二段說明。"
    chunks = chunk_text(
        text,
        target_tokens=estimate_tokens("第一段說明。\n\n"),
        max_tokens=estimate_tokens(normalized),
        overlap_tokens=2,
    )
    assert all("\r" not in chunk.text for chunk in chunks)
    assert len(chunks) >= 2
    cover(normalized, chunks)


def test_acceptance_languages_overlap_and_hard_cap():
    chinese = "".join(f"第{index:04d}段{'漢' * 40}。" for index in range(80))
    english = " ".join(f"Sentence {index:04d} talks about refunds." for index in range(80))
    mixed = "".join(
        f"第{index:04d}條規定 refund item {index}. 必須保留{'憑' * 20}。"
        for index in range(80)
    )
    for text in (chinese, english, mixed):
        chunks = chunk_text(text)
        assert len(chunks) > 1
        cover(text, chunks)
        for chunk in chunks:
            assert chunk.token_estimate == estimate_tokens(chunk.text)
            assert chunk.token_estimate <= MAX_TOKENS
            assert chunk.image_id is None
        for previous, nxt in zip(chunks, chunks[1:]):
            shared = overlap_within(previous.text, nxt.text, OVERLAP_TOKENS)
            assert shared != ""
            assert estimate_tokens(shared) <= OVERLAP_TOKENS
            if estimate_tokens(previous.text) > OVERLAP_TOKENS:
                assert estimate_tokens(shared) == OVERLAP_TOKENS


def test_unsplittable_sentence_may_pass_the_target_but_not_the_maximum():
    sentence = "中" * 30
    assert TARGET_TOKENS > estimate_tokens(sentence)
    # No separator, over the target, under the hard maximum: one chunk.
    chunks = chunk_text(sentence, target_tokens=10, max_tokens=estimate_tokens(sentence), overlap_tokens=2)
    assert len(chunks) == 1
    assert chunks[0].text == sentence
    assert chunks[0].token_estimate > 10


def test_ocr_header_and_image_id_repeat_on_every_piece():
    header = "[圖像 OCR] 報告.pdf 第2頁"
    image_id = "c" * 16
    body = "".join(f"掃描列{index:03d}文字清楚。" for index in range(12))
    chunks = chunk_ocr_body(
        body,
        header=header,
        image_id=image_id,
        target_tokens=36,
        max_tokens=52,
        overlap_tokens=6,
    )
    assert len(chunks) > 1
    prefix = header + "\n"
    bodies: list[TextChunk] = []
    for chunk in chunks:
        assert chunk.text.startswith(prefix)
        assert chunk.text.count(header) == 1
        assert chunk.image_id == image_id
        assert image_id not in chunk.text
        assert chunk.token_estimate == estimate_tokens(chunk.text)
        assert chunk.token_estimate <= 52
        body_text = chunk.text.removeprefix(prefix)
        bodies.append(TextChunk(body_text, estimate_tokens(body_text)))
    cover(body, bodies)
    for previous, nxt in zip(bodies, bodies[1:]):
        assert overlap_within(previous.text, nxt.text, 6) != ""


def test_short_ocr_body_is_one_chunk():
    header = "[圖像 OCR] scan.png 第1頁"
    body = "圖中只有一行。"
    chunks = chunk_ocr_body(body, header=header, image_id="img_1")
    assert len(chunks) == 1
    assert chunks[0].text == f"{header}\n{body}"
    assert chunks[0].image_id == "img_1"
    assert chunks[0].token_estimate <= MAX_TOKENS


def test_ocr_header_that_cannot_fit_is_rejected():
    with pytest.raises(ValueError, match="硬上限"):
        chunk_ocr_body(
            "正文",
            header="標" * 40,
            image_id="img1",
            target_tokens=20,
            max_tokens=30,
            overlap_tokens=4,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"target_tokens": 0, "max_tokens": 10, "overlap_tokens": 0}, "target_tokens"),
        ({"target_tokens": 20, "max_tokens": 10, "overlap_tokens": 2}, "max_tokens"),
        ({"target_tokens": 20, "max_tokens": 30, "overlap_tokens": -1}, "overlap_tokens"),
        ({"target_tokens": 20, "max_tokens": 30, "overlap_tokens": 20}, "overlap_tokens"),
        ({"target_tokens": True, "max_tokens": 30, "overlap_tokens": 2}, "target_tokens"),
    ],
)
def test_budget_arguments_are_rejected(kwargs, match: str):
    with pytest.raises((TypeError, ValueError), match=match):
        chunk_text("文字。", **kwargs)


def test_header_and_identity_arguments_are_rejected():
    with pytest.raises(ValueError, match="單行"):
        chunk_ocr_body("正文", header="第一行\n第二行", image_id="img1")
    with pytest.raises(ValueError, match="image_id"):
        chunk_ocr_body("正文", header="[圖像 OCR] a.png 第1頁", image_id="has space")
    with pytest.raises(TypeError, match="text"):
        estimate_tokens(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="text"):
        chunk_text(b"bytes")  # type: ignore[arg-type]


def test_chunker_does_not_touch_storage_or_ollama():
    source = Path(chunk_text.__code__.co_filename).read_text(encoding="utf-8")
    assert "sqlite" not in source
    assert "ollama" not in source
    assert "open(" not in source
