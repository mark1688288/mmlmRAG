"""Protected FTS spans, reciprocal rank fusion, and yes/no scores.

No Ollama and no corpus. Probe sentences are built in ``retrieve.py`` with
``chr`` so the source has no Han literals.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import pytest

from ragcli.retrieve import (
    RERANK_PROBE_PAIRS,
    RERANK_TEMPLATE,
    MergedHit,
    _PROBE_ZH_IRRELEVANT,
    _PROBE_ZH_QUERY,
    _PROBE_ZH_RELEVANT,
    _fts_phrase,
    fts_match_query,
    rerank_user,
    rrf_merge,
    rrf_score,
    template_accepts_rerank,
    yes_no_score,
)

_BODY = "根據第12條退款金額為100元。"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("第12條", '"第12條"'),
        ("請問第12條如何適用", '"第12條"'),
        ("金額是100元嗎", '"100"'),
        ("這筆100元怎麼退", '"100"'),
        ("1,200", '"1,200"'),
        ("qwen3.5", '"qwen3.5"'),
        ("19M2a", '"19M2a"'),
        ("GPT-4", '"GPT-4"'),
        ("第123條", '"第123條"'),
        ("退款", None),
        ("請找出退款金額", None),
        ("AI", None),
        ("稅", None),
        ("", None),
    ],
)
def test_fts_match_query_keeps_protected_spans_only(question, expected):
    assert fts_match_query(question) == expected


def test_clause_question_does_not_keep_the_surrounding_han():
    match = fts_match_query("請問第12條如何適用")
    assert match == '"第12條"'
    assert "如何適用" not in match
    amount = fts_match_query("金額是100元嗎")
    assert amount == '"100"'
    assert "金額是" not in amount


def test_protected_phrases_hit_a_memory_trigram_table():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE VIRTUAL TABLE docs USING fts5("
        "text, tokenize = \"trigram case_sensitive 0\")"
    )
    connection.execute("INSERT INTO docs (text) VALUES (?)", (_BODY,))
    for question in ("請問第12條如何適用", "金額是100元嗎"):
        match = fts_match_query(question)
        assert match is not None
        rows = connection.execute(
            "SELECT rowid FROM docs WHERE docs MATCH ?", (match,)
        ).fetchall()
        assert rows


def test_quoted_amounts_and_model_ids_are_legal_match_strings():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE VIRTUAL TABLE docs USING fts5("
        "text, tokenize = \"trigram case_sensitive 0\")"
    )
    connection.execute(
        "INSERT INTO docs (text) VALUES (?)",
        ("pay 1,200 for qwen3.5 and 19M2a",),
    )
    for question in ("1,200", "qwen3.5", "19M2a"):
        match = fts_match_query(question)
        assert match is not None
        assert match.startswith('"')
        rows = connection.execute(
            "SELECT rowid FROM docs WHERE docs MATCH ?", (match,)
        ).fetchall()
        assert rows
    with pytest.raises(sqlite3.OperationalError):
        connection.execute("SELECT rowid FROM docs WHERE docs MATCH ?", ("1,200",))


def test_spans_are_truncated_deduped_and_capped():
    assert fts_match_query("a" * 70) == '"' + ("a" * 64) + '"'
    assert fts_match_query("100 100") == '"100"'
    numbers = [str(100 + index) for index in range(9)]
    match = fts_match_query(" ".join(numbers))
    assert match is not None
    assert match.split(" OR ") == [f'"{number}"' for number in numbers[:8]]


def test_phrase_doubles_internal_quotes():
    assert _fts_phrase('ab"c') == '"ab""c"'


def test_rrf_example_one_orders_the_shared_hit_first():
    merged = rrf_merge(
        [("A", 0.1), ("B", 0.2), ("C", 0.3)],
        ["C", "D"],
        8,
    )
    assert merged == [
        MergedHit("C", 0.3, "both"),
        MergedHit("A", 0.1, "vector"),
        MergedHit("B", 0.2, "vector"),
        MergedHit("D", None, "fts"),
    ]
    assert rrf_score([3, 1]) == pytest.approx(1 / 63 + 1 / 61)
    assert rrf_score([2]) == pytest.approx(1 / 62)


def test_yes_no_score_needs_both_tokens_and_matches_softmax():
    assert yes_no_score((("yes", -0.1),)) is None
    assert yes_no_score((("Yes", -0.1), ("no", -1.0))) is None
    assert yes_no_score((("yes ", -0.1), ("no", -1.0))) is None
    yes = -0.2
    no = -1.5
    score = yes_no_score((("maybe", -3.0), ("yes", yes), ("no", no)))
    pivot = max(yes, no)
    expected = math.exp(yes - pivot) / (math.exp(yes - pivot) + math.exp(no - pivot))
    assert score == pytest.approx(expected)


def test_rerank_template_closes_thinking_in_nineteen_bytes():
    block = "<think>\n\n</think>\n\n"
    assert len(block.encode()) == 19
    assert "{{if not .Think}}" in RERANK_TEMPLATE
    assert block in RERANK_TEMPLATE
    assert template_accepts_rerank(RERANK_TEMPLATE) is True
    spaced = RERANK_TEMPLATE.replace("{{if not .Think}}", "{{ if not .Think }}")
    assert template_accepts_rerank(spaced) is True
    assert template_accepts_rerank(RERANK_TEMPLATE.replace("not .Think", "if .Think")) is False
    assert template_accepts_rerank(RERANK_TEMPLATE.replace("<think>\n\n</think>", "")) is False


def test_probe_pairs_are_code_points_and_the_user_text_has_no_think_block():
    assert _PROBE_ZH_QUERY == "法國首都"
    assert _PROBE_ZH_RELEVANT == "法國的首都是巴黎。"
    assert _PROBE_ZH_IRRELEVANT == "這份菜單有茶和飯。"
    assert RERANK_PROBE_PAIRS[0][0] == "capital of France"
    assert RERANK_PROBE_PAIRS[1][0] == _PROBE_ZH_QUERY
    source = (Path(__file__).resolve().parents[1] / "src" / "ragcli" / "retrieve.py").read_text(
        encoding="utf-8"
    )
    assert "法國首都" not in source
    assert "法國的首都是巴黎。" not in source
    assert "這份菜單有茶和飯。" not in source
    user = rerank_user(
        "Given a web search query, retrieve relevant passages that answer the query",
        "capital of France",
        "Paris is the capital of France.",
    )
    assert "<think>" not in user
    assert "</think>" not in user
    assert "<|im_start|>" not in user
    assert user.startswith("<Instruct>: ")


def test_rrf_example_two_keeps_the_tied_vector_hit_ahead_of_fts():
    vector = [("A", 0.1), ("B", 0.2)]
    assert rrf_merge(vector, ["F"], 8) == [
        MergedHit("A", 0.1, "vector"),
        MergedHit("F", None, "fts"),
        MergedHit("B", 0.2, "vector"),
    ]
    assert rrf_merge(vector, ["F"], 2) == [
        MergedHit("A", 0.1, "vector"),
        MergedHit("F", None, "fts"),
    ]
    assert rrf_score([1]) == pytest.approx(1 / 61)
