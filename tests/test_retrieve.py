"""Protected FTS spans and reciprocal rank fusion. No Ollama and no corpus."""

from __future__ import annotations

import sqlite3

import pytest

from ragcli.retrieve import MergedHit, _fts_phrase, fts_match_query, rrf_merge, rrf_score

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
