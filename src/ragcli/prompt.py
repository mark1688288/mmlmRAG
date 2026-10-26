"""System prompt and the user message for one question.

The system text is fixed. ``think`` is not a setting, and this module does
not call Ollama or read the corpus. A missing page is omitted rather than
printed as -1. Citation brackets are instructions to the model; the program
does not parse them back out of the answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

REFUSAL_SENTENCE = "資料中找不到答案。"

SYSTEM_PROMPT = (
    "你是本機文件助理。只可以根據下方「摘錄」回答問題。\n"
    "規則：\n"
    "1. 如果摘錄不足以回答，只輸出這一句：資料中找不到答案。\n"
    "2. 不可以補充摘錄以外的事實，不可以猜測。\n"
    "3. 每個實質陳述都要在句末附上引用，格式為 "
    "[來源: {filename} p.{page} #{chunk_id}]。沒有頁碼就省略 p.N。\n"
    "4. 問題是英文就用英文回答；否則用繁體中文。\n"
    "5. 不要輸出思考過程。"
)


@dataclass(frozen=True, slots=True)
class Excerpt:
    """One retrieved chunk placed into the user message."""

    chunk_id: str
    filename: str
    page: int
    kind: str
    text: str
    distance: float


def is_refusal(text: str) -> bool:
    """True when the model followed the rule and said it cannot answer."""
    return isinstance(text, str) and text.strip() == REFUSAL_SENTENCE


def format_excerpt(index: int, excerpt: Excerpt) -> str:
    """One numbered block. Page -1 (and any page below 1) omits ``頁=``."""
    page = f" 頁={excerpt.page}" if excerpt.page >= 1 else ""
    header = (
        f"[{index}] 來源={excerpt.filename}{page} "
        f"chunk={excerpt.chunk_id} 種類={excerpt.kind}"
    )
    return f"{header}\n{excerpt.text}"


def format_user_message(question: str, excerpts: Sequence[Excerpt]) -> str:
    """Question plus the excerpts, numbered from 1 in the given order."""
    blocks = [
        format_excerpt(index, excerpt)
        for index, excerpt in enumerate(excerpts, start=1)
    ]
    return f"問題：\n{question}\n\n摘錄：\n" + "\n".join(blocks)
