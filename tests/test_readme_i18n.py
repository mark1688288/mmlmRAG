"""README language link and the Traditional Chinese translation."""

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_ENGLISH = _ROOT / "README.md"
_CHINESE = _ROOT / "README.zh-Hant.md"
_BADGE_URLS = (
    "https://img.shields.io/badge/version-0.1.1-2F6F8F",
    "https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white",
    "https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=white",
    "https://img.shields.io/badge/Ollama-local-000000?logo=ollama&logoColor=white",
    "https://img.shields.io/badge/license-AGPL--3.0--only-1F6FEB",
)


def _centered_block(text: str) -> str:
    start = text.index('<div align="center">')
    end = text.index("</div>", start)
    return text[start:end]


def test_readmes_link_each_other_and_keep_the_version_badge():
    english = _ENGLISH.read_text(encoding="utf-8")
    assert _CHINESE.is_file()
    chinese = _CHINESE.read_text(encoding="utf-8")

    english_block = _centered_block(english)
    chinese_block = _centered_block(chinese)
    assert "[中文](README.zh-Hant.md)" in english_block
    assert "](README.zh-Hant.md)" in english
    assert english_block.index("version-0.1.1") < english_block.index(
        "[中文](README.zh-Hant.md)"
    )
    assert "[English](README.md)" in chinese_block
    assert "](README.md)" in chinese
    assert chinese_block.index("version-0.1.1") < chinese_block.index(
        "[English](README.md)"
    )
    assert "version-0.1.1" in english
    assert "version-0.1.1" in chinese
    for url in _BADGE_URLS:
        assert url in english
        assert url in chinese
