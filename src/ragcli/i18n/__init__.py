"""Message catalogs for the rag command.

Call sites still pass their own strings. `t` reads the catalog for the
active language and fills named placeholders once, left to right.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from ragcli.i18n import en, zh_Hant

_MESSAGE_ID = re.compile(r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9_]*)+$")
_PLACEHOLDER = re.compile(r"^[a-z][a-z0-9_]*$")

__all__ = ["CatalogError", "current_language", "set_language", "t"]


class CatalogError(Exception):
    """A broken catalog or a bad `t` call. This is not a RagError."""


def _walk(template: str) -> list[tuple[str, str]]:
    """Split a template into `("lit", text)` and `("ph", name)` pieces.

    `{{` and `}}` are read from the template only. Placeholder text is not
    scanned, and this pass does not rewrite the template through stand-in
    characters.
    """
    parts: list[tuple[str, str]] = []
    literal: list[str] = []

    def flush() -> None:
        if literal:
            parts.append(("lit", "".join(literal)))
            literal.clear()

    index = 0
    length = len(template)
    while index < length:
        if template.startswith("{{", index):
            literal.append("{")
            index += 2
            continue
        if template.startswith("}}", index):
            literal.append("}")
            index += 2
            continue
        char = template[index]
        if char == "{":
            flush()
            end = template.find("}", index + 1)
            if end < 0:
                raise CatalogError("unclosed placeholder")
            name = template[index + 1 : end]
            if _PLACEHOLDER.fullmatch(name) is None:
                raise CatalogError(f"invalid placeholder {name!r}")
            parts.append(("ph", name))
            index = end + 1
            continue
        if char == "}":
            raise CatalogError("unmatched '}'")
        literal.append(char)
        index += 1
    flush()
    return parts


def _placeholder_names(template: str) -> frozenset[str]:
    return frozenset(name for kind, name in _walk(template) if kind == "ph")


def _render(template: str, values: Mapping[str, str]) -> str:
    """Fill placeholders in one left-to-right pass.

    Characters already copied into the output are not searched again.
    """
    used: set[str] = set()
    out: list[str] = []
    for kind, text in _walk(template):
        if kind == "lit":
            out.append(text)
            continue
        if text not in values:
            raise CatalogError(f"missing placeholder {text!r}")
        out.append(values[text])
        used.add(text)
    unused = sorted(set(values) - used)
    if unused:
        listed = ", ".join(unused)
        raise CatalogError(f"unused placeholder values: {listed}")
    return "".join(out)


def _as_text(value: object) -> str:
    # bool subclasses int, so this check has to run first.
    if isinstance(value, bool):
        raise CatalogError("bool is not a catalog value")
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    raise CatalogError(
        "catalog values must be str, int, float, or Path, "
        f"not {type(value).__name__}"
    )


def _compile(
    language: str, messages: object
) -> tuple[dict[str, str], dict[str, frozenset[str]]]:
    if not isinstance(messages, dict):
        raise CatalogError(f"{language} catalog is not a dict")
    templates: dict[str, str] = {}
    names: dict[str, frozenset[str]] = {}
    for key, template in messages.items():
        if not isinstance(key, str) or _MESSAGE_ID.fullmatch(key) is None:
            raise CatalogError(f"invalid message id {key!r}")
        if not isinstance(template, str):
            raise CatalogError(f"{language} {key!r} template is not a str")
        try:
            found = _placeholder_names(template)
        except CatalogError as exc:
            raise CatalogError(f"{language} {key}: {exc}") from None
        templates[key] = template
        names[key] = found
    return templates, names


def _load_catalogs(
    en_messages: object, zh_messages: object
) -> dict[str, Mapping[str, str]]:
    en_templates, en_names = _compile("en", en_messages)
    zh_templates, zh_names = _compile("zh", zh_messages)
    if set(en_templates) != set(zh_templates):
        only_en = ", ".join(sorted(set(en_templates) - set(zh_templates))) or "(none)"
        only_zh = ", ".join(sorted(set(zh_templates) - set(en_templates))) or "(none)"
        raise CatalogError(f"catalog keys differ: en only {only_en}; zh only {only_zh}")
    for key in en_templates:
        if en_names[key] != zh_names[key]:
            raise CatalogError(
                f"placeholders differ for {key!r}: "
                f"en {sorted(en_names[key])}; zh {sorted(zh_names[key])}"
            )
    return {
        "en": MappingProxyType(en_templates),
        "zh": MappingProxyType(zh_templates),
    }


_CATALOGS = _load_catalogs(en.MESSAGES, zh_Hant.MESSAGES)
en.MESSAGES = _CATALOGS["en"]
zh_Hant.MESSAGES = _CATALOGS["zh"]
_language = "en"


def set_language(language: str) -> None:
    """Select `en` or `zh` for this process. Any other value raises CatalogError."""
    global _language
    if language not in _CATALOGS:
        raise CatalogError("language must be 'en' or 'zh'")
    _language = language


def current_language() -> str:
    """Return the active language. Import starts at `en`."""
    return _language


def t(message_id: str, /, **kwargs: str | int | float | Path) -> str:
    """Return the active language's message, with each placeholder filled once.

    Bool is rejected before int. Paths are stringified after the type check.
    A missing id, a bad placeholder, or an unused keyword raises CatalogError.
    """
    try:
        template = _CATALOGS[_language][message_id]
    except KeyError:
        raise CatalogError(f"missing message id {message_id!r}") from None
    try:
        values = {key: _as_text(value) for key, value in kwargs.items()}
        return _render(template, values)
    except CatalogError as exc:
        raise CatalogError(f"{message_id}: {exc}") from None
