"""Message catalogs: lookup, placeholders, and import checks.

These tests do not change CLI output. Call sites still pass literal strings.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from ragcli.errors import RagError
from ragcli.i18n import (
    CatalogError,
    _as_text,
    _load_catalogs,
    _placeholder_names,
    _render,
    current_language,
    set_language,
    t,
)
from ragcli.i18n import en, zh_Hant

_MESSAGE_ID = re.compile(r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9_]*)+$")
_LANGUAGE_AT_IMPORT = current_language()
_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "ragcli" / "i18n"

EN_FILE = "This version expects a file path, not a directory."
ZH_FILE = "第一版要檔案路徑，不接受資料夾。"


@pytest.fixture(autouse=True)
def reset_language():
    set_language("en")
    yield
    set_language("en")


def test_default_language_is_english():
    assert _LANGUAGE_AT_IMPORT == "en"
    assert current_language() == "en"


def test_file_not_directory_in_both_languages():
    assert t("usage.file_not_directory") == EN_FILE
    set_language("zh")
    assert current_language() == "zh"
    assert t("usage.file_not_directory") == ZH_FILE
    set_language("en")
    assert t("usage.file_not_directory") == EN_FILE


def test_missing_message_id():
    with pytest.raises(CatalogError, match="missing message id 'usage.file_missing'"):
        t("usage.file_missing")


def test_unused_keyword_is_a_catalog_error():
    with pytest.raises(CatalogError, match="unused placeholder values: extra"):
        t("format.value", value="x", extra="y")


def test_missing_keyword_is_a_catalog_error():
    with pytest.raises(CatalogError, match="missing placeholder 'rest'"):
        t("format.user_rest", user="only")


@pytest.mark.parametrize(
    "template",
    ["{path!r}", "{path.__class__}", "{path:>8}", "}"],
)
def test_illegal_placeholders_are_catalog_errors(template: str):
    with pytest.raises(CatalogError):
        _placeholder_names(template)


def test_path_value_is_not_formatted_again():
    assert t("format.value", value=Path("a{b}%c")) == "a{b}%c"


def test_earlier_value_is_not_scanned_again():
    assert t("format.user_rest", user="{rest}", rest="owned") == "{rest} owned"


def test_value_private_use_and_braces_stay_literal():
    raw = "\ue000{{}}"
    assert t("format.value", value=raw) == raw


def test_template_escapes_do_not_touch_values():
    value = "\ue000{{}}"
    assert _render("{{{slot}}}", {"slot": value}) == "{" + value + "}"
    assert _placeholder_names("{{ok}}") == frozenset()
    assert _render("{{ok}}", {}) == "{ok}"


def test_catalogs_share_keys_placeholders_and_ids():
    assert set(en.MESSAGES) == set(zh_Hant.MESSAGES)
    assert en.MESSAGES["usage.file_not_directory"] == EN_FILE
    assert zh_Hant.MESSAGES["usage.file_not_directory"] == ZH_FILE
    assert en.MESSAGES["format.user_rest"] == "{user} {rest}"
    assert zh_Hant.MESSAGES["format.user_rest"] == "{user} {rest}"
    assert en.MESSAGES["format.value"] == "{value}"
    assert zh_Hant.MESSAGES["format.value"] == "{value}"
    for key in en.MESSAGES:
        assert _MESSAGE_ID.fullmatch(key)
        assert _placeholder_names(en.MESSAGES[key]) == _placeholder_names(
            zh_Hant.MESSAGES[key]
        )
    assert _placeholder_names(en.MESSAGES["format.user_rest"]) == frozenset(
        {"user", "rest"}
    )
    assert _placeholder_names(en.MESSAGES["format.value"]) == frozenset({"value"})
    assert _placeholder_names(en.MESSAGES["usage.file_not_directory"]) == frozenset()


def test_catalog_modules_only_assign_messages():
    for name in ("en.py", "zh_Hant.py"):
        tree = ast.parse((_PACKAGE / name).read_text(encoding="utf-8"))
        assert len(tree.body) == 1
        statement = tree.body[0]
        assert isinstance(statement, ast.Assign)
        assert len(statement.targets) == 1
        target = statement.targets[0]
        assert isinstance(target, ast.Name)
        assert target.id == "MESSAGES"
        assert isinstance(statement.value, ast.Dict)
        for key, value in zip(statement.value.keys, statement.value.values, strict=True):
            assert isinstance(key, ast.Constant) and isinstance(key.value, str)
            assert isinstance(value, ast.Constant) and isinstance(value.value, str)


def test_bool_is_rejected_before_int():
    with pytest.raises(CatalogError, match="bool is not a catalog value"):
        t("format.value", value=True)
    with pytest.raises(CatalogError, match="bool is not a catalog value"):
        t("format.value", value=False)
    assert t("format.value", value=0) == "0"
    assert t("format.value", value=1) == "1"
    assert t("format.value", value=1.5) == "1.5"
    with pytest.raises(CatalogError):
        t("format.value", value=None)

    tree = ast.parse((_PACKAGE / "__init__.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_as_text"
    )
    checks: list[str] = []
    for node in function.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Call)
            and isinstance(test.func, ast.Name)
            and test.func.id == "isinstance"
            and len(test.args) == 2
            and isinstance(test.args[1], ast.Name)
        ):
            continue
        checks.append(test.args[1].id)
    assert checks[0] == "bool"
    assert checks.index("bool") < checks.index("int")
    assert _as_text(1) == "1"


def test_empty_string_is_a_value():
    assert t("format.value", value="") == ""


def test_set_language_rejects_unknown_codes():
    set_language("zh")
    for language in ("fr", "zh ", "ZH", "zh-Hant", "en-US"):
        with pytest.raises(CatalogError):
            set_language(language)
        assert current_language() == "zh"


def test_message_id_is_positional_only():
    with pytest.raises(TypeError):
        t(message_id="usage.file_not_directory")


def test_catalog_error_is_not_a_rag_error():
    assert issubclass(CatalogError, Exception)
    assert not issubclass(CatalogError, RagError)


def test_import_check_rejects_bad_catalogs():
    good = {"usage.file_not_directory": "x"}
    with pytest.raises(CatalogError, match="catalog keys differ"):
        _load_catalogs(good, {})
    with pytest.raises(CatalogError, match="placeholders differ"):
        _load_catalogs(
            {"usage.file_not_directory": "see {path}"},
            {"usage.file_not_directory": "see"},
        )
    with pytest.raises(CatalogError, match="invalid placeholder 'path!r'"):
        _load_catalogs(
            {"usage.file_not_directory": "{path!r}"},
            {"usage.file_not_directory": "{path!r}"},
        )
    with pytest.raises(CatalogError, match="invalid placeholder 'path.__class__'"):
        _load_catalogs(
            {"usage.file_not_directory": "{path.__class__}"},
            {"usage.file_not_directory": "{path.__class__}"},
        )
    with pytest.raises(CatalogError, match="invalid message id 'usage'"):
        _load_catalogs({"usage": "x"}, {"usage": "x"})


def test_catalogs_are_frozen():
    with pytest.raises(TypeError):
        en.MESSAGES["format.value"] = "{value}"
    with pytest.raises(TypeError):
        zh_Hant.MESSAGES["usage.file_not_directory"] = ZH_FILE
