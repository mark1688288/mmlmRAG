"""AST guard for interface copy.

Screen text has to be a ``t()`` call written inside a function. Han
characters outside that shape are allowed only on the list below.
Comments never reach the AST, so they are not interface copy.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src" / "ragcli"

_HELP_FUNCTIONS = frozenset({"build_parser", "_add_globals"})
_HELP_KEYWORDS = frozenset({"help", "description", "metavar"})
_LOG_METHODS = frozenset({"debug", "info", "warning", "error"})
_LOG_KEYWORDS = frozenset({"extra", "exc_info", "stack_info", "stacklevel"})
_INTERNAL_ERROR_FILES = frozenset(
    {
        "src/ragcli/chunking.py",
        "src/ragcli/images/route.py",
        "src/ragcli/images/caption.py",
        "src/ragcli/store/chroma.py",
    }
)

# Fixed text that is not an interface sentence. ``描述失敗。`` is an
# ``images.error`` value, same column as the pixel-overflow sentence.
# ``標題`` matches a Chinese Word heading style. The two code points are
# the CJK Unified Ideographs block in the token formula.
_ROUTE_CHUNK_TEXT = frozenset(
    {
        "圖像 OCR",
        "圖像",
        " 第",
        "頁",
        "描述：",
        "上文：",
        "下文：",
        "圖中文字：",
    }
)
_CJK_BLOCK_BOUNDS = frozenset({"\u4e00", "\u9fff"})
_STORED_PIXEL_OVERFLOW = "像素超過上限，未解碼。"
_STORED_CAPTION_FAILURE = "描述失敗。"
_DOCX_HEADING_STYLE = "標題"


@dataclass(frozen=True, slots=True)
class Violation:
    """One guard failure, with the file, line, and a short prefix."""

    path: str
    line: int
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.detail}"


def _has_han(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _prefix(text: str) -> str:
    flat = text.replace("\n", "\\n")
    if len(flat) > 40:
        return flat[:40] + "…"
    return flat


def _is_str_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _called_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_call_named(node: ast.AST, names: frozenset[str]) -> bool:
    return isinstance(node, ast.Call) and _called_name(node.func) in names


def _is_t_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "t"


def _docstring_ids(tree: ast.AST) -> set[int]:
    found: set[int] = set()

    def consider(body: list[ast.stmt]) -> None:
        if not body or not isinstance(body[0], ast.Expr):
            return
        value = body[0].value
        if _is_str_constant(value):
            found.add(id(value))

    def walk(node: ast.AST) -> None:
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            consider(node.body)
        for child in ast.iter_child_nodes(node):
            walk(child)

    walk(tree)
    return found


def _returns_only_t(func: ast.AST) -> bool:
    """True when every return in this function, ignoring nested defs, is ``t(...)``."""
    found: list[ast.Return] = []

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            if isinstance(child, ast.Return):
                found.append(child)
            walk(child)

    walk(func)
    return bool(found) and all(item.value is not None and _is_t_call(item.value) for item in found)


def _is_log_shape(call: ast.Call) -> bool:
    """``logger.warning("%s", t(...))`` with no further positional arguments."""
    if len(call.args) != 2 or any(keyword.arg not in _LOG_KEYWORDS for keyword in call.keywords):
        return False
    first, second = call.args
    return _is_str_constant(first) and first.value == "%s" and _is_t_call(second)


def _is_literal_template(call: ast.Call) -> bool:
    return bool(call.args) and _is_str_constant(call.args[0])


class _Guard(ast.NodeVisitor):
    def __init__(self, path: str, tree: ast.AST) -> None:
        self.path = path
        self.functions: list[str] = []
        self.violations: list[Violation] = []
        self._docstrings = _docstring_ids(tree)
        self._parents: dict[ast.AST, ast.AST] = {}
        self._functions_by_name: dict[str, list[ast.AST]] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                self._parents[child] = node
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._functions_by_name.setdefault(node.name, []).append(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit(node.args)
        self.functions.append("<lambda>")
        self.visit(node.body)
        self.functions.pop()

    def visit_Call(self, node: ast.Call) -> None:
        self._check_call(node)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and _has_han(node.value) and not self._han_allowed(node):
            shown = _prefix(node.value)
            self._add(node, f"Han literal outside the allowlist: {shown!r}")

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for param in node.type_params:
            self.visit(param)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        self.functions.append(node.name)
        for stmt in node.body:
            self.visit(stmt)
        self.functions.pop()

    def _check_call(self, node: ast.Call) -> None:
        if _is_t_call(node):
            self._check_t(node)
        if _is_call_named(node, frozenset({"RagError"})):
            self._check_rag_error(node)
        if _is_call_named(node, frozenset({"ArgumentTypeError"})):
            self._check_message_call(node, "ArgumentTypeError message must be a t() call")
        if _is_parser_error(node):
            self._check_message_call(node, "parser.error message must be a t() call")
        elif _is_log_call(node):
            self._check_log(node)
        if _is_call_named(node, frozenset({"print"})):
            self._check_print(node)
        self._check_help(node)

    def _check_t(self, node: ast.Call) -> None:
        if not self.functions:
            self._add(node, "module-level t() call")
        if not node.args or not _is_str_constant(node.args[0]):
            self._add(node, "t() id must be a string constant")

    def _check_rag_error(self, node: ast.Call) -> None:
        message = _keyword(node, "message")
        if message is None and len(node.args) >= 2:
            message = node.args[1]
        if message is None or not _is_t_call(message):
            self._add(node, "RagError message must be a t() call")

    def _check_message_call(self, node: ast.Call, detail: str) -> None:
        message = node.args[0] if node.args else _keyword(node, "message")
        if message is None or not _is_t_call(message):
            self._add(node, detail)

    def _check_log(self, node: ast.Call) -> None:
        assert isinstance(node.func, ast.Attribute)
        method = node.func.attr
        if method == "debug" and self._in_log_counts():
            if _is_log_shape(node) or _is_literal_template(node):
                return
            self._add(node, "logger.debug in _log_counts must use a literal template")
            return
        if not _is_log_shape(node):
            self._add(node, f'logger.{method}() must be "%s" plus t()')

    def _check_print(self, node: ast.Call) -> None:
        if not node.args:
            self._add(node, "print() must show t(), a catalog helper, exc.message, or code")
            return
        first = node.args[0]
        if _is_t_call(first) or _is_message_attribute(first) or _is_name(first, "code"):
            return
        if isinstance(first, ast.Call) and isinstance(first.func, ast.Name):
            definitions = self._functions_by_name.get(first.func.id, [])
            if definitions and all(_returns_only_t(item) for item in definitions):
                return
        self._add(node, "print() must show t(), a catalog helper, exc.message, or code")

    def _check_help(self, node: ast.Call) -> None:
        for keyword in node.keywords:
            if keyword.arg not in _HELP_KEYWORDS:
                continue
            if not _is_t_call(keyword.value):
                self._add(keyword.value, f"{keyword.arg}= must be a t() call")
                continue
            if not self.functions or self.functions[-1] not in _HELP_FUNCTIONS:
                self._add(
                    keyword.value,
                    f"{keyword.arg}= t() must be written inside build_parser or _add_globals",
                )

    def _han_allowed(self, node: ast.Constant) -> bool:
        assert isinstance(node.value, str)
        text = node.value
        if self.path == "src/ragcli/i18n/zh_Hant.py" or id(node) in self._docstrings:
            return True
        assigned = self._module_assign_name(node)
        if self.path == "src/ragcli/prompt.py" and assigned == "_SYSTEM_PROMPT":
            return True
        if self.path == "src/ragcli/images/caption.py" and assigned in {
            "SYSTEM_PROMPT",
            "USER_PROMPT",
        }:
            return True
        if self.path == "src/ragcli/parse/model.py" and text == _STORED_PIXEL_OVERFLOW:
            return True
        if self.path in _INTERNAL_ERROR_FILES and self._inside_internal_error(node):
            return True
        if self.path == "src/ragcli/images/route.py" and text in _ROUTE_CHUNK_TEXT:
            return True
        if self.path == "src/ragcli/ingest.py" and text == _STORED_CAPTION_FAILURE:
            return True
        if self.path == "src/ragcli/parse/docx.py" and text == _DOCX_HEADING_STYLE:
            return True
        return self.path == "src/ragcli/chunking.py" and text in _CJK_BLOCK_BOUNDS

    def _module_assign_name(self, node: ast.AST) -> str | None:
        current = node
        while current in self._parents:
            parent = self._parents[current]
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                return None
            if isinstance(parent, ast.Assign):
                if not isinstance(self._parents.get(parent), ast.Module):
                    return None
                if len(parent.targets) != 1 or not isinstance(parent.targets[0], ast.Name):
                    return None
                return parent.targets[0].id
            current = parent
        return None

    def _inside_internal_error(self, node: ast.AST) -> bool:
        current = node
        while current in self._parents:
            parent = self._parents[current]
            if _is_call_named(parent, frozenset({"ValueError", "TypeError"})):
                return True
            current = parent
        return False

    def _in_log_counts(self) -> bool:
        return (
            self.path == "src/ragcli/ollama.py"
            and bool(self.functions)
            and self.functions[-1] == "_log_counts"
        )

    def _add(self, node: ast.AST, detail: str) -> None:
        self.violations.append(Violation(self.path, getattr(node, "lineno", 1), detail))


def _is_parser_error(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "error"
        and isinstance(func.value, ast.Name)
        and func.value.id == "parser"
    )


def _is_log_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Attribute) and node.func.attr in _LOG_METHODS


def _is_message_attribute(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "message"


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _keyword(node: ast.Call, name: str) -> ast.AST | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def scan_tree(path: str, tree: ast.AST) -> list[Violation]:
    """Return guard failures for one parsed module."""
    guard = _Guard(path, tree)
    guard.visit(tree)
    return sorted(guard.violations, key=lambda item: (item.line, item.detail))


def scan_package() -> list[Violation]:
    """Scan every ``src/ragcli`` module. Tests are out of scope."""
    found: list[Violation] = []
    for path in sorted(_SRC.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        found.extend(scan_tree(path.relative_to(_ROOT).as_posix(), tree))
    return found


def _scan(source: str, path: str = "src/ragcli/cli.py") -> list[Violation]:
    return scan_tree(path, ast.parse(source))


def _details(source: str, path: str = "src/ragcli/cli.py") -> list[str]:
    return [item.detail for item in _scan(source, path)]


def test_package_has_no_leftover_interface_literals():
    paths = list(_SRC.rglob("*.py"))
    assert len(paths) >= 20
    found = scan_package()
    assert found == [], "\n".join(str(item) for item in found)


def test_guard_accepts_percent_s_logger_warning():
    source = """
def f(pieces):
    logger.warning("%s", t("log.example", pieces=pieces))
    logger.info("%s", t("log.example"))
    logger.error("%s", t("log.example"), stacklevel=2)
"""
    assert _scan(source) == []


def test_guard_rejects_logger_warning_of_t():
    source = """
def f():
    logger.warning(t("log.example"))
"""
    details = _details(source)
    assert any('logger.warning() must be "%s" plus t()' in item for item in details)


def test_guard_rejects_logger_argument_that_is_not_a_t_call():
    source = """
def f(translated, model):
    logger.warning("%s", translated)
    logger.warning(translated, model)
"""
    details = _details(source)
    assert len(details) == 2
    assert all('must be "%s" plus t()' in item for item in details)


def test_guard_rejects_module_level_t():
    source = """
LOCK_WAIT_MESSAGE = t("lock.wait")
PREFIX_WARNING = t("ask.prefix_warning")
"""
    details = _details(source)
    assert details == ["module-level t() call", "module-level t() call"]


def test_guard_rejects_t_default_and_variable_id():
    source = """
def frozen(message=t("lock.wait")):
    return message

def dynamic(name):
    return t(name)
"""
    details = _details(source)
    assert "module-level t() call" in details
    assert "t() id must be a string constant" in details


def test_guard_requires_rag_error_message_to_be_a_t_call():
    accepted = """
def f(path):
    raise RagError(2, t("config.file_missing", path=path))
    raise RagError(2, message=t("usage.file_not_directory"))
"""
    rejected = """
def f(message):
    raise RagError(2, message)
    raise RagError(2, "檔案已損壞，無法解析。")
"""
    assert _scan(accepted) == []
    details = _details(rejected)
    assert details.count("RagError message must be a t() call") == 2
    assert any(item.detail.startswith("Han literal") for item in _scan(rejected))


def test_guard_print_accepts_catalog_passthrough_and_helper():
    source = """
def _hit_line(hit):
    return t("status.hit", chunk_id=hit.chunk_id)

def f(exc, failure, code, hit):
    print(t("lock.wait"), file=sys.stderr)
    print(exc.message, file=sys.stderr)
    print(failure.message)
    print(code, file=sys.stderr)
    print(_hit_line(hit), file=sys.stderr)
"""
    assert _scan(source) == []


def test_guard_print_rejects_local_names_and_plain_helpers():
    source = """
MESSAGE = "Already ingested"

def plain(hit):
    return "hit"

def f(other, hit):
    print(other)
    print(MESSAGE)
    print(plain(hit))
"""
    details = _details(source)
    assert details.count("print() must show t(), a catalog helper, exc.message, or code") == 3


def test_guard_requires_argument_type_error_and_parser_error_to_call_t():
    accepted = """
def build_parser():
    def _positive_int(raw):
        raise argparse.ArgumentTypeError(t("usage.positive_int"))
    parser.error(t("usage.subcommand_missing"))
"""
    rejected = """
def build_parser():
    raise argparse.ArgumentTypeError("必須是正整數")
    parser.error("缺少子命令")
"""
    assert _scan(accepted) == []
    details = _details(rejected)
    assert "ArgumentTypeError message must be a t() call" in details
    assert "parser.error message must be a t() call" in details


def test_guard_help_text_must_be_built_in_the_parser_functions():
    accepted = """
def build_parser():
    parser = argparse.ArgumentParser(description=t("help.description"))
    parser.add_subparsers(metavar=t("help.metavar.command"))

def _add_globals(parser):
    parser.add_argument("--lang", help=t("help.lang"))
"""
    rejected = """
def _cmd_ingest():
    parser.add_argument("file", help=t("help.ingest.file"))

def build_parser():
    def inner():
        parser.add_argument("--force", help=t("help.ingest.force"))
"""
    literal = """
def build_parser():
    parser = argparse.ArgumentParser(description="本機混合語言 RAG。")
"""
    assert _scan(accepted) == []
    assert _details(rejected) == [
        "help= t() must be written inside build_parser or _add_globals",
        "help= t() must be written inside build_parser or _add_globals",
    ]
    details = _details(literal)
    assert "description= must be a t() call" in details
    assert any(item.startswith("Han literal") for item in details)


def test_guard_debug_template_is_only_exempt_inside_log_counts():
    exempt = """
def _log_counts(kind, payload):
    logger.debug("%s prompt_eval_count=%s eval_count=%s", kind, payload)
"""
    translated = """
def _run():
    logging.getLogger("ragcli").debug("%s", t("log.ui_language", language="en", source="default"))
"""
    other = """
def f(kind):
    logger.debug("%s prompt_eval_count=%s", kind)
"""
    assert _scan(exempt, "src/ragcli/ollama.py") == []
    assert _scan(translated) == []
    details = _details(other, "src/ragcli/ollama.py")
    assert details == ['logger.debug() must be "%s" plus t()']


def test_guard_han_allowlist_keeps_non_interface_text():
    assert _scan('"""本機混合語言 RAG 命令列。"""\n', "src/ragcli/__init__.py") == []
    assert _scan("# 這是註解\nx = 1\n") == []
    assert _scan('MESSAGES = {"ask.refusal": "資料中找不到答案。"}\n', "src/ragcli/i18n/zh_Hant.py") == []
    assert (
        _scan(
            '_SYSTEM_PROMPT = "你是本機文件助理。{refusal}"\n',
            "src/ragcli/prompt.py",
        )
        == []
    )
    header = _scan('HEADER = " 頁="\n', "src/ragcli/prompt.py")
    assert header
    assert "Han literal outside the allowlist:" in header[0].detail
    assert (
        _scan(
            'SYSTEM_PROMPT = "你是圖像描述器。"\nUSER_PROMPT = "描述這張圖。"\n',
            "src/ragcli/images/caption.py",
        )
        == []
    )
    assert (
        _scan(
            'error = "像素超過上限，未解碼。"\n',
            "src/ragcli/parse/model.py",
        )
        == []
    )
    assert (
        _scan(
            'def f():\n    raise ValueError("集合名稱不合法。")\n    raise TypeError(f"{name} 必須是整數。")\n',
            "src/ragcli/chunking.py",
        )
        == []
    )
    assert _scan('label = "圖像 OCR" if ocr else "圖像"\n', "src/ragcli/images/route.py") == []
    assert _scan('error = "描述失敗。"\n', "src/ragcli/ingest.py") == []
    assert _scan('name.startswith("標題")\n', "src/ragcli/parse/docx.py") == []
    assert (
        _scan(
            'def estimate_tokens(text):\n    return "\\u4e00" <= text <= "\\u9fff"\n',
            "src/ragcli/chunking.py",
        )
        == []
    )


def test_guard_rejects_excerpt_header_literals():
    literals = (
        "來源=",
        " 頁=",
        " 種類=",
        "問題：\n",
        "\n\n摘錄：\n",
    )
    for text in literals:
        found = _scan(
            f"def format_excerpt():\n    label = {text!r}\n",
            "src/ragcli/prompt.py",
        )
        assert found, repr(text)
        assert found[0].path == "src/ragcli/prompt.py"
        assert "Han literal outside the allowlist:" in found[0].detail
        assert (
            _scan(
                f"MESSAGES = {{'prompt.header': {text!r}}}\n",
                "src/ragcli/i18n/zh_Hant.py",
            )
            == []
        )


def test_guard_rejects_han_outside_the_allowlist():
    cases = [
        ('label = "還沒有進目錄。"\n', "src/ragcli/cli.py"),
        ('MESSAGES = {"ask.note": "不得譯成問題的語言"}\n', "src/ragcli/i18n/en.py"),
        ('error = "像素超過上限，未解碼。"\n', "src/ragcli/ingest.py"),
        ('error = "描述失敗。"\n', "src/ragcli/parse/model.py"),
        ('name.startswith("標題")\n', "src/ragcli/cli.py"),
        ('label = "一開始"\n', "src/ragcli/chunking.py"),
        ('def f():\n    raise ValueError("集合名稱不合法。")\n', "src/ragcli/cli.py"),
        ('note = "新的畫面句子。"\n', "src/ragcli/images/route.py"),
        ('def f():\n    SYSTEM_PROMPT = "新的畫面句子。"\n', "src/ragcli/images/caption.py"),
    ]
    for source, path in cases:
        found = _scan(source, path)
        assert found, path
        assert found[0].path == path
        assert "Han literal outside the allowlist:" in found[0].detail
        assert found[0].line >= 1
