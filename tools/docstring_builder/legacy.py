"""Legacy auto-docstring helpers backed by the docstring-builder pipelines."""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from tools.docstring_builder.apply import apply_edits
from tools.docstring_builder.config import DEFAULT_MARKER
from tools.docstring_builder.harvest import HarvestResult
from tools.docstring_builder.overrides import (
    _STANDARD_METHOD_EXTENDED_SUMMARIES,
    DEFAULT_MAGIC_METHOD_FALLBACK,
    DEFAULT_PYDANTIC_ARTIFACT_SUMMARY,
    MAGIC_METHOD_EXTENDED_SUMMARIES,
    PYDANTIC_ARTIFACT_SUMMARIES,
    QUALIFIED_NAME_OVERRIDES,
)
from tools.docstring_builder.overrides import (
    _is_magic as overrides_is_magic,
)
from tools.docstring_builder.overrides import (
    _is_pydantic_artifact as overrides_is_pydantic_artifact,
)
from tools.docstring_builder.overrides import (
    extended_summary as overrides_extended_summary,
)
from tools.docstring_builder.overrides import (
    summarize as overrides_summarize,
)
from tools.docstring_builder.schema import DocstringEdit

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"


@dataclass(slots=True)
class ParameterInfo:
    """Structure describing a callable parameter for docstring emission."""

    name: str
    annotation: str | None
    default: str | None
    kind: str
    required: bool

    @property
    def display_name(self) -> str:
        if self.kind == "vararg":
            return f"*{self.name}"
        if self.kind == "kwarg":
            return f"**{self.name}"
        return self.name

    @property
    def is_variadic(self) -> bool:
        return self.kind in {"vararg", "kwarg"}


@dataclass(slots=True)
class _CollectedSymbol:
    qname: str
    node: ast.AST
    kind: str
    docstring: str | None


class _SymbolCollector(ast.NodeVisitor):
    """Collect callable and class symbols for docstring generation."""

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self.namespace: list[str] = []
        self.symbols: list[_CollectedSymbol] = []

    def _qualify(self, name: str) -> str:
        return ".".join(part for part in [self.module_name, *self.namespace, name] if part)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qname = self._qualify(node.name)
        docstring = ast.get_docstring(node, clean=False)
        self.symbols.append(_CollectedSymbol(qname, node, "class", docstring))
        self.namespace.append(node.name)
        self.generic_visit(node)
        self.namespace.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        qname = self._qualify(node.name)
        docstring = ast.get_docstring(node, clean=False)
        self.symbols.append(_CollectedSymbol(qname, node, "function", docstring))
        self.namespace.append(node.name)
        self.generic_visit(node)
        self.namespace.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        qname = self._qualify(node.name)
        docstring = ast.get_docstring(node, clean=False)
        self.symbols.append(_CollectedSymbol(qname, node, "function", docstring))
        self.namespace.append(node.name)
        self.generic_visit(node)
        self.namespace.pop()


def annotation_to_text(annotation: ast.AST | None) -> str | None:
    """Convert an annotation node to source text."""
    if annotation is None:
        return None
    try:
        return ast.unparse(annotation)
    except AttributeError:  # pragma: no cover - Python <3.9 fallback
        return getattr(annotation, "id", None)


def _default_to_text(default: ast.AST | None) -> str | None:
    if default is None:
        return None
    try:
        return ast.unparse(default)
    except AttributeError:  # pragma: no cover - Python <3.9 fallback
        return getattr(default, "id", None)


def parameters_for(node: ast.AST) -> list[ParameterInfo]:
    """Return parameter metadata for function or method nodes."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    params: list[ParameterInfo] = []
    arguments = node.args

    def handle_parameters(
        items: list[ast.arg],
        defaults: list[ast.expr | None],
        *,
        kind: str,
    ) -> None:
        padding = len(items) - len(defaults)
        padded_defaults: list[ast.AST | None] = [None] * padding
        padded_defaults.extend(cast(ast.AST | None, default) for default in defaults)
        for arg, default in zip(items, padded_defaults, strict=True):
            params.append(
                ParameterInfo(
                    name=arg.arg,
                    annotation=annotation_to_text(arg.annotation),
                    default=_default_to_text(default),
                    kind=kind,
                    required=default is None and kind not in {"kwarg", "vararg"},
                )
            )

    handle_parameters(arguments.posonlyargs, [], kind="positional")
    handle_parameters(arguments.args, list(arguments.defaults), kind="positional")
    if arguments.vararg:
        params.append(
            ParameterInfo(
                name=arguments.vararg.arg,
                annotation=annotation_to_text(arguments.vararg.annotation),
                default=None,
                kind="vararg",
                required=False,
            )
        )
    handle_parameters(arguments.kwonlyargs, list(arguments.kw_defaults), kind="kw-only")
    if arguments.kwarg:
        params.append(
            ParameterInfo(
                name=arguments.kwarg.arg,
                annotation=annotation_to_text(arguments.kwarg.annotation),
                default=None,
                kind="kwarg",
                required=False,
            )
        )
    return params


def module_name_for(file_path: Path) -> str:
    """Return the dotted module path for a file."""
    resolved = file_path.resolve()
    if SRC_ROOT in resolved.parents or resolved == SRC_ROOT:
        relative = resolved.relative_to(SRC_ROOT)
    elif REPO_ROOT in resolved.parents or resolved == REPO_ROOT:
        relative = resolved.relative_to(REPO_ROOT)
    else:
        relative = resolved
    dotted = ".".join(relative.with_suffix("").parts)
    if dotted.endswith(".__init__"):
        dotted = dotted.rsplit(".", 1)[0]
    return dotted


def _normalize_qualified_name(name: str) -> str:
    """Canonicalise a fully-qualified name using the override catalogue."""
    base = name.split("[", 1)[0]
    return QUALIFIED_NAME_OVERRIDES.get(base, base)


def _wrap_text(text: str) -> list[str]:
    wrapped: list[str] = []
    for raw_paragraph in text.splitlines():
        paragraph = raw_paragraph.strip()
        if not paragraph:
            wrapped.append("")
            continue
        wrapped.append(
            textwrap.fill(paragraph, width=88, break_long_words=False, break_on_hyphens=False)
        )
    while wrapped and not wrapped[-1]:
        wrapped.pop()
    return wrapped


def _required_sections(  # noqa: PLR0913
    kind: str,
    parameters: list[ParameterInfo],
    returns: str | None,
    raises: list[str],
    *,
    name: str,
    is_public: bool,
) -> list[str]:
    """Return the ordered docstring section headers required for a symbol."""
    del name
    required: list[str] = []
    if parameters:
        required.append("Parameters")
    if returns and returns != "None":
        required.append("Returns")
    if raises:
        required.append("Raises")
    if kind == "function" and is_public:
        required.append("Examples")
    return required


def build_examples(  # noqa: PLR0913
    module_name: str,
    name: str,
    parameters: list[ParameterInfo],
    include_import: bool,
    *,
    is_async: bool = False,
    returns_value: bool = True,
) -> list[str]:
    """Generate doctest-style examples for a callable."""
    lines: list[str] = []
    if include_import and module_name:
        lines.append(f">>> from {module_name} import {name}")

    call_parts: list[str] = []
    trailing_parts: list[str] = []
    for param in parameters:
        if param.kind == "vararg":
            trailing_parts.append(f"*{param.name}")
        elif param.kind == "kwarg":
            trailing_parts.append(f"**{param.name}")
        elif param.required:
            call_parts.append("...")
    call_parts.extend(trailing_parts)
    call_fragment = ", ".join(call_parts)
    invocation = f"{name}({call_fragment})" if call_fragment else f"{name}()"

    if is_async:
        lines.append(f">>> result = {invocation}")
        lines.append(">>> result  # doctest: +ELLIPSIS")
        lines.append("...")
        return lines

    if returns_value:
        lines.append(f">>> result = {invocation}")
        lines.append(">>> result  # doctest: +ELLIPSIS")
    else:
        lines.append(f">>> {invocation}  # doctest: +ELLIPSIS")
    return lines


def _extract_exception_name(exc: ast.AST | None) -> str | None:
    """Return the exception name for a raise expression."""
    if exc is None:
        return None
    if isinstance(exc, ast.Name):
        return exc.id
    if isinstance(exc, ast.Attribute):
        return exc.attr
    if isinstance(exc, ast.Call):
        return _extract_exception_name(exc.func)
    return None


def _child_blocks(statement: ast.stmt) -> list[list[ast.stmt]]:
    """Return child statement blocks that should be inspected for raises."""
    if isinstance(statement, (ast.If, ast.For, ast.AsyncFor, ast.While)):
        return [statement.body, statement.orelse]
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        return [statement.body]
    if isinstance(statement, ast.Try):
        blocks = [statement.body, statement.orelse, statement.finalbody]
        blocks.extend(handler.body for handler in statement.handlers)
        return blocks
    if isinstance(statement, ast.Match):
        return [case.body for case in statement.cases]
    return []


def detect_raises(node: ast.AST) -> list[str]:
    """Detect top-level exceptions raised by a callable."""
    raises: list[str] = []
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return raises

    stack: list[ast.stmt] = list(reversed(node.body))
    skip_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

    while stack:
        current = stack.pop()
        if isinstance(current, ast.Raise):
            name = _extract_exception_name(current.exc)
            if name and name not in raises:
                raises.append(name)
            continue
        if isinstance(current, skip_types):
            continue
        for block in _child_blocks(current):
            stack.extend(reversed(block))

    return raises


def extended_summary(kind: str, name: str, module: str, node: ast.AST | None = None) -> str:
    """Return the extended summary paragraph for the symbol."""
    return overrides_extended_summary(kind, name, module, node)


def summarize(kind: str, name: str) -> str:
    """Return the short summary sentence for the symbol."""
    return overrides_summarize(name, kind)


def _is_magic(name: str) -> bool:
    """Return ``True`` when the provided callable is a Python magic method."""
    return overrides_is_magic(name)


def _is_pydantic_artifact(name: str) -> bool:
    """Return ``True`` when the provided name refers to a Pydantic helper."""
    return overrides_is_pydantic_artifact(name)


def build_docstring(kind: str, node: ast.AST, module_name: str) -> list[str]:
    """Build a NumPy style docstring as a list of lines."""
    name = getattr(node, "name", "")
    is_public = not name.startswith("_")
    params = parameters_for(node)
    returns_text = annotation_to_text(getattr(node, "returns", None))
    raises = detect_raises(node)
    summary = summarize(kind, name)
    extended = extended_summary(kind, name, module_name, node)
    is_async = isinstance(node, ast.AsyncFunctionDef)
    returns_value = returns_text not in {None, "None"}

    lines: list[str] = ['"""', summary]
    lines.append(DEFAULT_MARKER)

    if extended:
        lines.append("")
        lines.extend(_wrap_text(extended))

    if params:
        lines.append("")
        lines.append("Parameters")
        lines.append("----------")
        for param in params:
            annotation = param.annotation or "Any"
            optional = ", optional" if not param.required and not param.is_variadic else ""
            default = f", by default {param.default}" if param.default not in {None, "..."} else ""
            lines.append(f"{param.display_name} : {annotation}{optional}{default}")
            lines.append(f"    Description for ``{param.name}``.")

    if returns_value:
        lines.append("")
        lines.append("Returns")
        lines.append("-------")
        lines.append(returns_text or "Any")
        lines.append("    Describe the returned value.")

    if raises:
        lines.append("")
        lines.append("Raises")
        lines.append("------")
        for exc in raises:
            lines.append(exc)
            lines.append(f"    Raised when ``{exc}`` is encountered.")

    if kind == "function" and is_public:
        examples = build_examples(
            module_name,
            name,
            params,
            True,
            is_async=is_async,
            returns_value=returns_value,
        )
        if examples:
            lines.append("")
            lines.append("Examples")
            lines.append("--------")
            lines.append("")
            lines.extend(examples)

    lines.append('"""')
    return lines


def _lines_to_docstring_text(lines: list[str]) -> str:
    """Transform emitted docstring lines into the inner text payload."""
    opening = lines[0]
    if not opening.startswith('"""'):
        message = "Docstring must start with triple quotes."
        raise ValueError(message)
    core_lines = ["", *lines[1:-1]]
    text = "\n".join(core_lines)
    if not text.endswith("\n"):
        text += "\n"
    return text


def process_file(file_path: Path) -> bool:
    """Generate docstrings for the supplied module."""
    module_name = module_name_for(file_path)
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    collector = _SymbolCollector(module_name)
    collector.visit(tree)

    edits: list[DocstringEdit] = []
    for symbol in collector.symbols:
        if symbol.kind != "function":
            continue
        doc = symbol.docstring or ""
        if doc and DEFAULT_MARKER not in doc:
            continue
        doc_lines = build_docstring(symbol.kind, symbol.node, module_name)
        doc_text = _lines_to_docstring_text(doc_lines)
        edits.append(DocstringEdit(qname=symbol.qname, text=doc_text))

    if not edits:
        return False

    result = HarvestResult(module=module_name, filepath=file_path, symbols=[], cst_index={})
    changed, _ = apply_edits(result, edits, write=True)
    if changed:
        _ensure_trailing_blank_lines(file_path)
    return changed


def _ensure_trailing_blank_lines(file_path: Path) -> None:
    """Ensure generated docstrings leave a spacer line after the closing quotes."""
    lines = file_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return
    new_lines: list[str] = []
    pending_marker = False
    for index, line in enumerate(lines):
        new_lines.append(line)
        if DEFAULT_MARKER in line:
            pending_marker = True
        if line.strip() != '"""':
            continue
        if not pending_marker:
            continue
        next_index = index + 1
        if next_index >= len(lines) or lines[next_index].strip():
            new_lines.append("")
        pending_marker = False
    file_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_MAGIC_METHOD_FALLBACK",
    "DEFAULT_PYDANTIC_ARTIFACT_SUMMARY",
    "MAGIC_METHOD_EXTENDED_SUMMARIES",
    "PYDANTIC_ARTIFACT_SUMMARIES",
    "QUALIFIED_NAME_OVERRIDES",
    "_STANDARD_METHOD_EXTENDED_SUMMARIES",
    "_is_magic",
    "_is_pydantic_artifact",
    "_normalize_qualified_name",
    "_required_sections",
    "annotation_to_text",
    "build_docstring",
    "build_examples",
    "detect_raises",
    "extended_summary",
    "module_name_for",
    "parameters_for",
    "process_file",
    "summarize",
]
