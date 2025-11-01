"""Apply rendered docstrings to source files using LibCST."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, Self, cast, overload

import libcst as cst

from tools.docstring_builder.harvest import HarvestResult
from tools.docstring_builder.schema import DocstringEdit


def _escape_docstring(text: str, indent: str) -> str:
    normalized = text.replace("\r\n", "\n").rstrip("\n")
    escaped = normalized.replace('"""', '\\"""')
    if not escaped:
        return "\n"
    lines = escaped.split("\n")
    if len(lines) == 1:
        return f"{lines[0]}\n"
    summary, *rest = lines
    indented_rest = [
        f"{indent}{line}" if line else ""  # Preserve blank lines without trailing spaces
        for line in rest
    ]
    joined = "\n".join([summary, *indented_rest])
    return f"{joined}\n"


class _SuiteProtocol(Protocol):
    body: tuple[cst.BaseStatement, ...]

    def with_changes(self, *, body: tuple[cst.BaseStatement, ...]) -> Self:
        """Return a copy of the suite with ``body`` replaced."""
        ...


@dataclass(slots=True)
class _DocstringTransformer(cst.CSTTransformer):
    module_name: str
    edits: Mapping[str, DocstringEdit]
    changed: bool = False

    def __post_init__(self) -> None:
        self.namespace: list[str] = []

    def _qualify(self, name: str) -> str:
        if self.namespace:
            pieces = [self.module_name, *self.namespace[:-1], name]
        else:
            pieces = [self.module_name, name]
        return ".".join(piece for piece in pieces if piece)

    @overload
    def _inject_docstring(self, node: cst.FunctionDef, qname: str) -> cst.FunctionDef: ...

    @overload
    def _inject_docstring(self, node: cst.ClassDef, qname: str) -> cst.ClassDef: ...

    def _inject_docstring(
        self, node: cst.FunctionDef | cst.ClassDef, qname: str
    ) -> cst.FunctionDef | cst.ClassDef:
        edit = self.edits.get(qname)
        if not edit:
            return node
        indent_level = max(len(self.namespace), 1)
        desired = _escape_docstring(edit.text, " " * 4 * indent_level)
        literal = cst.SimpleString(f'"""{desired}"""')
        expr = cst.Expr(value=literal)
        docstring_stmt: cst.BaseStatement = cst.SimpleStatementLine(body=[expr])
        body = node.body
        statements: tuple[cst.BaseStatement, ...] = tuple(body.body)
        updated_body: tuple[cst.BaseStatement, ...]
        if statements and isinstance(statements[0], cst.SimpleStatementLine):
            first = statements[0]
            if (
                first.body
                and isinstance(first.body[0], cst.Expr)
                and isinstance(first.body[0].value, cst.SimpleString)
            ):
                current_value = cast(str, ast.literal_eval(first.body[0].value.value))
                if current_value == desired:
                    return node
                updated_body = (docstring_stmt, *statements[1:])
            else:
                updated_body = (docstring_stmt, *statements)
        else:
            updated_body = (docstring_stmt, *statements)
        if updated_body == statements:
            return node
        self.changed = True
        suite = cast(_SuiteProtocol, node.body)
        updated_block = suite.with_changes(body=updated_body)
        updated_node_raw = node.with_changes(body=updated_block)  # type: ignore[arg-type]  # LibCST stubs lack precise Suite typing
        updated_node = updated_node_raw
        if isinstance(updated_node, cst.FunctionDef):
            return updated_node
        return updated_node

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:  # noqa: N802 - LibCST API contract
        self.namespace.append(node.name.value)
        return True

    def leave_ClassDef(  # noqa: N802 - LibCST API contract
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        qname = self._qualify(original_node.name.value)
        updated = self._inject_docstring(updated_node, qname)
        self.namespace.pop()
        return updated

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:  # noqa: N802 - LibCST API contract
        self.namespace.append(node.name.value)
        return True

    def leave_FunctionDef(  # noqa: N802 - LibCST API contract
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        qname = self._qualify(original_node.name.value)
        updated = self._inject_docstring(updated_node, qname)
        self.namespace.pop()
        return updated


def apply_edits(
    result: HarvestResult, edits: Iterable[DocstringEdit], write: bool = True
) -> tuple[bool, str | None]:
    """Apply docstring edits to the harvested file.

    Parameters
    ----------
    result : HarvestResult
        The harvested module metadata, including the target filepath.
    edits : Iterable[DocstringEdit]
        The sequence of docstring edits to apply to the module.
    write : bool, default True
        When ``True`` (default), persist the modified module back to disk. When ``False``,
        skip writing and return the rendered source code for inspection.

    Returns
    -------
    tuple[bool, str | None]
        A tuple ``(changed, code)`` where ``changed`` indicates whether any docstrings were
        updated. The ``code`` element contains the rendered module when ``write`` is ``False``;
        otherwise, it is ``None``.

    Raises
    ------
    libcst.ParserSyntaxError
        Raised when the harvested module cannot be parsed by LibCST.
    """
    mapping = {edit.qname: edit for edit in edits}
    if not mapping:
        return False, None
    original = result.filepath.read_text(encoding="utf-8")
    module = cst.parse_module(original)
    transformer = _DocstringTransformer(module_name=result.module, edits=mapping)
    new_module = module.visit(transformer)
    code = new_module.code
    if transformer.changed and write:
        result.filepath.write_text(code, encoding="utf-8")
    return transformer.changed, (code if not write else None)


__all__ = ["apply_edits"]
