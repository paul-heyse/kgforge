"""Apply rendered docstrings to source files using LibCST."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast, overload

import libcst as cst
from libcst import (
    BaseSmallStatement,
    BaseStatement,
    FlattenSentinel,
    RemovalSentinel,
)

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
        docstring_stmt = cst.SimpleStatementLine(body=(expr,))

        body = node.body
        original_statements = cast(Sequence[cst.BaseStatement | BaseSmallStatement], body.body)

        def _as_statement(item: cst.BaseStatement | BaseSmallStatement) -> cst.BaseStatement:
            if isinstance(item, cst.BaseStatement):
                return item
            return cst.SimpleStatementLine(body=(item,))

        existing_statements: list[cst.BaseStatement] = [
            _as_statement(stmt) for stmt in original_statements
        ]

        if existing_statements and isinstance(existing_statements[0], cst.SimpleStatementLine):
            first = existing_statements[0]
            if (
                first.body
                and isinstance(first.body[0], cst.Expr)
                and isinstance(first.body[0].value, cst.SimpleString)
            ):
                current_value = cast(str, ast.literal_eval(first.body[0].value.value))
                if current_value == desired:
                    return node
                new_statements: list[cst.BaseStatement] = [
                    docstring_stmt,
                    *existing_statements[1:],
                ]
            else:
                new_statements = [docstring_stmt, *existing_statements]
        else:
            new_statements = [docstring_stmt, *existing_statements]

        if new_statements == existing_statements:
            return node
        self.changed = True
        new_body = body.with_changes(
            body=cast(tuple[cst.BaseStatement, ...], tuple(new_statements))
        )
        return node.with_changes(body=new_body)

    def visit_classdef(self, node: cst.ClassDef) -> bool:
        self.namespace.append(node.name.value)
        return True

    def leave_classdef(
        self, _original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> BaseStatement | FlattenSentinel[BaseStatement] | RemovalSentinel:
        qname = self._qualify(updated_node.name.value)
        self.namespace.pop()
        transformed = self._inject_docstring(updated_node, qname)
        return cast(BaseStatement, transformed)

    def visit_functiondef(self, node: cst.FunctionDef) -> bool:
        self.namespace.append(node.name.value)
        return True

    def leave_functiondef(
        self, _original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> BaseStatement | FlattenSentinel[BaseStatement] | RemovalSentinel:
        qname = self._qualify(updated_node.name.value)
        self.namespace.pop()
        transformed = self._inject_docstring(updated_node, qname)
        return cast(BaseStatement, transformed)


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
