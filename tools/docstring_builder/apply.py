"""Apply rendered docstrings to source files using LibCST."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

import libcst as cst
from tools.docstring_builder.harvest import HarvestResult
from tools.docstring_builder.schema import DocstringEdit


def _escape_docstring(text: str) -> str:
    normalized = text.replace("\r\n", "\n").rstrip("\n")
    escaped = normalized.replace('"""', '\\"""')
    return f"{escaped}\n"


@dataclass(slots=True)
class _DocstringTransformer(cst.CSTTransformer):
    module_name: str
    edits: Mapping[str, DocstringEdit]
    changed: bool = False

    def __post_init__(self) -> None:
        self.namespace: list[str] = []

    def _qualify(self, name: str) -> str:
        pieces = [self.module_name, *self.namespace, name]
        return ".".join(piece for piece in pieces if piece)

    def _inject_docstring(
        self, node: cst.FunctionDef | cst.ClassDef, qname: str
    ) -> cst.FunctionDef | cst.ClassDef:
        edit = self.edits.get(qname)
        if not edit:
            return node
        desired = _escape_docstring(edit.text)
        literal = cst.SimpleString(f'"""{desired}"""')
        expr = cst.Expr(value=literal)
        docstring_stmt: cst.BaseStatement = cst.SimpleStatementLine(body=[expr])
        body = node.body
        statements = cast(tuple[cst.BaseStatement, ...], body.body)
        updated_body: tuple[cst.BaseStatement, ...]
        if statements and isinstance(statements[0], cst.SimpleStatementLine):
            first = statements[0]
            if (
                first.body
                and isinstance(first.body[0], cst.Expr)
                and isinstance(first.body[0].value, cst.SimpleString)
            ):
                current_value = ast.literal_eval(first.body[0].value.value)
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
        return node.with_changes(body=body.with_changes(body=updated_body))

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:  # noqa: N802 - LibCST API contract
        self.namespace.append(node.name.value)
        return True

    def leave_ClassDef(  # noqa: N802 - LibCST API contract
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.BaseStatement:
        qname = self._qualify(original_node.name.value)
        updated = self._inject_docstring(updated_node, qname)
        self.namespace.pop()
        return updated

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:  # noqa: N802 - LibCST API contract
        self.namespace.append(node.name.value)
        return True

    def leave_FunctionDef(  # noqa: N802 - LibCST API contract
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.BaseStatement:
        qname = self._qualify(original_node.name.value)
        updated = self._inject_docstring(updated_node, qname)
        self.namespace.pop()
        return updated


def apply_edits(
    result: HarvestResult, edits: Iterable[DocstringEdit], write: bool = True
) -> tuple[bool, str | None]:
    """Apply docstring edits to the harvested file.

    Returns a tuple of ``(changed, code)`` where ``code`` contains the rendered text when ``write`` is
    ``False`` for dry-run mode.
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
