"""Semantic analysis to enrich harvested symbols with synthesized details."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from tools.docstring_builder.config import BuilderConfig
from tools.docstring_builder.harvest import HarvestResult, SymbolHarvest, parameter_display_name
from tools.docstring_builder.overrides import extended_summary as overrides_extended_summary
from tools.docstring_builder.schema import DocstringSchema, ParameterDoc, RaiseDoc, ReturnDoc


@dataclass(slots=True)
class SemanticResult:
    """Rich schema information for a harvested symbol."""

    symbol: SymbolHarvest
    schema: DocstringSchema


def _summary_for(symbol: SymbolHarvest, config: BuilderConfig) -> str:
    package = symbol.module.split(".")[0] if symbol.module else "package"
    verb = config.package_settings.summary_verbs.get(package, "Describe")
    target = symbol.qname.split(".")[-1].replace("_", " ")
    return f"{verb.capitalize()} {target}."


def _infer_optional(
    parameter: ParameterDoc, raw_annotation: str | None, default: str | None
) -> bool:
    if default is not None and default != "...":
        return True
    if raw_annotation and ("Optional" in raw_annotation or "None" in raw_annotation):
        return True
    return parameter.optional


def _describe_parameter(name: str, default: str | None, annotation: str | None) -> str:
    readable = name.replace("_", " ")
    lower = readable.lower()
    if annotation and "bool" in annotation.lower():
        base = f"Indicate whether {lower}."
    elif name.startswith("is_"):
        subject = readable.split(" ", 1)[1] if " " in readable else readable[3:]
        base = f"Indicate whether {subject or lower}."
    elif name.startswith("has_"):
        subject = readable.split(" ", 1)[1] if " " in readable else readable[4:]
        base = f"Indicate whether {subject or lower}."
    elif name.endswith("_id"):
        subject = readable[:-3].strip() or "resource"
        base = f"Identifier for the {subject}."
    elif name.endswith("_path"):
        subject = readable[:-5].strip() or "resource"
        base = f"Filesystem path for the {subject}."
    elif name.endswith("_url"):
        subject = readable[:-4].strip() or "resource"
        base = f"URL for the {subject}."
    else:
        base = f"Configure the {lower}."
    if default and default != "...":
        base += f" Defaults to ``{default}``."
    if not base.endswith("."):
        base += "."
    if base and not base[0].isupper():
        base = base[0].upper() + base[1:]
    return base


def _build_parameters(symbol: SymbolHarvest) -> list[ParameterDoc]:
    docs: list[ParameterDoc] = []
    for parameter in symbol.parameters:
        if parameter.name == "self":
            continue
        annotation = parameter.annotation
        display = parameter_display_name(parameter)
        description = _describe_parameter(parameter.name, parameter.default, annotation)
        doc = ParameterDoc(
            name=parameter.name,
            annotation=annotation,
            description=description,
            optional=False,
            default=parameter.default,
            display_name=display,
            kind=parameter.kind.name.lower(),
        )
        doc.optional = _infer_optional(doc, annotation, parameter.default)
        docs.append(doc)
    return docs


def _build_returns(symbol: SymbolHarvest) -> list[ReturnDoc]:
    if symbol.return_annotation in {None, "None"}:
        return []
    kind: Literal["returns", "yields"] = "yields" if symbol.is_generator else "returns"
    description = "TODO: describe return value."
    return [ReturnDoc(annotation=symbol.return_annotation, description=description, kind=kind)]


def _walk_raises(node: ast.AST) -> Iterable[tuple[str, str | None]]:
    for child in ast.walk(node):
        if isinstance(child, ast.Raise) and child.exc is not None:
            exc = child.exc
            name = _exception_name(exc)
            if not name:
                continue
            reason = _exception_reason(exc)
            yield name, reason


def _exception_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        return _exception_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _exception_reason(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call) and node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            message = first.value.strip()
            return message if message else None
        text = _stringify_expression(first)
        if text:
            return f"Raised when {text}."
    return None


def _stringify_expression(node: ast.AST) -> str | None:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - ast.unparse guard
        return None


def _build_raises(node: ast.AST | None) -> list[RaiseDoc]:
    if node is None:
        return []
    seen: set[tuple[str, str | None]] = set()
    docs: list[RaiseDoc] = []
    for name, reason in _walk_raises(node):
        key = (name, reason)
        if key in seen:
            continue
        seen.add(key)
        description = reason or f"Raised when ``{name}`` is triggered."
        if not description.endswith("."):
            description += "."
        docs.append(RaiseDoc(exception=name, description=description))
    return docs


def _ast_index(result: HarvestResult) -> dict[str, ast.AST]:
    tree = ast.parse(result.filepath.read_text(encoding="utf-8"))
    index: dict[str, ast.AST] = {}

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.namespace: list[str] = []

        def _qualify(self, name: str) -> str:
            return ".".join(part for part in [result.module, *self.namespace, name] if part)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            qname = self._qualify(node.name)
            index[qname] = node
            self.namespace.append(node.name)
            self.generic_visit(node)
            self.namespace.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            qname = self._qualify(node.name)
            index[qname] = node
            self.namespace.append(node.name)
            self.generic_visit(node)
            self.namespace.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            qname = self._qualify(node.name)
            index[qname] = node
            self.namespace.append(node.name)
            self.generic_visit(node)
            self.namespace.pop()

    _Visitor().visit(tree)
    return index


def build_semantic_schemas(result: HarvestResult, config: BuilderConfig) -> list[SemanticResult]:
    """Generate docstring schemas for the harvested symbols in a file."""
    ast_nodes = _ast_index(result)
    entries: list[SemanticResult] = []
    for symbol in result.symbols:
        if not symbol.owned and not config.normalize_sections:
            continue
        parameters = _build_parameters(symbol)
        returns = _build_returns(symbol)
        ast_node = ast_nodes.get(symbol.qname)
        raises = _build_raises(ast_node)
        notes: list[str] = []
        if symbol.is_async:
            notes.append("This coroutine executes asynchronously.")
        if symbol.is_generator:
            notes.append("This callable yields values instead of returning once.")
        simple_name = symbol.qname.split(".")[-1]
        extended = overrides_extended_summary(symbol.kind, simple_name, symbol.module, ast_node)
        schema = DocstringSchema(
            summary=_summary_for(symbol, config),
            extended=extended,
            parameters=parameters,
            returns=returns,
            raises=raises,
            notes=notes,
            see_also=[],
            examples=[],
        )
        entries.append(SemanticResult(symbol=symbol, schema=schema))
    return entries


__all__ = ["SemanticResult", "build_semantic_schemas"]
