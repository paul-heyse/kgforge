"""Plugin that derives docstring content from dataclass field metadata."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

from tools.docstring_builder.plugins.base import PluginContext, PluginStage
from tools.docstring_builder.schema import ParameterDoc
from tools.docstring_builder.semantics import SemanticResult

_DATACLASS_DECORATORS = {
    "dataclass",
    "dataclasses.dataclass",
    "attr.s",
    "attr.attrs",
    "attr.frozen",
    "attr.mutable",
    "attr.define",
    "attrs.define",
    "attrs.mutable",
    "attrs.frozen",
}

_FIELD_FACTORY_NAMES = {
    "field",
    "dataclasses.field",
    "attr.ib",
    "attr.attrib",
    "attr.field",
    "attrs.field",
}


@dataclass(slots=True)
class _FieldInfo:
    """Lightweight container describing a dataclass field."""

    name: str
    annotation: str | None
    default: str | None
    optional: bool
    description: str | None


def _decorator_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _decorator_name(node.value)
        if prefix:
            return f"{prefix}.{node.attr}"
        return node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return None


def _has_dataclass_decorator(decorators: list[ast.expr]) -> bool:
    for decorator in decorators:
        name = _decorator_name(decorator)
        if name in _DATACLASS_DECORATORS:
            return True
    return False


def _stringify(node: ast.AST) -> str | None:
    try:
        return ast.unparse(node)
    except (AttributeError, ValueError):  # pragma: no cover - ast.unparse can fail for exotic nodes
        return None


def _annotation_is_optional(annotation: str | None) -> bool:
    if not annotation:
        return False
    lowered = annotation.replace(" ", "").lower()
    return bool("optional[" in lowered or "|none" in lowered or "none|" in lowered)


def _default_description(name: str, default: str | None) -> str:
    readable = name.replace("_", " ")
    sentence = f"Configure the {readable}."
    if default:
        sentence += f" Defaults to ``{default}``."
    return sentence


def _field_from_annassign(node: ast.AnnAssign) -> _FieldInfo | None:
    if not isinstance(node.target, ast.Name):
        return None
    name = node.target.id
    annotation = _stringify(node.annotation) if node.annotation is not None else None
    optional = False
    default = None
    description: str | None = None
    value = node.value
    if value is not None:
        if isinstance(value, ast.Call) and _decorator_name(value.func) in _FIELD_FACTORY_NAMES:
            for keyword in value.keywords:
                if keyword.arg == "default" and keyword.value is not None:
                    default = _stringify(keyword.value)
                    optional = default is not None or optional
                elif keyword.arg == "default_factory" and keyword.value is not None:
                    factory = _stringify(keyword.value)
                    if factory:
                        default = f"{factory}()"
                        optional = True
                elif keyword.arg == "metadata" and isinstance(keyword.value, ast.Dict):
                    description = _metadata_doc(keyword.value)
        else:
            default = _stringify(value)
            optional = default is not None
    optional = optional or _annotation_is_optional(annotation)
    return _FieldInfo(
        name=name,
        annotation=annotation,
        default=default,
        optional=optional,
        description=description,
    )


def _metadata_doc(node: ast.Dict) -> str | None:
    for key_node, value_node in zip(node.keys, node.values, strict=False):
        if isinstance(key_node, ast.Constant) and key_node.value in {"doc", "description"}:
            text = _stringify(value_node)
            if text:
                return text.strip("\"'")
    return None


class _DataclassFieldCollector(ast.NodeVisitor):
    """Collect dataclass field metadata keyed by fully-qualified class name."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.namespace: list[str] = []
        self.fields: dict[str, list[_FieldInfo]] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qname = ".".join(part for part in [self.module, *self.namespace, node.name] if part)
        if _has_dataclass_decorator(node.decorator_list):
            collected: list[_FieldInfo] = []
            for statement in node.body:
                if isinstance(statement, ast.AnnAssign):
                    field = _field_from_annassign(statement)
                    if field is not None:
                        collected.append(field)
            if collected:
                self.fields[qname] = collected
        self.namespace.append(node.name)
        self.generic_visit(node)
        self.namespace.pop()


class DataclassFieldDocPlugin:
    """Populate dataclass parameter documentation from field definitions."""

    name: ClassVar[str] = "dataclass_field_docs"
    stage: ClassVar[PluginStage] = "transformer"

    def __init__(self) -> None:
        self._cache: dict[Path, dict[str, list[_FieldInfo]]] = {}

    @staticmethod
    def on_start(context: PluginContext) -> None:
        """Reset caches before processing begins."""
        del context

    @staticmethod
    def on_finish(context: PluginContext) -> None:
        """Release cached field metadata at the end of processing."""
        del context

    def apply(self, context: PluginContext, result: SemanticResult) -> SemanticResult:
        """Populate dataclass parameter metadata for ``result``."""
        if result.symbol.kind != "class":
            return result
        if not self._decorators_indicate_dataclass(result.symbol.decorators):
            return result
        file_path = context.file_path
        if file_path is None or not file_path.exists():
            return result
        field_map = self._cache.get(file_path)
        if field_map is None:
            field_map = self._collect_fields(file_path, result.symbol.module)
            self._cache[file_path] = field_map
        fields = field_map.get(result.symbol.qname)
        if not fields:
            return result
        return self._apply_fields(result, fields)

    @staticmethod
    def _decorators_indicate_dataclass(decorators: list[str]) -> bool:
        normalized = {decorator.lower() for decorator in decorators}
        return any(decorator in normalized for decorator in _DATACLASS_DECORATORS)

    @staticmethod
    def _collect_fields(path: Path, module: str) -> dict[str, list[_FieldInfo]]:
        """Parse ``path`` and return dataclass field metadata keyed by qualified name."""
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - non-readable file
            return {}
        try:
            tree = ast.parse(source)
        except SyntaxError:  # pragma: no cover - malformed source
            return {}
        collector = _DataclassFieldCollector(module)
        collector.visit(tree)
        return collector.fields

    @staticmethod
    def _apply_fields(result: SemanticResult, fields: list[_FieldInfo]) -> SemanticResult:
        """Return ``result`` updated with dataclass field documentation."""
        schema = result.schema
        existing = {parameter.name: parameter for parameter in schema.parameters}
        updated: list[ParameterDoc] = []
        field_names = {field.name for field in fields}
        for field in fields:
            current = existing.get(field.name)
            description = field.description or (current.description if current else None)
            if not description:
                description = _default_description(field.name, field.default)
            annotation = field.annotation or (current.annotation if current else None)
            default = (
                field.default
                if field.default is not None
                else (current.default if current else None)
            )
            optional = field.optional
            if current is not None and current.optional and not optional:
                optional = True
            display_name = current.display_name if current else field.name
            kind = current.kind if current else "positional_or_keyword"
            updated.append(
                ParameterDoc(
                    name=field.name,
                    annotation=annotation,
                    description=description,
                    optional=optional,
                    default=default,
                    display_name=display_name,
                    kind=kind,
                )
            )
        for parameter in schema.parameters:
            if parameter.name not in field_names:
                updated.append(parameter)
        if updated == schema.parameters:
            return result
        updated_schema = replace(schema, parameters=updated)
        return replace(result, schema=updated_schema)


def collect_dataclass_field_names(path: Path, module: str) -> dict[str, list[str]]:
    """Return dataclass field names keyed by fully-qualified class name."""
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - file may not exist
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - malformed source
        return {}
    collector = _DataclassFieldCollector(module)
    collector.visit(tree)
    return {key: [field.name for field in fields] for key, fields in collector.fields.items()}


__all__ = ["DataclassFieldDocPlugin", "collect_dataclass_field_names"]
