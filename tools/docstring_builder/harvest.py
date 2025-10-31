"""Harvest phase using Griffe for structural information."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import libcst as cst

from tools.docstring_builder.config import BuilderConfig
from tools.griffe_utils import resolve_griffe

try:  # Import lazily so unit tests can patch when griffe is unavailable.
    _GRIFFE_API = resolve_griffe()
except (ModuleNotFoundError, AttributeError) as exc:  # pragma: no cover - runtime guard
    message = "Griffe is required to run the docstring builder. Install the optional dependency via ``uv sync``."
    raise RuntimeError(message) from exc

if TYPE_CHECKING:
    from griffe import Class as GriffeClass
    from griffe import Function as GriffeFunction
    from griffe import GriffeLoader as GriffeLoaderType
    from griffe import Module as GriffeModule
    from griffe import Object as GriffeObject
else:  # pragma: no cover - typing fallback when ``griffe`` is absent at runtime
    GriffeClass = GriffeFunction = GriffeModule = GriffeObject = GriffeLoaderType = object

Class = cast(type[GriffeClass], _GRIFFE_API.class_type)
Function = cast(type[GriffeFunction], _GRIFFE_API.function_type)
Module = cast(type[GriffeModule], _GRIFFE_API.module_type)
Object = cast(type[GriffeObject], _GRIFFE_API.object_type)
GriffeLoader = cast(type[GriffeLoaderType], _GRIFFE_API.loader_type)


_KIND_LOOKUP: dict[str, inspect._ParameterKind] = {
    "positional_only": inspect._ParameterKind.POSITIONAL_ONLY,
    "positional_or_keyword": inspect._ParameterKind.POSITIONAL_OR_KEYWORD,
    "var_positional": inspect._ParameterKind.VAR_POSITIONAL,
    "variadic_positional": inspect._ParameterKind.VAR_POSITIONAL,
    "keyword_only": inspect._ParameterKind.KEYWORD_ONLY,
    "var_keyword": inspect._ParameterKind.VAR_KEYWORD,
    "variadic_keyword": inspect._ParameterKind.VAR_KEYWORD,
}


def _normalize_parameter_kind(value: object) -> inspect._ParameterKind:
    """Coerce ``griffe`` parameter kinds into :class:`inspect._ParameterKind`."""
    if isinstance(value, inspect._ParameterKind):
        return value
    name = getattr(value, "name", None)
    token = name if isinstance(name, str) else getattr(value, "value", None)
    key = str(token or value).replace("-", "_").lower()
    return _KIND_LOOKUP.get(key, inspect._ParameterKind.POSITIONAL_OR_KEYWORD)


@dataclass(slots=True)
class ParameterHarvest:
    """Harvested signature information for a single parameter."""

    name: str
    kind: inspect._ParameterKind
    annotation: str | None
    default: str | None

    def display_name(self) -> str:
        """Return the formatted name suitable for docstring rendering."""
        return parameter_display_name(self)


def parameter_display_name(parameter: ParameterHarvest) -> str:
    """Format a harvested parameter for docstring sections."""
    if parameter.kind is inspect._ParameterKind.VAR_POSITIONAL:
        return f"*{parameter.name}"
    if parameter.kind is inspect._ParameterKind.VAR_KEYWORD:
        return f"**{parameter.name}"
    return parameter.name


@dataclass(slots=True)
class SymbolHarvest:
    """Metadata describing a symbol subject to docstring generation."""

    qname: str
    module: str
    kind: Literal["function", "method", "class"]
    parameters: list[ParameterHarvest]
    return_annotation: str | None
    docstring: str | None
    owned: bool
    filepath: Path
    lineno: int
    end_lineno: int | None
    col_offset: int
    decorators: list[str]
    is_async: bool
    is_generator: bool


@dataclass(slots=True)
class HarvestResult:
    """Container mapping harvested symbols to CST nodes."""

    module: str
    filepath: Path
    symbols: list[SymbolHarvest]
    cst_index: dict[str, cst.CSTNode] = field(default_factory=dict)


def _module_name(root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root)
    parts = rel.with_suffix("").parts
    if parts and parts[0] in {"src", "tools", "docs"}:
        parts = parts[1:]
    return ".".join(parts)


def _load_module(module_name: str, search_paths: list[str]) -> GriffeModule:
    loader: GriffeLoaderType = GriffeLoader(search_paths=search_paths)
    load_candidate = getattr(loader, "load_module", None)
    load_fn: Callable[[str], GriffeModule]
    if callable(load_candidate):
        load_fn = cast(Callable[[str], GriffeModule], load_candidate)
    else:
        load_fn = cast(Callable[[str], GriffeModule], loader.load)
    try:
        return load_fn(module_name)
    except Exception:
        if module_name.endswith(".__init__"):
            package_name = module_name.rsplit(".", 1)[0]
            return load_fn(package_name)
        raise


def _safe_getattr(value: object, attr: str) -> object | None:
    try:
        attr_value = getattr(value, attr)
    except Exception:  # pragma: no cover - defensive fallback
        return None
    return cast(object, attr_value)


def _extract_string_attribute(value: object, attr: str) -> str | None:
    candidate = _safe_getattr(value, attr)
    if isinstance(candidate, str):
        return candidate
    return None


def _expr_to_str(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    as_string = _safe_getattr(value, "as_string")
    if callable(as_string):
        try:
            candidate = as_string()
        except Exception:  # pragma: no cover - defensive fallback
            candidate = None
        if isinstance(candidate, str):
            return candidate
    for attribute in ("name", "path", "value"):
        attr_value = _extract_string_attribute(value, attribute)
        if attr_value is not None:
            return attr_value
    try:
        return str(value)
    except Exception:  # pragma: no cover - defensive fallback
        return None


def _annotation_to_str(annotation: object | None) -> str | None:
    return _expr_to_str(annotation)


def _default_to_str(default: object | None) -> str | None:
    return _expr_to_str(default)


def _decorator_names(decorators: Iterable[object]) -> list[str]:
    names: list[str] = []
    for decorator in decorators or []:
        name_attr = _safe_getattr(decorator, "name")
        if isinstance(name_attr, str):
            names.append(name_attr)
            continue
        value_attr = _safe_getattr(decorator, "value") or decorator
        name = _expr_to_str(value_attr)
        if name is not None:
            names.append(name)
        else:  # pragma: no cover - defensive fallback
            names.append(str(decorator))
    return names


def _iter_parameters(function: GriffeFunction) -> Iterator[ParameterHarvest]:
    for param in function.parameters:
        yield ParameterHarvest(
            name=param.name,
            kind=_normalize_parameter_kind(param.kind),
            annotation=_annotation_to_str(param.annotation),
            default=_default_to_str(param.default),
        )


def _collect_symbols(
    obj: GriffeObject,
    module_name: str,
    file_path: Path,
    config: BuilderConfig,
    prefix: list[str] | None = None,
) -> Iterator[SymbolHarvest]:
    prefix = prefix or []
    if isinstance(obj, Class):
        qname = ".".join([module_name, *prefix, obj.name])
        docstring = obj.docstring.value if obj.docstring else None
        owned = (docstring and config.ownership_marker in docstring) or docstring is None
        if qname in config.package_settings.opt_out:
            owned = False
        yield SymbolHarvest(
            qname=qname,
            module=module_name,
            kind="class",
            parameters=[],
            return_annotation=None,
            docstring=docstring,
            owned=owned,
            filepath=file_path,
            lineno=obj.lineno or 1,
            end_lineno=obj.endlineno,
            col_offset=getattr(obj, "col_offset", 0) or 0,
            decorators=_decorator_names(getattr(obj, "decorators", [])),
            is_async=bool(getattr(obj, "is_async", False)),
            is_generator=bool(getattr(obj, "is_generator", False)),
        )
        yield from _walk_members(obj, module_name, file_path, config, [*prefix, obj.name])
        return
    if isinstance(obj, Function):
        qname = ".".join([module_name, *prefix, obj.name])
        parameters = list(_iter_parameters(obj))
        docstring = obj.docstring.value if obj.docstring else None
        owned = (docstring and config.ownership_marker in docstring) or docstring is None
        if qname in config.package_settings.opt_out:
            owned = False
        return_annotation_obj = getattr(obj, "return_annotation", None) or getattr(
            obj, "returns", None
        )
        yield SymbolHarvest(
            qname=qname,
            module=module_name,
            kind="method" if prefix else "function",
            parameters=parameters,
            return_annotation=_annotation_to_str(return_annotation_obj),
            docstring=docstring,
            owned=owned,
            filepath=file_path,
            lineno=obj.lineno or 1,
            end_lineno=obj.endlineno,
            col_offset=getattr(obj, "col_offset", 0) or 0,
            decorators=_decorator_names(getattr(obj, "decorators", [])),
            is_async=bool(getattr(obj, "is_async", False)),
            is_generator=bool(getattr(obj, "is_generator", False)),
        )
        return
    if isinstance(obj, Module):
        yield from _walk_members(obj, module_name, file_path, config, prefix)


def _walk_members(
    obj: GriffeModule | GriffeClass,
    module_name: str,
    file_path: Path,
    config: BuilderConfig,
    prefix: list[str] | None = None,
) -> Iterator[SymbolHarvest]:
    prefix = prefix or []
    for member in obj.members.values():
        yield from _collect_symbols(member, module_name, file_path, config, prefix)


class _IndexVisitor(cst.CSTVisitor):
    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self.namespace: list[str] = []
        self.index: dict[str, cst.CSTNode] = {}

    def _qualify(self, name: str) -> str:
        pieces = [self.module_name, *self.namespace, name]
        return ".".join(piece for piece in pieces if piece)

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:  # noqa: N802 - LibCST API contract
        qname = self._qualify(node.name.value)
        self.index[qname] = node
        self.namespace.append(node.name.value)
        return True

    def leave_ClassDef(self, _original_node: cst.ClassDef) -> None:  # noqa: N802
        self.namespace.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:  # noqa: N802 - LibCST API contract
        qname = self._qualify(node.name.value)
        self.index[qname] = node
        self.namespace.append(node.name.value)
        return True

    def leave_FunctionDef(self, _original_node: cst.FunctionDef) -> None:  # noqa: N802
        self.namespace.pop()


def _build_cst_index(module_name: str, file_path: Path) -> dict[str, cst.CSTNode]:
    module = cst.parse_module(file_path.read_text(encoding="utf-8"))
    visitor = _IndexVisitor(module_name)
    module.visit(visitor)
    return visitor.index


def harvest_file(file_path: Path, config: BuilderConfig, repo_root: Path) -> HarvestResult:
    """Collect symbol metadata for a single file."""
    module_name = _module_name(repo_root, file_path)
    if not module_name:
        module_name = file_path.stem
    search_paths = [str(repo_root / "src"), str(repo_root / "tools"), str(repo_root)]
    module = _load_module(module_name, search_paths)
    symbols = list(_walk_members(module, module_name, file_path, config))
    index = _build_cst_index(module_name, file_path)
    return HarvestResult(module=module_name, filepath=file_path, symbols=symbols, cst_index=index)


def iter_target_files(config: BuilderConfig, repo_root: Path) -> Iterator[Path]:
    """Yield files matching include/exclude patterns."""
    for pattern in config.include:
        for candidate in repo_root.glob(pattern):
            if not candidate.is_file() or candidate.suffix != ".py":
                continue
            rel = candidate.relative_to(repo_root)
            if any(rel.match(exclude) for exclude in config.exclude):
                continue
            yield candidate


__all__ = [
    "HarvestResult",
    "ParameterHarvest",
    "SymbolHarvest",
    "harvest_file",
    "iter_target_files",
    "parameter_display_name",
]
