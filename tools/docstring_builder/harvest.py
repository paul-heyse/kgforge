"""Harvest phase using Griffe for structural information."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, TypeGuard, cast, runtime_checkable

import libcst as cst

from tools.docstring_builder.config import BuilderConfig
from tools.docstring_builder.models import SymbolResolutionError
from tools.griffe_utils import resolve_griffe

try:  # Import lazily so unit tests can patch when griffe is unavailable.
    _GRIFFE_API = resolve_griffe()
except (ModuleNotFoundError, AttributeError) as exc:  # pragma: no cover - runtime guard
    message = "Griffe is required to run the docstring builder. Install the optional dependency via ``uv sync``."
    raise RuntimeError(message) from exc


@runtime_checkable
class GriffeDocstringLike(Protocol):
    value: str


@runtime_checkable
class GriffeParameterLike(Protocol):
    name: str
    kind: object
    annotation: object | None
    default: object | None


@runtime_checkable
class GriffeFunctionLike(Protocol):
    name: str
    docstring: GriffeDocstringLike | None
    lineno: int | None
    endlineno: int | None
    col_offset: int | None
    parameters: Sequence[GriffeParameterLike]
    decorators: Sequence[object]
    return_annotation: object | None
    returns: object | None
    is_async: bool
    is_generator: bool


@runtime_checkable
class GriffeClassLike(Protocol):
    name: str
    docstring: GriffeDocstringLike | None
    lineno: int | None
    endlineno: int | None
    col_offset: int | None
    members: Mapping[str, object]
    decorators: Sequence[object]
    is_async: bool
    is_generator: bool


@runtime_checkable
class GriffeModuleLike(Protocol):
    name: str
    members: Mapping[str, object]


class GriffeLoaderInstanceLike(Protocol):
    def load(self, module_name: str) -> GriffeModuleLike: ...

    def load_module(self, module_name: str) -> GriffeModuleLike: ...


class GriffeLoaderFactoryLike(Protocol):
    def __call__(self, *, search_paths: Sequence[str]) -> GriffeLoaderInstanceLike: ...


ClassType = cast(type, _GRIFFE_API.class_type)
FunctionType = cast(type, _GRIFFE_API.function_type)
ModuleType = cast(type, _GRIFFE_API.module_type)
GriffeLoaderFactory = cast(GriffeLoaderFactoryLike, _GRIFFE_API.loader_type)


def _is_griffe_class(obj: object) -> TypeGuard[GriffeClassLike]:
    return isinstance(obj, ClassType)


def _is_griffe_function(obj: object) -> TypeGuard[GriffeFunctionLike]:
    return isinstance(obj, FunctionType)


def _is_griffe_module(obj: object) -> TypeGuard[GriffeModuleLike]:
    return isinstance(obj, ModuleType)


_KIND_LOOKUP: dict[str, inspect._ParameterKind] = {
    "positional_only": inspect._ParameterKind.POSITIONAL_ONLY,
    "positional_or_keyword": inspect._ParameterKind.POSITIONAL_OR_KEYWORD,
    "var_positional": inspect._ParameterKind.VAR_POSITIONAL,
    "variadic_positional": inspect._ParameterKind.VAR_POSITIONAL,
    "keyword_only": inspect._ParameterKind.KEYWORD_ONLY,
    "var_keyword": inspect._ParameterKind.VAR_KEYWORD,
    "variadic_keyword": inspect._ParameterKind.VAR_KEYWORD,
}


def _safe_getattr(value: object, attr: str) -> object | None:
    try:
        attr_value = cast(object, getattr(value, attr))
    except AttributeError:  # pragma: no cover - defensive fallback
        return None
    return attr_value


def _extract_string_attribute(value: object, attr: str) -> str | None:
    candidate = _safe_getattr(value, attr)
    if isinstance(candidate, str):
        return candidate
    return None


def _docstring_value(docstring: GriffeDocstringLike | None) -> str | None:
    if docstring is None:
        return None
    value = _extract_string_attribute(docstring, "value")
    return value if value is not None else None


def _normalize_parameter_kind(value: object) -> inspect._ParameterKind:
    """Coerce ``griffe`` parameter kinds into :class:`inspect._ParameterKind`."""
    if isinstance(value, inspect._ParameterKind):
        return value
    name = _extract_string_attribute(value, "name")
    token_source = name if name is not None else _safe_getattr(value, "value")
    key = str(token_source if token_source is not None else value).replace("-", "_").lower()
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
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _load_module(module_name: str, search_paths: Sequence[str]) -> GriffeModuleLike:
    loader = GriffeLoaderFactory(search_paths=tuple(search_paths))
    load_candidate = cast(Callable[[str], object] | None, getattr(loader, "load_module", None))
    if callable(load_candidate):
        load_fn: Callable[[str], object] = load_candidate
    else:
        load_fallback = cast(Callable[[str], object] | None, getattr(loader, "load", None))
        if not callable(load_fallback):  # pragma: no cover - defensive fallback
            message = "Griffe loader is missing a callable 'load' method"
            raise SymbolResolutionError(message)
        load_fn = load_fallback
    try:
        module = load_fn(module_name)
    except ModuleNotFoundError as exc:
        if module_name.endswith(".__init__"):
            package_name = module_name.rsplit(".", 1)[0]
            module = load_fn(package_name)
        else:
            message = f"Unable to load module '{module_name}'"
            raise SymbolResolutionError(message) from exc
    if not _is_griffe_module(module):  # pragma: no cover - defensive fallback
        message = f"Griffe loader returned unsupported module type for '{module_name}'"
        raise SymbolResolutionError(message)
    return module


def _expr_to_str(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    as_string = _safe_getattr(value, "as_string")
    if callable(as_string):
        as_string_callable = cast(Callable[[], object], as_string)
        try:
            candidate = as_string_callable()
        except (TypeError, ValueError):  # pragma: no cover - defensive fallback
            candidate = None
        if isinstance(candidate, str):
            return candidate
    for attribute in ("name", "path", "value"):
        attr_value = _extract_string_attribute(value, attribute)
        if attr_value is not None:
            return attr_value
    try:
        return str(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
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


def _iter_parameters(function: GriffeFunctionLike) -> Iterator[ParameterHarvest]:
    for param in function.parameters:
        yield ParameterHarvest(
            name=param.name,
            kind=_normalize_parameter_kind(param.kind),
            annotation=_annotation_to_str(param.annotation),
            default=_default_to_str(param.default),
        )


def _collect_symbols(
    obj: object,
    module_name: str,
    file_path: Path,
    config: BuilderConfig,
    prefix: list[str] | None = None,
) -> Iterator[SymbolHarvest]:
    prefix = prefix or []
    if _is_griffe_class(obj):
        qname = ".".join([module_name, *prefix, obj.name])
        docstring = _docstring_value(obj.docstring)
        owned = (docstring and config.ownership_marker in docstring) or docstring is None
        if qname in config.package_settings.opt_out:
            owned = False
        col_offset = getattr(obj, "col_offset", 0) or 0
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
            col_offset=col_offset,
            decorators=_decorator_names(tuple(obj.decorators)),
            is_async=False,
            is_generator=False,
        )
        yield from _walk_members(obj, module_name, file_path, config, [*prefix, obj.name])
        return
    if _is_griffe_function(obj):
        qname = ".".join([module_name, *prefix, obj.name])
        parameters = list(_iter_parameters(obj))
        docstring = _docstring_value(obj.docstring)
        owned = (docstring and config.ownership_marker in docstring) or docstring is None
        if qname in config.package_settings.opt_out:
            owned = False
        return_annotation_obj = getattr(obj, "return_annotation", None) or getattr(
            obj, "returns", None
        )
        col_offset = getattr(obj, "col_offset", 0) or 0
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
            col_offset=col_offset,
            decorators=_decorator_names(tuple(obj.decorators)),
            is_async=bool(getattr(obj, "is_async", False)),
            is_generator=bool(getattr(obj, "is_generator", False)),
        )
        return
    if _is_griffe_module(obj):
        yield from _walk_members(obj, module_name, file_path, config, prefix)


def _walk_members(
    obj: GriffeModuleLike | GriffeClassLike,
    module_name: str,
    file_path: Path,
    config: BuilderConfig,
    prefix: list[str] | None = None,
) -> Iterator[SymbolHarvest]:
    prefix = prefix or []
    for member in obj.members.values():
        yield from _collect_symbols(member, module_name, file_path, config, prefix)


class _IndexCollector(cst.CSTTransformer):
    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self.namespace: list[str] = []
        self.index: dict[str, cst.CSTNode] = {}

    def _qualify(self, name: str) -> str:
        pieces = [self.module_name, *self.namespace, name]
        return ".".join(piece for piece in pieces if piece)

    def visit_classdef(self, node: cst.ClassDef) -> bool:
        self.namespace.append(node.name.value)
        qualified = self._qualify(node.name.value)
        self.index[qualified] = node
        return True

    def leave_classdef(
        self, _original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.CSTNode:
        self.namespace.pop()
        return updated_node

    def visit_functiondef(self, node: cst.FunctionDef) -> bool:
        self.namespace.append(node.name.value)
        qualified = self._qualify(node.name.value)
        self.index[qualified] = node
        return True

    def leave_functiondef(
        self, _original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.CSTNode:
        self.namespace.pop()
        return updated_node


def _build_cst_index(module_name: str, file_path: Path) -> dict[str, cst.CSTNode]:
    module = cst.parse_module(file_path.read_text(encoding="utf-8"))
    collector = _IndexCollector(module_name)
    module.visit(collector)
    return collector.index


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
