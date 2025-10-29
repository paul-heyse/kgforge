"""Harvest phase using Griffe for structural information."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Literal

import libcst as cst

from .config import BuilderConfig

try:  # Import lazily so unit tests can patch when griffe is unavailable.
    from griffe.dataclasses import Class, Function, Module, Object
    from griffe.loader import GriffeLoader
except ModuleNotFoundError as exc:  # pragma: no cover - handled in runtime tests
    raise RuntimeError(
        "Griffe is required to run the docstring builder. Install the optional dependency via ``uv sync``."
    ) from exc


@dataclass(slots=True)
class ParameterHarvest:
    """Harvested signature information for a single parameter."""

    name: str
    kind: str
    annotation: str | None
    default: str | None


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


def _load_module(module_name: str, search_paths: list[str]) -> Module:
    loader = GriffeLoader(search_paths=search_paths)
    return loader.load_module(module_name)


def _annotation_to_str(annotation: object | None) -> str | None:
    if annotation is None:
        return None
    try:
        return annotation.as_string()
    except AttributeError:  # pragma: no cover - defensive fallback
        return str(annotation)


def _default_to_str(default: object | None) -> str | None:
    if default is None:
        return None
    try:
        return default.as_string()
    except AttributeError:  # pragma: no cover - defensive fallback
        return str(default)


def _iter_parameters(function: Function) -> Iterator[ParameterHarvest]:
    for param in function.parameters:
        yield ParameterHarvest(
            name=param.name,
            kind=param.kind.value,
            annotation=_annotation_to_str(param.annotation),
            default=_default_to_str(param.default),
        )


def _collect_symbols(
    obj: Object,
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
            col_offset=obj.col_offset or 0,
            decorators=[decorator.name for decorator in obj.decorators],
            is_async=False,
            is_generator=False,
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
        yield SymbolHarvest(
            qname=qname,
            module=module_name,
            kind="method" if prefix else "function",
            parameters=parameters,
            return_annotation=_annotation_to_str(obj.return_annotation),
            docstring=docstring,
            owned=owned,
            filepath=file_path,
            lineno=obj.lineno or 1,
            end_lineno=obj.endlineno,
            col_offset=obj.col_offset or 0,
            decorators=[decorator.name for decorator in obj.decorators],
            is_async=obj.is_async,
            is_generator=obj.is_generator,
        )
        return
    if isinstance(obj, Module):
        yield from _walk_members(obj, module_name, file_path, config, prefix)


def _walk_members(
    obj: Module | Class,
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

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:  # noqa: D401 - behaviour doc inherited
        qname = self._qualify(node.name.value)
        self.index[qname] = node
        self.namespace.append(node.name.value)
        return True

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:  # noqa: D401
        self.namespace.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:  # noqa: D401
        qname = self._qualify(node.name.value)
        self.index[qname] = node
        self.namespace.append(node.name.value)
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:  # noqa: D401
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
    "SymbolHarvest",
    "ParameterHarvest",
    "harvest_file",
    "iter_target_files",
]
