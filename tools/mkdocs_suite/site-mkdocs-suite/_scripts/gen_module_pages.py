"""Generate per-module MkDocs pages using Griffe and the navmap integration.

The script is executed via ``mkdocs-gen-files`` as part of the MkDocs build for
the experimental documentation suite. It loads the KgFoundry packages with
Griffe, derives module relationships, and emits one Markdown file per module.

Integration with the existing Griffe-based navmap generator ensures both
systems rely on the same traversal semantics, extensions, and docstring parser.
Whenever the navmap surface changes the MkDocs pages automatically pick up the
revisions without additional glue code.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeGuard, cast

import mkdocs_gen_files

LOGGER = logging.getLogger(__name__)


@contextmanager
def _suppress_griffe_errors() -> Iterator[None]:
    """Temporarily raise the griffe logger level to suppress known load failures.

    Yields
    ------
    None
        Context manager sentinel used to restore the previous log level.
    """
    griffe_logger = logging.getLogger("griffe")
    previous_disabled = griffe_logger.disabled
    previous_level = griffe_logger.level
    previous_propagate = griffe_logger.propagate
    root_logger = logging.getLogger()
    previous_root_level = root_logger.level
    griffe_logger.disabled = True
    griffe_logger.setLevel(logging.CRITICAL)
    griffe_logger.propagate = False
    root_logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        griffe_logger.disabled = previous_disabled
        griffe_logger.setLevel(previous_level)
        griffe_logger.propagate = previous_propagate
        root_logger.setLevel(previous_root_level)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from tools.navmap.griffe_navmap import (
        DEFAULT_EXTENSIONS as _DEFAULT_EXTENSIONS,
    )
    from tools.navmap.griffe_navmap import (
        DEFAULT_SEARCH_PATHS as _DEFAULT_SEARCH_PATHS,
    )
    from tools.navmap.griffe_navmap import (
        NavmapBuildSettings as _NavmapBuildSettings,
    )
    from tools.navmap.griffe_navmap import (
        Symbol as _Symbol,
    )
    from tools.navmap.griffe_navmap import (
        build_navmap as _build_navmap,
    )
else:
    _griffe_navmap_module_path = Path(__file__).resolve().parents[3] / "navmap" / "griffe_navmap.py"
    _griffe_navmap_spec = importlib.util.spec_from_file_location(
        "tools.mkdocs_suite._griffe_navmap", _griffe_navmap_module_path
    )
    if (
        _griffe_navmap_spec is None or _griffe_navmap_spec.loader is None
    ):  # pragma: no cover - defensive guard
        message = f"Unable to load griffe navmap module from {_griffe_navmap_module_path}"
        raise RuntimeError(message)
    _griffe_navmap = importlib.util.module_from_spec(_griffe_navmap_spec)
    sys.modules[_griffe_navmap_spec.name] = _griffe_navmap
    _griffe_navmap_spec.loader.exec_module(_griffe_navmap)  # type: ignore[union-attr]

    NavmapBuildSettings = _griffe_navmap.NavmapBuildSettings
    Symbol = _griffe_navmap.Symbol
    build_navmap = _griffe_navmap.build_navmap
    DEFAULT_EXTENSIONS = list(_griffe_navmap.DEFAULT_EXTENSIONS)
    DEFAULT_SEARCH_PATHS = list(_griffe_navmap.DEFAULT_SEARCH_PATHS)

if TYPE_CHECKING:
    NavmapBuildSettings = _NavmapBuildSettings
    Symbol = _Symbol
    build_navmap = _build_navmap
    DEFAULT_EXTENSIONS = list(_DEFAULT_EXTENSIONS)
    DEFAULT_SEARCH_PATHS = list(_DEFAULT_SEARCH_PATHS)

PACKAGE_ROOTS: tuple[str, ...] = ("kgfoundry", "kgfoundry_common")
NAVMAP_SETTINGS = NavmapBuildSettings(
    docstring_parser="numpy",
    include_inherited=True,
    resolve_external=False,
)
API_USAGE_FILE = Path(__file__).with_name("api_usage.json")

_griffe = importlib.import_module("griffe")
_load = cast("Callable[..., object]", _griffe.load)
_load_extensions = cast("Callable[..., object]", _griffe.load_extensions)
_GriffeError = getattr(_griffe, "GriffeError", Exception)


class _Docstring(Protocol):
    value: str | None


class _GriffeObject(Protocol):
    path: str


class _GriffeWithMembers(_GriffeObject, Protocol):
    members: Mapping[str, object]


class _GriffeModule(_GriffeWithMembers, Protocol):
    docstring: _Docstring | None
    exports: Sequence[object] | None
    relative_filepath: Path | None


class _GriffeAlias(_GriffeObject, Protocol):
    is_imported: bool
    target_path: str | None
    target: object | None


class _GriffeClass(_GriffeWithMembers, Protocol):
    bases: Sequence[object]


class _GriffeFunction(_GriffeWithMembers, Protocol): ...


@dataclass(slots=True)
class _RelationshipTables:
    imports: dict[str, set[str]]
    classes: dict[str, list[str]]
    functions: dict[str, list[str]]
    bases: dict[str, list[str]]


@dataclass(slots=True)
class ModuleFacts:
    """Aggregated metadata used when rendering module pages."""

    imports: dict[str, set[str]]
    imported_by: dict[str, set[str]]
    exports: dict[str, set[str]]
    classes: dict[str, list[str]]
    functions: dict[str, list[str]]
    bases: dict[str, list[str]]
    api_usage: Mapping[str, list[str]]


def _is_kind(obj: object, expected: str) -> bool:
    return obj.__class__.__name__ == expected


def _is_module(obj: object) -> TypeGuard[_GriffeModule]:
    return _is_kind(obj, "Module")


def _is_alias(obj: object) -> TypeGuard[_GriffeAlias]:
    return _is_kind(obj, "Alias")


def _is_class(obj: object) -> TypeGuard[_GriffeClass]:
    return _is_kind(obj, "Class")


def _is_function(obj: object) -> TypeGuard[_GriffeFunction]:
    return _is_kind(obj, "Function")


def _first_paragraph(module: _GriffeModule) -> str | None:
    """
    Return the first paragraph from the module docstring, if present.

    Returns
    -------
    str | None
        First paragraph of the module docstring when available; otherwise ``None``.
    """
    if module.docstring is None or not module.docstring.value:
        return None
    content = module.docstring.value.strip()
    if not content:
        return None
    return content.split("\n\n", maxsplit=1)[0]


def _discover_extensions(candidates: Iterable[str]) -> tuple[list[str], object | None]:
    """
    Return available Griffe extensions and a preloaded bundle.

    Parameters
    ----------
    candidates
        Extension module names to probe.

    Returns
    -------
    tuple[list[str], object | None]
        A tuple containing the subset of importable extension names and the
        optional bundle returned by ``griffe.load_extensions``.
    """
    names = list(candidates)
    if not names:
        return [], None
    try:
        return names, _load_extensions(*names)
    except ModuleNotFoundError:
        available: list[str] = []
        for name in names:
            try:
                _load_extensions(name)
            except ModuleNotFoundError:
                continue
            available.append(name)
        if not available:
            return [], None
        return available, _load_extensions(*available)


def _collect_modules(extensions_bundle: object | None) -> dict[str, _GriffeModule]:
    """
    Load all submodules for the configured package roots.

    Parameters
    ----------
    extensions_bundle
        Preloaded extension bundle returned by ``griffe.load_extensions``.

    Returns
    -------
    dict[str, _GriffeModule]
        Mapping of module path to Griffe module objects.
    """
    modules: dict[str, _GriffeModule] = {}
    for package in PACKAGE_ROOTS:
        try:
            with _suppress_griffe_errors():
                root = _load(
                    package,
                    submodules=True,
                    extensions=extensions_bundle,
                    search_paths=DEFAULT_SEARCH_PATHS,
                    docstring_parser="numpy",
                    resolve_aliases=True,
                    resolve_external=False,
                )
        except _GriffeError as exc:  # pragma: no cover - degradation for broken modules
            LOGGER.warning("Skipping package %s due to Griffe load error: %s", package, exc)
            continue
        if not _is_module(root):  # pragma: no cover - defensive guard
            continue

        def _visit(module: _GriffeModule) -> None:
            modules[module.path] = module
            for child in module.members.values():
                if _is_module(child):
                    _visit(child)

        _visit(root)
    return modules


def _load_api_usage() -> dict[str, list[str]]:
    """
    Return optional API usage mappings for deep-linking ReDoc operations.

    Returns
    -------
    dict[str, list[str]]
        Mapping of symbol path to related OpenAPI ``operationId`` entries.
    """
    if not API_USAGE_FILE.exists():
        return {}
    return json.loads(API_USAGE_FILE.read_text(encoding="utf-8"))


def _module_exports(module: _GriffeModule) -> set[str]:
    """
    Extract export names declared in ``__all__`` for the module.

    Returns
    -------
    set[str]
        Names explicitly exported via ``__all__``.
    """
    names: set[str] = set()
    exports = module.exports or ()
    for export in exports:
        name = getattr(export, "name", None)
        if isinstance(name, str):
            names.add(name)
    return names


def _register_member(module_path: str, member: object, tables: _RelationshipTables) -> None:
    """Populate relationship tables based on a member from ``module_path``."""
    if _is_alias(member) and member.is_imported:
        target = member.target_path or getattr(member.target, "path", None)
        if isinstance(target, str):
            tables.imports[module_path].add(target)
        return
    if _is_class(member):
        tables.classes[module_path].append(member.path)
        for base in member.bases:
            base_path = getattr(base, "path", None)
            if isinstance(base_path, str):
                tables.bases[member.path].append(base_path)
        return
    if _is_function(member):
        tables.functions[module_path].append(member.path)


def _build_relationships(
    modules: Mapping[str, _GriffeModule], api_usage: Mapping[str, list[str]]
) -> ModuleFacts:
    """
    Return imports, exports, symbols, and related API usage facts.

    Returns
    -------
    ModuleFacts
        Aggregated relationship tables used for rendering module pages.
    """
    tables = _RelationshipTables(
        imports=defaultdict(set),
        classes=defaultdict(list),
        functions=defaultdict(list),
        bases=defaultdict(list),
    )
    imports = tables.imports
    exports: dict[str, set[str]] = defaultdict(set)
    for module_path, module in modules.items():
        exports[module_path].update(_module_exports(module))
        for member in module.members.values():
            _register_member(module_path, member, tables)
    imported_by: dict[str, set[str]] = defaultdict(set)
    for src, targets in imports.items():
        for target in targets:
            imported_by[target].add(src)
    return ModuleFacts(
        imports=dict(imports),
        imported_by=dict(imported_by),
        exports=dict(exports),
        classes=dict(tables.classes),
        functions=dict(tables.functions),
        bases=dict(tables.bases),
        api_usage=api_usage,
    )


def _write_relationships(fd: mkdocs_gen_files.files, module_path: str, facts: ModuleFacts) -> None:
    """Write the relationships section for ``module_path``."""
    outgoing = sorted(facts.imports.get(module_path, set()))
    incoming = sorted(facts.imported_by.get(module_path, set()))
    exported = sorted(facts.exports.get(module_path, set()))
    if not (outgoing or incoming or exported):
        return
    fd.write("## Relationships\n\n")
    if outgoing:
        out_links = ", ".join(f"[{name}](../modules/{name}.md)" for name in outgoing)
        fd.write(f"**Imports:** {out_links}\n\n")
    if incoming:
        in_links = ", ".join(f"[{name}](../modules/{name}.md)" for name in incoming)
        fd.write(f"**Imported by:** {in_links}\n\n")
    if exported:
        export_list = ", ".join(f"`{name}`" for name in exported)
        fd.write(f"**Exports (`__all__`):** {export_list}\n\n")


def _write_related_operations(
    fd: mkdocs_gen_files.files, module_path: str, facts: ModuleFacts
) -> None:
    """Emit related API operations for symbols in ``module_path``."""
    used_operations: set[str] = set()
    for symbol_path in facts.classes.get(module_path, []) + facts.functions.get(module_path, []):
        for operation in facts.api_usage.get(symbol_path, []):
            used_operations.add(operation)
    if not used_operations:
        return
    links = ", ".join(f"[{op}](../api/index/#operation/{op})" for op in sorted(used_operations))
    fd.write("## Related API operations\n\n")
    fd.write(f"{links}\n\n")


def _write_contents(fd: mkdocs_gen_files.files, module_path: str, facts: ModuleFacts) -> None:
    """Render the class and function inventory for ``module_path``."""
    classes = facts.classes.get(module_path, [])
    functions = facts.functions.get(module_path, [])
    if not classes and not functions:
        return
    fd.write("## Contents\n\n")
    for class_path in sorted(classes):
        fd.write(f"### {class_path}\n\n::: {class_path}\n\n")
        base_paths = facts.bases.get(class_path, [])
        if base_paths:
            base_links = ", ".join(sorted(base_paths))
            fd.write(f"*Bases:* {base_links}\n\n")
    for function_path in sorted(functions):
        fd.write(f"### {function_path}\n\n::: {function_path}\n\n")


def _render_module_page(
    module_path: str,
    module: _GriffeModule,
    nav_summary: Symbol | None,
    facts: ModuleFacts,
) -> None:
    """Generate the Markdown page for ``module_path``."""
    page_path = f"modules/{module_path}.md"
    with mkdocs_gen_files.open(page_path, "w") as fd:
        fd.write(f"# {module_path}\n\n")
        summary = nav_summary.summary if nav_summary else None
        if not summary:
            summary = _first_paragraph(module)
        if summary:
            fd.write(f"{summary}\n\n")
        _write_relationships(fd, module_path, facts)
        _write_related_operations(fd, module_path, facts)
        _write_contents(fd, module_path, facts)
    relative_path = getattr(module, "relative_filepath", None)
    if isinstance(relative_path, Path):
        mkdocs_gen_files.set_edit_path(page_path, str(relative_path))


def _write_module_index(modules: Mapping[str, _GriffeModule]) -> None:
    """Create the landing page listing all discovered modules."""
    with mkdocs_gen_files.open("modules/index.md", "w") as fd:
        fd.write("# Modules\n\n")
        for module_path in sorted(modules):
            fd.write(f"- [{module_path}](./{module_path}.md)\n")


def main() -> None:
    """Entry point executed by mkdocs-gen-files."""
    extensions, extensions_bundle = _discover_extensions(DEFAULT_EXTENSIONS)
    try:
        with _suppress_griffe_errors():
            navmap = build_navmap(
                packages=PACKAGE_ROOTS,
                search_paths=DEFAULT_SEARCH_PATHS,
                extensions=extensions,
                settings=NAVMAP_SETTINGS,
            )
        navmap_symbols = navmap.symbols
    except _GriffeError as exc:  # pragma: no cover - degradation when navmap build fails
        LOGGER.warning("Griffe navmap build failed: %s", exc)
        navmap_symbols = []
    modules = _collect_modules(extensions_bundle)
    if not modules:
        LOGGER.warning("No modules loaded via Griffe; skipping module page generation.")
        _write_module_index({})
        return
    api_usage = _load_api_usage()
    navmap_by_path: dict[str, Symbol] = {
        symbol.path: symbol
        for symbol in navmap_symbols
        if getattr(symbol, "kind", None) == "module"
    }
    facts = _build_relationships(modules, api_usage)
    for module_path, module in modules.items():
        _render_module_page(module_path, module, navmap_by_path.get(module_path), facts)
    _write_module_index(modules)


main()
