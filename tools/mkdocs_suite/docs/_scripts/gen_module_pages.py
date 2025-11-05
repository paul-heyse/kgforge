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

import copy
import importlib
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard, cast

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


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TOOLS_ROOT = PROJECT_ROOT / "tools"
SRC_ROOT = PROJECT_ROOT / "src"

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from kgfoundry_common.navmap_loader import load_nav_metadata as _load_nav_metadata
    from tools.navmap.griffe_navmap import (
        DEFAULT_EXTENSIONS as _DEFAULT_EXTENSIONS,
    )
    from tools.navmap.griffe_navmap import (
        DEFAULT_SEARCH_PATHS as _DEFAULT_SEARCH_PATHS,
    )
else:
    _griffe_navmap_module_path = TOOLS_ROOT / "navmap" / "griffe_navmap.py"
    _griffe_navmap_spec = importlib.util.spec_from_file_location(
        "tools.mkdocs_suite._griffe_navmap", _griffe_navmap_module_path
    )
    if (
        _griffe_navmap_spec is None or _griffe_navmap_spec.loader is None
    ):  # pragma: no cover - defensive guard
        error_message = f"Unable to load griffe navmap module from {_griffe_navmap_module_path}"
        raise RuntimeError(error_message)
    _griffe_navmap = importlib.util.module_from_spec(_griffe_navmap_spec)
    sys.modules[_griffe_navmap_spec.name] = _griffe_navmap
    _griffe_navmap_spec.loader.exec_module(_griffe_navmap)  # type: ignore[union-attr]

    _nav_loader_path = SRC_ROOT / "kgfoundry_common" / "navmap_loader.py"
    _nav_loader_spec = importlib.util.spec_from_file_location(
        "tools.mkdocs_suite._navmap_loader", _nav_loader_path
    )
    if _nav_loader_spec is None or _nav_loader_spec.loader is None:  # pragma: no cover
        error_message = f"Unable to load navmap loader from {_nav_loader_path}"
        raise RuntimeError(error_message)
    _nav_loader = importlib.util.module_from_spec(_nav_loader_spec)
    sys.modules[_nav_loader_spec.name] = _nav_loader
    _nav_loader_spec.loader.exec_module(_nav_loader)  # type: ignore[union-attr]

    DEFAULT_EXTENSIONS = list(_griffe_navmap.DEFAULT_EXTENSIONS)
    DEFAULT_SEARCH_PATHS = list(_griffe_navmap.DEFAULT_SEARCH_PATHS)
    load_nav_metadata = _nav_loader.load_nav_metadata

if TYPE_CHECKING:
    DEFAULT_EXTENSIONS = list(_DEFAULT_EXTENSIONS)
    DEFAULT_SEARCH_PATHS = list(_DEFAULT_SEARCH_PATHS)
    load_nav_metadata = _load_nav_metadata

PACKAGE_ROOTS: tuple[str, ...] = (
    "kgfoundry_common",
    "docling",
    "download",
    "embeddings_dense",
    "embeddings_sparse",
    "kg_builder",
    "linking",
    "observability",
    "ontology",
    "orchestration",
    "registry",
    "search_api",
    "search_client",
    "vectorstore_faiss",
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
    documented_modules: set[str]


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


def _collect_modules(extensions_bundle: object) -> dict[str, _GriffeModule]:
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
    visited: set[str] = set()
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
                    resolve_external=True,
                )
        except _GriffeError as exc:  # pragma: no cover - degradation for broken modules
            LOGGER.warning("Skipping package %s due to Griffe load error: %s", package, exc)
            continue
        if not _is_module(root):  # pragma: no cover - defensive guard
            continue

        def _visit(module: _GriffeModule) -> None:
            if module.path in visited:
                return
            visited.add(module.path)
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
        documented_modules=set(modules.keys()),
    )


def _format_module_links(module_names: Iterable[str], documented: set[str]) -> str:
    """Return a comma-separated list of module names with links where possible.

    Returns
    -------
    str
        Comma-separated string where documented modules link to their pages and
        external modules are rendered as inline code.
    """
    links: list[str] = []
    for name in module_names:
        if name in documented:
            links.append(f"[{name}](../modules/{name}.md)")
        else:
            links.append(f"`{name}`")
    return ", ".join(links)


def _nav_metadata_for_module(
    module_path: str, module: _GriffeModule, facts: ModuleFacts
) -> dict[str, Any]:
    """Return nav metadata merged with runtime defaults for ``module_path``.

    Returns
    -------
    dict[str, Any]
        Normalized navigation metadata including exports, sections, synopsis,
        and relationship details for the module.
    """
    exports = sorted(_module_exports(module))
    raw_meta = load_nav_metadata(module_path, tuple(exports))
    meta: dict[str, Any] = copy.deepcopy(raw_meta)
    meta["exports"] = exports

    symbols_meta = meta.get("symbols")
    if not isinstance(symbols_meta, dict):
        symbols_meta = {}
    meta["symbols"] = {name: symbols_meta.get(name, {}) for name in exports}

    sections = meta.get("sections")
    if not isinstance(sections, list) or not sections:
        meta["sections"] = [
            {
                "id": "public-api",
                "title": "Public API",
                "symbols": exports,
            }
        ]

    synopsis = meta.get("synopsis")
    if not synopsis:
        synopsis = _first_paragraph(module)
        if synopsis:
            meta["synopsis"] = synopsis

    imports = sorted(facts.imports.get(module_path, set()))
    imported_by = sorted(facts.imported_by.get(module_path, set()))
    relationships: dict[str, list[str]] = {}
    if imports:
        relationships["imports"] = imports
    if imported_by:
        relationships["imported_by"] = imported_by
    existing_relationships = meta.get("relationships")
    if isinstance(existing_relationships, dict):
        relationships = {**existing_relationships, **relationships}
    if relationships:
        meta["relationships"] = relationships

    return meta


def _write_navmap_json(module_path: str, nav_meta: dict[str, Any]) -> None:
    """Persist nav metadata for ``module_path`` to the MkDocs virtual FS."""
    rel_path = module_path.replace(".", "/") + ".json"
    json_path = f"_data/navmaps/{rel_path}"
    with mkdocs_gen_files.open(json_path, "w") as fd:
        json.dump(nav_meta, fd, indent=2)


def _write_relationships(fd: mkdocs_gen_files.files, module_path: str, facts: ModuleFacts) -> None:
    """Write the relationships section for ``module_path``."""
    outgoing = sorted(facts.imports.get(module_path, set()))
    incoming = sorted(facts.imported_by.get(module_path, set()))
    exported = sorted(facts.exports.get(module_path, set()))
    if not (outgoing or incoming or exported):
        return
    fd.write("## Relationships\n\n")
    if outgoing:
        out_links = _format_module_links(outgoing, facts.documented_modules)
        fd.write(f"**Imports:** {out_links}\n\n")
    if incoming:
        in_links = _format_module_links(incoming, facts.documented_modules)
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
    links = ", ".join(f"`{op}`" for op in sorted(used_operations))
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
    nav_meta: Mapping[str, Any],
    facts: ModuleFacts,
) -> None:
    """Generate the Markdown page for ``module_path``."""
    page_path = f"modules/{module_path}.md"
    with mkdocs_gen_files.open(page_path, "w") as fd:
        fd.write(f"# {module_path}\n\n")
        summary = nav_meta.get("synopsis")
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
    _extensions, extensions_bundle = _discover_extensions(DEFAULT_EXTENSIONS)
    modules = _collect_modules(extensions_bundle)
    if not modules:
        LOGGER.warning("No modules loaded via Griffe; skipping module page generation.")
        _write_module_index({})
        return
    api_usage = _load_api_usage()
    facts = _build_relationships(modules, api_usage)
    manifest: dict[str, Any] = {}
    for module_path, module in modules.items():
        nav_meta = _nav_metadata_for_module(module_path, module, facts)
        manifest[module_path] = nav_meta
        _write_navmap_json(module_path, nav_meta)
        _render_module_page(module_path, module, nav_meta, facts)
    _write_module_index(modules)
    with mkdocs_gen_files.open("_data/navmaps/manifest.json", "w") as fd:
        json.dump(manifest, fd, indent=2, sort_keys=True)


main()
