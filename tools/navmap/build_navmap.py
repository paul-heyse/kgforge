#!/usr/bin/env python3
"""Overview of build navmap.

This module bundles build navmap logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import singledispatch
from pathlib import Path
from typing import TypedDict, cast

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
OUT = REPO / "site" / "_build" / "navmap"
OUT.mkdir(parents=True, exist_ok=True)
INDEX_PATH = OUT / "navmap.json"

# Link settings
G_ORG = os.getenv("DOCS_GITHUB_ORG")
G_REPO = os.getenv("DOCS_GITHUB_REPO")
G_SHA = os.getenv("DOCS_GITHUB_SHA")
LINK_MODE = os.getenv("DOCS_LINK_MODE", "editor").lower()  # editor|github|both
EDITOR_MODE = os.getenv("DOCS_EDITOR", "vscode").lower()

SECTION_RE = re.compile(r"^\s*#\s*\[nav:section\s+([a-z0-9]+(?:-[a-z0-9]+)*)\]\s*$")
ANCHOR_RE = re.compile(r"^\s*#\s*\[nav:anchor\s+([A-Za-z_]\w*)\]\s*$")
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")


class NavSectionDict(TypedDict):
    """Represent a normalized nav section."""

    id: str
    symbols: list[str]


class SymbolMetaDict(TypedDict, total=False):
    """Represent metadata for a single symbol."""

    owner: str
    stability: str
    since: str
    deprecated_in: str


class ModuleMetaDict(TypedDict, total=False):
    """Represent module-level metadata defaults."""

    owner: str
    stability: str
    since: str
    deprecated_in: str


class ModuleEntryDict(TypedDict):
    """Represent the JSON entry for a module."""

    path: str
    exports: list[str]
    sections: list[NavSectionDict]
    section_lines: dict[str, int]
    anchors: dict[str, int]
    links: dict[str, str]
    meta: dict[str, SymbolMetaDict]
    module_meta: ModuleMetaDict
    tags: list[str]
    synopsis: str
    see_also: list[str]
    deps: list[str]


class NavIndexDict(TypedDict):
    """Represent the persisted nav index."""

    commit: str
    policy_version: str
    link_mode: str
    modules: dict[str, ModuleEntryDict]


class AllPlaceholder:
    """Sentinel used to expand ``__all__`` placeholders."""

    __slots__ = ()


PLACEHOLDER_ALL = AllPlaceholder()


class NavmapError(Exception):
    """Base exception for navmap parsing issues."""


class NavmapLiteralError(NavmapError):
    """Raised when literal evaluation encounters unsupported constructs."""


class NavmapPlaceholderError(NavmapError):
    """Raised when placeholder expansion fails."""


class AllDictTemplate:
    """Model the AllDictTemplate.

    Represent the alldicttemplate data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    __slots__ = ("template",)

    def __init__(self, template: NavTree) -> None:
        """Compute init.

        Initialise a new instance with validated parameters. The constructor prepares internal state and coordinates any setup required by the class. Subclasses should call ``super().__init__`` to keep validation and defaults intact.

        Parameters
        ----------
        template : NavTree
            Description for ``template``.
        """
        self.template = template


NavPrimitive = str | int | float | bool | None
type NavTree = (
    NavPrimitive
    | list[NavTree]
    | dict[str, NavTree]
    | set[NavTree]
    | AllDictTemplate
    | AllPlaceholder
)
type ResolvedNavValue = (
    NavPrimitive | list[ResolvedNavValue] | dict[str, ResolvedNavValue] | set[str]
)


@singledispatch
def _eval_nav_literal(node: ast.AST) -> NavTree:
    """Return a parsed navmap literal for an AST node."""
    message = f"Unsupported navmap literal node: {ast.dump(node)}"
    raise NavmapLiteralError(message)


@_eval_nav_literal.register
def _(node: ast.Constant) -> NavTree:
    """.

    Parameters
    ----------
    node : ast.Constant
        Description.

    Returns
    -------
    NavTree
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _(...)
    """
    value = node.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    message = f"Unsupported constant in navmap literal: {value!r}"
    raise NavmapLiteralError(message)


@_eval_nav_literal.register
def _(node: ast.Name) -> NavTree:
    """.

    Parameters
    ----------
    node : ast.Name
        Description.

    Returns
    -------
    NavTree
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _(...)
    """
    if node.id == "__all__":
        return PLACEHOLDER_ALL
    message = f"Unsupported name in navmap literal: {node.id!r}"
    raise NavmapLiteralError(message)


def _eval_sequence(nodes: Sequence[ast.AST]) -> list[NavTree]:
    """Evaluate a list of AST nodes into navmap literals."""
    return [_literal_eval_navmap(child) for child in nodes]


@_eval_nav_literal.register
def _(node: ast.List) -> NavTree:
    """.

    Parameters
    ----------
    node : ast.List
        Description.

    Returns
    -------
    NavTree
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _(...)
    """
    return _eval_sequence(node.elts)


@_eval_nav_literal.register
def _(node: ast.Tuple) -> NavTree:
    """.

    Parameters
    ----------
    node : ast.Tuple
        Description.

    Returns
    -------
    NavTree
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _(...)
    """
    return _eval_sequence(node.elts)


@_eval_nav_literal.register
def _(node: ast.Set) -> NavTree:
    """.

    Parameters
    ----------
    node : ast.Set
        Description.

    Returns
    -------
    NavTree
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _(...)
    """
    return {_literal_eval_navmap(elt) for elt in node.elts}


def _eval_dict(node: ast.Dict) -> dict[str, NavTree]:
    """Evaluate a dict literal, enforcing string keys."""
    result: dict[str, NavTree] = {}
    for key_node, value_node in zip(node.keys, node.values, strict=False):
        key_literal = _literal_eval_navmap(key_node)
        if not isinstance(key_literal, str):
            message = "Navmap dictionary keys must be strings."
            raise NavmapLiteralError(message)
        result[key_literal] = _literal_eval_navmap(value_node)
    return result


@_eval_nav_literal.register
def _(node: ast.Dict) -> NavTree:
    """.

    Parameters
    ----------
    node : ast.Dict
        Description.

    Returns
    -------
    NavTree
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _(...)
    """
    return _eval_dict(node)


def _eval_dict_comprehension(node: ast.DictComp) -> AllDictTemplate:
    """Evaluate supported dict comprehensions into AllDictTemplate."""
    if len(node.generators) != 1:
        message = "Navmap dict comprehension must contain exactly one generator."
        raise NavmapLiteralError(message)
    generator = node.generators[0]
    if generator.ifs:
        message = "Navmap dict comprehension may not include filters."
        raise NavmapLiteralError(message)
    if generator.is_async:
        message = "Navmap dict comprehension may not be async."
        raise NavmapLiteralError(message)
    target = generator.target
    iterator = generator.iter
    if not isinstance(target, ast.Name):
        message = "Navmap dict comprehension target must be a simple name."
        raise NavmapLiteralError(message)
    if not isinstance(iterator, ast.Name) or iterator.id != "__all__":
        message = "Navmap dict comprehension iterator must be __all__."
        raise NavmapLiteralError(message)
    template = _literal_eval_navmap(node.value)
    return AllDictTemplate(template)


@_eval_nav_literal.register
def _(node: ast.DictComp) -> NavTree:
    """.

    Parameters
    ----------
    node : ast.DictComp
        Description.

    Returns
    -------
    NavTree
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _(...)
    """
    return _eval_dict_comprehension(node)


def _literal_eval_navmap(node: ast.AST | None) -> NavTree:
    """Return the navmap literal represented by ``node``."""
    if node is None:
        message = "Navmap literal must not be empty."
        raise NavmapLiteralError(message)
    return _eval_nav_literal(node)


def _expand_all_placeholder(exports: Sequence[str]) -> list[ResolvedNavValue]:
    """Return a deduplicated list of exports for ``__all__`` placeholders."""
    return cast(list[ResolvedNavValue], _dedupe_exports(exports))


def _expand_all_dict_template(
    template: NavTree, exports: Sequence[str]
) -> dict[str, ResolvedNavValue]:
    """Expand ``AllDictTemplate`` instances into concrete symbol mappings."""
    expanded: dict[str, ResolvedNavValue] = {}
    for name in exports:
        expanded[name] = _replace_placeholders(template, exports)
    return expanded


def _expand_list(values: Sequence[NavTree], exports: Sequence[str]) -> list[ResolvedNavValue]:
    """Expand placeholder-aware lists while flattening nested sequences."""
    expanded: list[ResolvedNavValue] = []
    for entry in values:
        resolved = _replace_placeholders(entry, exports)
        if isinstance(resolved, list):
            expanded.extend(resolved)
        else:
            expanded.append(resolved)
    return expanded


def _expand_dict(values: dict[str, NavTree], exports: Sequence[str]) -> dict[str, ResolvedNavValue]:
    """Expand placeholders within dictionary values."""
    return {key: _replace_placeholders(sub_value, exports) for key, sub_value in values.items()}


def _expand_set(values: set[NavTree], exports: Sequence[str]) -> set[str]:
    """Expand placeholders within sets, enforcing string membership."""
    unique_items: set[str] = set()
    for entry in values:
        resolved = _replace_placeholders(entry, exports)
        if isinstance(resolved, list):
            for value in resolved:
                if isinstance(value, str):
                    unique_items.add(value)
                else:
                    message = "Navmap sets may only contain strings after expansion."
                    raise NavmapPlaceholderError(message)
        elif isinstance(resolved, str):
            unique_items.add(resolved)
        else:
            message = "Navmap sets must resolve to strings."
            raise NavmapPlaceholderError(message)
    return unique_items


def _replace_placeholders(value: NavTree, exports: Sequence[str]) -> ResolvedNavValue:
    """Resolve navmap placeholders using the provided export list."""
    if isinstance(value, AllPlaceholder):
        return _expand_all_placeholder(exports)
    if isinstance(value, AllDictTemplate):
        return _expand_all_dict_template(value.template, exports)
    if isinstance(value, list):
        return _expand_list(value, exports)
    if isinstance(value, dict):
        return _expand_dict(value, exports)
    if isinstance(value, set):
        return _expand_set(value, exports)
    return value


@dataclass
class ModuleInfo:
    """Model the ModuleInfo.

    Represent the moduleinfo data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    module: str
    path: Path
    exports: list[str]
    sections: dict[str, int]  # id -> lineno (1-based)
    anchors: dict[str, int]  # symbol -> lineno (1-based)
    nav_sections: list[NavSectionDict]
    navmap_dict: dict[str, ResolvedNavValue]  # parsed __navmap__ (may be {})


def _rel(p: Path) -> str:
    """Return ``p`` relative to the repository root when possible."""
    try:
        return str(p.relative_to(REPO))
    except Exception:
        return str(p)


def _git_sha() -> str:
    """Return the current Git commit hash, falling back to environment overrides."""
    if G_SHA:
        return G_SHA
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
        ).strip()
    except Exception:
        return "HEAD"


def _gh_link(path: Path, start: int | None, end: int | None) -> str | None:
    """Commit-stable GitHub permalink using #L anchors."""
    if not (G_ORG and G_REPO):
        return None
    sha = _git_sha()
    frag = ""
    if start and end and end >= start:
        frag = f"#L{start}-L{end}"
    elif start:
        frag = f"#L{start}"
    return f"https://github.com/{G_ORG}/{G_REPO}/blob/{sha}/{_rel(path)}{frag}"


def _editor_link(path: Path, line: int | None = None) -> str | None:
    """Build an editor deep link respecting ``DOCS_EDITOR`` mode."""
    if EDITOR_MODE == "relative":
        try:
            rel_path = path.relative_to(REPO).as_posix()
        except ValueError:
            rel_path = path.as_posix()
        suffix = f":{line}:1" if line else ""
        return f"./{rel_path}{suffix}"
    if EDITOR_MODE == "vscode":
        abs_path = path if path.is_absolute() else (REPO / path).resolve()
        suffix = f":{line}:1" if line else ""
        return f"vscode://file/{abs_path}{suffix}"
    return None


def _module_name(py: Path) -> str | None:
    """Return the dotted module name for ``py``.

    Parameters
    ----------
    py : Path
        Description.

    Returns
    -------
    str | None
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _module_name(...)
    """
    if py.suffix != ".py":
        return None
    try:
        rel = py.relative_to(SRC)
    except Exception:
        return None
    parts = list(rel.with_suffix("").parts)
    if not parts:
        return None
    return ".".join(parts)


def _literal_list_of_strs(node: ast.AST | None) -> list[str] | None:
    """Best-effort get list/tuple of strings from AST node."""
    if node is None:
        return None
    if isinstance(node, (ast.List, ast.Tuple)):
        vals = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                vals.append(elt.value)
            elif isinstance(elt, ast.Name) and IDENT_RE.match(elt.id):
                vals.append(elt.id)
            else:
                return None
        return vals
    return None


def _parse_module(py: Path) -> ast.Module | None:
    """Return the parsed AST for ``py`` or ``None`` when parsing fails."""
    try:
        source = py.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _maybe_extract_exports(node: ast.AST) -> list[str] | None:
    """Return a literal ``__all__`` declaration when present."""
    return _literal_list_of_strs(node)


def _maybe_extract_navmap(node: ast.AST) -> dict[str, NavTree] | None:
    """Return a literal ``__navmap__`` dictionary when supported constructs are used."""
    try:
        literal = _literal_eval_navmap(node)
    except NavmapLiteralError:
        return None
    if isinstance(literal, dict):
        return literal
    return None


def _process_assign_targets(
    node: ast.Assign, exports: list[str], nav_literal: dict[str, NavTree] | None
) -> tuple[list[str], dict[str, NavTree] | None]:
    """Update exports/navmap literals based on assignment targets."""
    target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
    if "__all__" in target_names:
        extracted = _maybe_extract_exports(node.value)
        if extracted is not None:
            exports = extracted
    if "__navmap__" in target_names:
        nav_candidate = _maybe_extract_navmap(node.value)
        if nav_candidate is not None:
            nav_literal = nav_candidate
    return exports, nav_literal


def _process_ann_assign(
    node: ast.AnnAssign, exports: list[str], nav_literal: dict[str, NavTree] | None
) -> tuple[list[str], dict[str, NavTree] | None]:
    """Update exports/navmap literals for annotated assignments."""
    target = node.target
    if not isinstance(target, ast.Name) or node.value is None:
        return exports, nav_literal
    if target.id == "__all__":
        extracted = _maybe_extract_exports(node.value)
        if extracted is not None:
            exports = extracted
    if target.id == "__navmap__":
        nav_candidate = _maybe_extract_navmap(node.value)
        if nav_candidate is not None:
            nav_literal = nav_candidate
    return exports, nav_literal


def _gather_module_literals(module: ast.Module) -> tuple[list[str], dict[str, NavTree] | None]:
    """Return literal ``__all__`` and ``__navmap__`` values declared in ``module``."""
    exports: list[str] = []
    nav_literal: dict[str, NavTree] | None = None
    for node in module.body:
        if isinstance(node, ast.Assign):
            exports, nav_literal = _process_assign_targets(node, exports, nav_literal)
            continue
        if isinstance(node, ast.AnnAssign):
            exports, nav_literal = _process_ann_assign(node, exports, nav_literal)
    return exports, nav_literal


def _dedupe_exports(exports: Sequence[str]) -> list[str]:
    """Return exports with original ordering preserved and duplicates removed."""
    return list(dict.fromkeys(exports))


def _exports_from_navmap(nav_literal: dict[str, NavTree]) -> list[str]:
    """Return export hints defined within the navmap literal."""
    raw_exports = nav_literal.get("exports")
    if not isinstance(raw_exports, list):
        return []
    string_exports = [item for item in raw_exports if isinstance(item, str)]
    return _dedupe_exports(string_exports)


def _resolve_navmap_literal(
    nav_literal: dict[str, NavTree], exports: Sequence[str]
) -> dict[str, ResolvedNavValue]:
    """Return the navmap literal with placeholders expanded."""
    try:
        resolved = _replace_placeholders(nav_literal, exports)
    except NavmapPlaceholderError:
        return {}
    if isinstance(resolved, dict):
        return resolved
    return {}


def _resolve_navmap(
    nav_literal: dict[str, NavTree] | None, exports: list[str]
) -> tuple[dict[str, ResolvedNavValue], list[str]]:
    """Resolve navmap placeholders and potentially update exports."""
    if not nav_literal:
        return {}, exports
    exports_hint = exports or _exports_from_navmap(nav_literal)
    nav_map = _resolve_navmap_literal(nav_literal, exports_hint)
    if not nav_map:
        return {}, exports
    nav_exports = nav_map.get("exports")
    if isinstance(nav_exports, list):
        exports = _dedupe_exports([item for item in nav_exports if isinstance(item, str)])
    return nav_map, exports


def _parse_py(py: Path) -> tuple[dict[str, ResolvedNavValue], list[str]]:
    """Return (__navmap__ dict, __all__ list or [])."""
    module = _parse_module(py)
    if module is None:
        return {}, []
    exports, nav_literal = _gather_module_literals(module)
    exports = _dedupe_exports(exports)
    nav_map, exports = _resolve_navmap(nav_literal, exports)
    return nav_map, exports


def _scan_inline_markers(py: Path) -> tuple[dict[str, int], dict[str, int]]:
    """Return (sections, anchors) with 1-based line numbers."""
    sections: dict[str, int] = {}
    anchors: dict[str, int] = {}
    try:
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            m = SECTION_RE.match(line)
            if m:
                sid = m.group(1)
                sections[sid] = i
            m = ANCHOR_RE.match(line)
            if m:
                anchors[m.group(1)] = i
    except Exception:
        pass
    return sections, anchors


def _kebab(s: str) -> str:
    """Normalize ``s`` into a kebab-case identifier string."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9-]", "-", s)
    return re.sub(r"-{2,}", "-", s).strip("-")


def _collect_module(py: Path) -> ModuleInfo | None:
    """Parse ``py`` and return gathered navmap metadata if it is a module."""
    mod = _module_name(py)
    if not mod:
        return None
    navmap_dict, exports = _parse_py(py)
    sections, anchors = _scan_inline_markers(py)

    # Normalize sections to kebab-case and ensure symbol lists are unique & stable
    nav_sections: list[NavSectionDict] = []
    raw_sections = navmap_dict.get("sections")
    if isinstance(raw_sections, list):
        for section in raw_sections:
            if not isinstance(section, dict):
                continue
            raw_id = section.get("id", "")
            sid = _kebab(str(raw_id))
            raw_symbols = section.get("symbols", [])
            if isinstance(raw_symbols, list):
                filtered_symbols = _dedupe_exports(
                    [sym for sym in raw_symbols if isinstance(sym, str)]
                )
            else:
                filtered_symbols = []
            nav_sections.append({"id": sid, "symbols": filtered_symbols})
    # Stable, deduped exports
    raw_exports = navmap_dict.get("exports")
    if isinstance(raw_exports, list):
        nav_exports = _dedupe_exports([item for item in raw_exports if isinstance(item, str)])
    else:
        nav_exports = _dedupe_exports(exports)
    exports = nav_exports

    return ModuleInfo(
        module=mod,
        path=py,
        exports=exports,
        sections=sections,
        anchors=anchors,
        nav_sections=nav_sections,
        navmap_dict=navmap_dict,
    )


def _build_links(info: ModuleInfo) -> dict[str, str]:
    """Return source links for a module entry based on configured link mode."""
    links: dict[str, str] = {}
    if LINK_MODE in ("editor", "both"):
        editor_link = _editor_link(info.path)
        if editor_link:
            links["source"] = editor_link
    if LINK_MODE in ("github", "both"):
        github_link = _gh_link(info.path, None, None)
        if github_link:
            links["github"] = github_link
    return links


def _module_meta_fields(navmap: dict[str, ResolvedNavValue]) -> ModuleMetaDict:
    """Return module-level metadata fields."""
    module_meta: ModuleMetaDict = {}
    owner = navmap.get("owner")
    if isinstance(owner, str) and owner:
        module_meta["owner"] = owner
    stability = navmap.get("stability")
    if isinstance(stability, str) and stability:
        module_meta["stability"] = stability
    since = navmap.get("since")
    if isinstance(since, str) and since:
        module_meta["since"] = since
    deprecated_in = navmap.get("deprecated_in")
    if isinstance(deprecated_in, str) and deprecated_in:
        module_meta["deprecated_in"] = deprecated_in
    return module_meta


def _symbol_meta_fields(navmap: dict[str, ResolvedNavValue]) -> dict[str, SymbolMetaDict]:
    """Return per-symbol metadata declarations."""
    raw_symbols = navmap.get("symbols")
    if not isinstance(raw_symbols, dict):
        return {}
    symbols_meta: dict[str, SymbolMetaDict] = {}
    for name, payload in raw_symbols.items():
        if not isinstance(name, str) or not isinstance(payload, dict):
            continue
        filtered: SymbolMetaDict = {}
        owner = payload.get("owner")
        if isinstance(owner, str) and owner:
            filtered["owner"] = owner
        stability = payload.get("stability")
        if isinstance(stability, str) and stability:
            filtered["stability"] = stability
        since = payload.get("since")
        if isinstance(since, str) and since:
            filtered["since"] = since
        deprecated_in = payload.get("deprecated_in")
        if isinstance(deprecated_in, str) and deprecated_in:
            filtered["deprecated_in"] = deprecated_in
        symbols_meta[name] = filtered
    return symbols_meta


def _merge_symbol_meta(
    module_meta: ModuleMetaDict, symbol_meta: SymbolMetaDict | None
) -> SymbolMetaDict:
    """Return symbol metadata with module defaults applied."""
    merged: SymbolMetaDict = {}
    owner = symbol_meta.get("owner") if symbol_meta else None
    if not isinstance(owner, str) or not owner:
        owner = module_meta.get("owner")
    if isinstance(owner, str) and owner:
        merged["owner"] = owner
    stability = symbol_meta.get("stability") if symbol_meta else None
    if not isinstance(stability, str) or not stability:
        stability = module_meta.get("stability")
    if isinstance(stability, str) and stability:
        merged["stability"] = stability
    since = symbol_meta.get("since") if symbol_meta else None
    if not isinstance(since, str) or not since:
        since = module_meta.get("since")
    if isinstance(since, str) and since:
        merged["since"] = since
    deprecated_in = symbol_meta.get("deprecated_in") if symbol_meta else None
    if not isinstance(deprecated_in, str) or not deprecated_in:
        deprecated_in = module_meta.get("deprecated_in")
    if isinstance(deprecated_in, str) and deprecated_in:
        merged["deprecated_in"] = deprecated_in
    return merged


def _apply_symbol_defaults(
    symbols_meta: dict[str, SymbolMetaDict],
    module_meta: ModuleMetaDict,
    exports: Sequence[str],
) -> dict[str, SymbolMetaDict]:
    """Apply module-level defaults to symbol metadata."""
    if symbols_meta:
        return {name: _merge_symbol_meta(module_meta, meta) for name, meta in symbols_meta.items()}
    return {export: _merge_symbol_meta(module_meta, None) for export in exports}


def _string_list(value: ResolvedNavValue | None) -> list[str]:
    """Return a string-only list coerced from ``value``."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _string_value(value: ResolvedNavValue | None) -> str:
    """Return ``value`` when it is a string, otherwise an empty string."""
    if isinstance(value, str):
        return value
    return ""


def _module_entry(info: ModuleInfo) -> ModuleEntryDict:
    """Return the serialized navmap entry for ``info``."""
    module_meta = _module_meta_fields(info.navmap_dict)
    symbols_meta = _symbol_meta_fields(info.navmap_dict)
    symbols = _apply_symbol_defaults(symbols_meta, module_meta, info.exports)
    return {
        "path": _rel(info.path),
        "exports": _dedupe_exports(info.exports),
        "sections": info.nav_sections,
        "section_lines": info.sections,
        "anchors": info.anchors,
        "links": _build_links(info),
        "meta": symbols,
        "module_meta": module_meta,
        "tags": _string_list(info.navmap_dict.get("tags")),
        "synopsis": _string_value(info.navmap_dict.get("synopsis")),
        "see_also": _string_list(info.navmap_dict.get("see_also")),
        "deps": _string_list(info.navmap_dict.get("deps")),
    }


def _collect_module_entries(files: Sequence[Path]) -> dict[str, ModuleEntryDict]:
    """Return serialized module entries for the provided Python files."""
    modules: dict[str, ModuleEntryDict] = {}
    for py in files:
        info = _collect_module(py)
        if info is None:
            continue
        modules[info.module] = _module_entry(info)
    return modules


def _discover_py_files(root: Path = SRC) -> list[Path]:
    """Return every Python source file under ``root`` sorted lexicographically."""
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def build_index(root: Path = SRC, json_path: Path | None = None) -> NavIndexDict:
    """Build the navmap index and optionally persist it to disk.

    Parameters
    ----------
    root : Path, optional
        Directory to scan for Python modules, by default ``SRC``.
    json_path : Path | None
        Override destination for the generated JSON document, by default ``None`` which causes ``INDEX_PATH`` to be used.

    Returns
    -------
    NavIndexDict
        Serialized navmap index keyed by dotted module name.

    Examples
    --------
    >>> from tools.navmap.build_navmap import build_index
    >>> result = build_index()
    >>> result  # doctest: +ELLIPSIS
    """
    files = _discover_py_files(root)
    modules = _collect_module_entries(files)
    data: NavIndexDict = {
        "commit": _git_sha(),
        "policy_version": "1",
        "link_mode": LINK_MODE,
        "modules": modules,
    }

    # Write
    out = json_path or INDEX_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the navmap index and optionally override the output path."""
    parser = argparse.ArgumentParser(description="Build the navigation index JSON")
    parser.add_argument(
        "--write",
        type=Path,
        metavar="PATH",
        help="Destination for the generated navmap JSON",
        default=None,
    )
    args = parser.parse_args(argv)

    target = args.write if args.write is not None else None
    build_index(json_path=target)
    destination = target or INDEX_PATH
    print(f"[navmap] regenerated {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
