#!/usr/bin/env python
#!/usr/bin/env python3
"""Overview of check navmap.

This module bundles check navmap logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypedDict, cast

from packaging.version import InvalidVersion, Version

import tools.navmap.build_navmap as _build_navmap
from tools import get_logger

_BuildNavmapError = Exception

BuildNavmapError = cast(type[Exception], getattr(_build_navmap, "NavmapError", _BuildNavmapError))
build_index = cast(Callable[..., dict[str, object]], _build_navmap.build_index)

LOGGER = get_logger(__name__)

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
INDEX = REPO / "site" / "_build" / "navmap" / "navmap.json"

# Regexes
SECTION_RE = re.compile(r"^\s*#\s*\[nav:section\s+([a-z0-9]+(?:-[a-z0-9]+)*)\]\s*$")
ANCHOR_RE = re.compile(r"^\s*#\s*\[nav:anchor\s+([A-Za-z_]\w*)\]\s*$")
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")
STABILITY = {"stable", "beta", "experimental", "deprecated", "internal", "frozen"}


class NavmapError(Exception):
    """Base exception for navmap parsing issues."""


class NavmapLiteralError(NavmapError):
    """Raised when a navmap literal cannot be parsed safely."""


class NavmapPlaceholderError(NavmapError):
    """Raised when placeholder expansion fails."""


class AllPlaceholder:
    """Sentinel for ``__all__`` placeholders."""

    __slots__ = ()


class AllDictTemplate:
    """Sentinel for ``{name: TEMPLATE for name in __all__}`` structures."""

    __slots__ = ("template",)

    def __init__(self, template: NavTree) -> None:
        """Init  .

        Parameters
        ----------
        template : NavTree
            Description.

        Returns
        -------
        None
            Description.

        Raises
        ------
        Exception
            Description.

        Examples
        --------
        >>> __init__(...)
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
    NavPrimitive | list["ResolvedNavValue"] | dict[str, "ResolvedNavValue"] | set[str]
)

PLACEHOLDER_ALL = AllPlaceholder()


class SymbolMetaDict(TypedDict, total=False):
    """Metadata requirements for a single symbol."""

    owner: str
    stability: str
    since: str
    deprecated_in: str


class ModuleEntryDict(TypedDict, total=False):
    """Minimal navmap subset used by the checker."""

    path: str
    exports: list[str]
    sections: list[dict[str, ResolvedNavValue]]
    section_lines: dict[str, int]
    anchors: dict[str, int]
    symbols: dict[str, SymbolMetaDict]


class NavIndexDict(TypedDict, total=False):
    """Serialized navmap index structure."""

    commit: str
    policy_version: str
    link_mode: str
    modules: dict[str, ModuleEntryDict]


def _eval_nav_literal(node: ast.AST) -> NavTree:
    """Return the navmap literal represented by ``node``."""
    if isinstance(node, ast.Constant):
        return _eval_constant(node)
    if isinstance(node, ast.Name):
        return _eval_name(node)
    if isinstance(node, (ast.List, ast.Tuple)):
        return _eval_sequence(node.elts)
    if isinstance(node, ast.Set):
        return _eval_set(node)
    if isinstance(node, ast.Dict):
        return _eval_dict(node)
    if isinstance(node, ast.DictComp):
        return _eval_dict_comprehension(node)
    message = f"Unsupported navmap literal node: {ast.dump(node)}"
    raise NavmapLiteralError(message)


def _eval_constant(node: ast.Constant) -> NavTree:
    """Return the literal value encoded by ``node`` when supported."""
    value = node.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    message = f"Unsupported constant in navmap literal: {value!r}"
    raise NavmapLiteralError(message)


def _eval_name(node: ast.Name) -> NavTree:
    """Resolve placeholder names used inside navmap literals."""
    if node.id == "__all__":
        return PLACEHOLDER_ALL
    message = f"Unsupported name in navmap literal: {node.id!r}"
    raise NavmapLiteralError(message)


def _eval_sequence(nodes: Sequence[ast.AST]) -> list[NavTree]:
    """Return a list of evaluated navmap literals for ``nodes``."""
    return [_literal_eval_navmap(child) for child in nodes]


def _eval_set(node: ast.Set) -> set[NavTree]:
    """Evaluate a set literal into navmap values."""
    return {_literal_eval_navmap(elt) for elt in node.elts}


def _eval_dict(node: ast.Dict) -> dict[str, NavTree]:
    """Evaluate a dict literal, enforcing string keys."""
    result: dict[str, NavTree] = {}
    for key_node, value_node in zip(node.keys, node.values, strict=False):
        key = _literal_eval_navmap(key_node)
        if not isinstance(key, str):
            message = "Navmap dictionary keys must be strings."
            raise NavmapLiteralError(message)
        result[key] = _literal_eval_navmap(value_node)
    return result


def _eval_dict_comprehension(node: ast.DictComp) -> AllDictTemplate:
    """Evaluate supported dict comprehensions into ``AllDictTemplate`` placeholders."""
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


def _literal_eval_navmap(node: ast.AST | None) -> NavTree:
    """Evaluate ``node`` into a navmap literal."""
    if node is None:
        message = "Navmap literal must not be empty."
        raise NavmapLiteralError(message)
    return _eval_nav_literal(node)


def _dedupe_str_list(items: Sequence[str]) -> list[str]:
    """Return ``items`` with original ordering and duplicates removed."""
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _expand_all_placeholder(exports: Sequence[str]) -> list[ResolvedNavValue]:
    """Return concrete export lists for ``__all__`` placeholders."""
    return cast(list[ResolvedNavValue], _dedupe_str_list(exports))


def _expand_dict_template(template: NavTree, exports: Sequence[str]) -> dict[str, ResolvedNavValue]:
    """Expand ``AllDictTemplate`` placeholders."""
    expanded: dict[str, ResolvedNavValue] = {}
    for name in exports:
        expanded[name] = _expand_nav_value(template, exports)
    return expanded


def _expand_list(values: Sequence[NavTree], exports: Sequence[str]) -> list[ResolvedNavValue]:
    """Expand navmap lists, flattening nested lists that arise from placeholders."""
    expanded: list[ResolvedNavValue] = []
    for entry in values:
        resolved = _expand_nav_value(entry, exports)
        if isinstance(resolved, list):
            expanded.extend(resolved)
        else:
            expanded.append(resolved)
    return expanded


def _expand_dict(values: dict[str, NavTree], exports: Sequence[str]) -> dict[str, ResolvedNavValue]:
    """Expand navmap dict values recursively."""
    return {key: _expand_nav_value(sub_value, exports) for key, sub_value in values.items()}


def _expand_set(values: set[NavTree], exports: Sequence[str]) -> set[str]:
    """Expand navmap sets and ensure all members resolve to strings."""
    resolved: set[str] = set()
    for entry in values:
        expanded = _expand_nav_value(entry, exports)
        if isinstance(expanded, list):
            for item in expanded:
                if isinstance(item, str):
                    resolved.add(item)
                else:
                    message = "Navmap sets may only contain strings after expansion."
                    raise NavmapPlaceholderError(message)
        elif isinstance(expanded, str):
            resolved.add(expanded)
        else:
            message = "Navmap sets must resolve to strings."
            raise NavmapPlaceholderError(message)
    return resolved


def _expand_nav_value(value: NavTree, exports: Sequence[str]) -> ResolvedNavValue:
    """Expand navmap placeholders for ``value`` using ``exports``."""
    if isinstance(value, AllPlaceholder):
        return _expand_all_placeholder(exports)
    if isinstance(value, AllDictTemplate):
        return _expand_dict_template(value.template, exports)
    if isinstance(value, list):
        return _expand_list(value, exports)
    if isinstance(value, dict):
        return _expand_dict(value, exports)
    if isinstance(value, set):
        return _expand_set(value, exports)
    return value


def _parse_module(py: Path) -> ast.Module | None:
    """Return an AST for ``py`` or ``None`` when parsing fails."""
    try:
        source = py.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _extract_navmap_literal(module: ast.Module) -> dict[str, NavTree] | None:
    """Return the literal ``__navmap__`` declaration from ``module``."""
    nav_literal: dict[str, NavTree] | None = None
    for node in module.body:
        if isinstance(node, ast.Assign):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "__navmap__" in targets:
                try:
                    candidate = _literal_eval_navmap(node.value)
                except NavmapLiteralError:
                    continue
                if isinstance(candidate, dict):
                    nav_literal = candidate
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id != "__navmap__" or node.value is None:
                continue
            try:
                candidate = _literal_eval_navmap(node.value)
            except NavmapLiteralError:
                continue
            if isinstance(candidate, dict):
                nav_literal = candidate
    return nav_literal


def _exports_from_nav_literal(nav_literal: dict[str, NavTree]) -> list[str]:
    """Return export hints embedded within ``nav_literal``."""
    exports_literal = nav_literal.get("exports")
    if not isinstance(exports_literal, list):
        return []
    exports = [item for item in exports_literal if isinstance(item, str)]
    return _dedupe_str_list(exports)


def _resolve_navmap_literal(
    nav_literal: dict[str, NavTree], exports: Sequence[str]
) -> dict[str, ResolvedNavValue]:
    """Expand placeholders and normalize exports inside ``nav_literal``."""
    try:
        resolved = _expand_nav_value(nav_literal, exports)
    except NavmapPlaceholderError:
        return {}
    if not isinstance(resolved, dict):
        return {}
    nav_exports = resolved.get("exports")
    if isinstance(nav_exports, list):
        resolved["exports"] = cast(
            list[ResolvedNavValue],
            _dedupe_str_list([item for item in nav_exports if isinstance(item, str)]),
        )
    return resolved


def _read_text(py: Path) -> list[str]:
    """Return file contents as a list of lines, or an empty list on failure."""
    try:
        return py.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []


def _scan_inline(py: Path) -> tuple[dict[str, int], dict[str, int]]:
    """Return (sections, anchors) mapping to 1-based line numbers."""
    sections: dict[str, int] = {}
    anchors: dict[str, int] = {}
    for i, line in enumerate(_read_text(py), 1):
        m = SECTION_RE.match(line)
        if m:
            sections[m.group(1)] = i
        m = ANCHOR_RE.match(line)
        if m:
            anchors[m.group(1)] = i
    return sections, anchors


def _parse_navmap_dict(py: Path) -> dict[str, ResolvedNavValue]:
    """Return the literal ``__navmap__`` dictionary for ``py`` if one exists."""
    module = _parse_module(py)
    if module is None:
        return {}
    nav_literal = _extract_navmap_literal(module)
    if nav_literal is None:
        return {}
    exports = _parse_all(py)
    if not exports:
        exports = _exports_from_nav_literal(nav_literal)
    return _resolve_navmap_literal(nav_literal, exports)


def _literal_string_sequence(node: ast.AST | None) -> list[str] | None:
    """Return a list of identifier/constant strings from ``node`` when possible."""
    if node is None:
        return None
    if isinstance(node, (ast.List, ast.Tuple)):
        strings: list[str] = []
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                strings.append(element.value)
            elif isinstance(element, ast.Name) and IDENT_RE.match(element.id):
                strings.append(element.id)
            else:
                return None
        return strings
    return None


def _extract_all_literal(module: ast.Module) -> list[str]:
    """Return the literal ``__all__`` declaration within ``module`` when present."""
    for node in module.body:
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "__all__" in targets:
                value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "__all__":
                value = node.value
        if value is None:
            continue
        strings = _literal_string_sequence(value)
        if strings is not None:
            return _dedupe_str_list(strings)
    return []


def _parse_all(py: Path) -> list[str]:
    """Parse all.

    Parameters
    ----------
    py : Path
        Description.

    Returns
    -------
    list[str]
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _parse_all(...)
    """
    module = _parse_module(py)
    if module is None:
        return []
    return _extract_all_literal(module)


def _exports_for(py: Path, nav: dict[str, ResolvedNavValue]) -> list[str]:
    """Derive the export list from ``__navmap__`` or ``__all__`` definitions."""
    nav_exports = nav.get("exports")
    if isinstance(nav_exports, list):
        strings = [item for item in nav_exports if isinstance(item, str)]
        if strings:
            return _dedupe_str_list(strings)
    all_literal = _parse_all(py)
    if all_literal:
        return _dedupe_str_list(all_literal)
    return []


def _sections_list(value: ResolvedNavValue | None) -> list[dict[str, ResolvedNavValue]]:
    """Return section entries when ``value`` is a list of dictionaries."""
    if not isinstance(value, list):
        return []
    entries: list[dict[str, ResolvedNavValue]] = []
    for item in value:
        if isinstance(item, dict):
            entries.append(item)
    return entries


def _symbols_meta_dict(value: ResolvedNavValue | None) -> dict[str, SymbolMetaDict]:
    """Return symbol metadata entries as ``SymbolMetaDict`` instances."""
    if not isinstance(value, dict):
        return {}
    meta: dict[str, SymbolMetaDict] = {}
    for key, payload in value.items():
        if not isinstance(key, str) or not isinstance(payload, dict):
            continue
        entry: SymbolMetaDict = {}
        owner = payload.get("owner")
        if isinstance(owner, str) and owner:
            entry["owner"] = owner
        stability = payload.get("stability")
        if isinstance(stability, str) and stability:
            entry["stability"] = stability
        since = payload.get("since")
        if isinstance(since, str) and since:
            entry["since"] = since
        deprecated_in = payload.get("deprecated_in")
        if isinstance(deprecated_in, str) and deprecated_in:
            entry["deprecated_in"] = deprecated_in
        meta[key] = entry
    return meta


def _validate_sections(
    py: Path, sections_value: ResolvedNavValue | None, anchors_inline: dict[str, int]
) -> list[str]:
    """Validate navmap sections and inline anchor coverage."""
    errors: list[str] = []
    sections = _sections_list(sections_value)
    if not sections:
        return errors
    first_id = sections[0].get("id")
    if first_id != "public-api":
        errors.append(f"{py}: first navmap section must have id 'public-api'")
    for section in sections:
        sid = section.get("id")
        symbols_value = section.get("symbols")
        if not isinstance(sid, str) or not sid:
            continue
        if not SLUG_RE.match(sid):
            errors.append(f"{py}: section id '{sid}' is not kebab-case")
        if not isinstance(symbols_value, list):
            continue
        for symbol in symbols_value:
            if not isinstance(symbol, str) or not IDENT_RE.match(symbol):
                errors.append(f"{py}: invalid symbol name '{symbol}' in section '{sid}'")
            elif symbol not in anchors_inline:
                errors.append(f"{py}: missing [nav:anchor] for section symbol '{symbol}'")
    return errors


def _validate_exports_match(
    py: Path, declared_exports: ResolvedNavValue | None, exports: list[str]
) -> list[str]:
    """Validate that declared exports match the discovered export list."""
    if not isinstance(declared_exports, list):
        return []
    declared_set = {item for item in declared_exports if isinstance(item, str)}
    if declared_set == set(exports):
        return []
    return [f"{py}: __navmap__['exports'] does not match __all__/exports set"]


def _validate_symbol_meta(
    py: Path, exports: list[str], symbols_value: ResolvedNavValue | None
) -> list[str]:
    """Validate per-symbol metadata requirements."""
    errors: list[str] = []
    meta = _symbols_meta_dict(symbols_value)
    for name in sorted(exports):
        entry = meta.get(name, {})
        stability = entry.get("stability")
        owner = entry.get("owner")
        if stability not in STABILITY:
            errors.append(f"{py}: symbol '{name}' missing/invalid stability (got {stability!r})")
        if not owner:
            errors.append(f"{py}: symbol '{name}' missing owner (e.g., '@team')")
        since = entry.get("since")
        deprecated_in = entry.get("deprecated_in")
        error_since = _validate_pep440(since)
        if error_since:
            errors.append(f"{py}: symbol '{name}' since invalid: {error_since}")
        error_deprecated = _validate_pep440(deprecated_in)
        if error_deprecated:
            errors.append(f"{py}: symbol '{name}' deprecated_in invalid: {error_deprecated}")
        if Version is None or not since or not deprecated_in:
            continue
        try:
            if Version(str(deprecated_in)) < Version(str(since)):
                errors.append(
                    f"{py}: symbol '{name}' deprecated_in ({deprecated_in}) < since ({since})"
                )
        except InvalidVersion:
            continue
    return errors


def _collect_module_errors() -> list[str]:
    """Run navmap checks across the ``src/`` tree and collect errors."""
    errors: list[str] = []
    for py in sorted(SRC.rglob("*.py")):
        errors.extend(_inspect(py))
    return errors


def _round_trip_line_errors(
    file_path: Path,
    lines: list[str],
    mapping: dict[str, object],
    pattern: re.Pattern[str],
    label: str,
) -> list[str]:
    """Return mismatches for ``mapping`` entries compared against ``lines``."""
    errors: list[str] = []
    for key, value in mapping.items():
        if not isinstance(key, str) or not isinstance(value, int):
            continue
        if value < 1 or value > len(lines) or not pattern.match(lines[value - 1]):
            errors.append(f"{file_path}: round-trip mismatch for {label} '{key}' at line {value}")
    return errors


def _round_trip_errors(index: NavIndexDict | dict[str, object]) -> list[str]:
    """Validate round-trip data from ``build_navmap`` against source files."""
    modules = index.get("modules")
    if not isinstance(modules, dict):
        return []
    errors: list[str] = []
    for entry in modules.values():
        if not isinstance(entry, dict):
            continue
        path_value = entry.get("path")
        if not isinstance(path_value, str):
            continue
        file_path = REPO / path_value
        lines = _read_text(file_path)
        section_lines = entry.get("sectionLines") or entry.get("section_lines")
        if isinstance(section_lines, dict):
            errors.extend(
                _round_trip_line_errors(file_path, lines, section_lines, SECTION_RE, "section")
            )
        anchors = entry.get("anchors")
        if isinstance(anchors, dict):
            errors.extend(_round_trip_line_errors(file_path, lines, anchors, ANCHOR_RE, "anchor"))
    return errors


def _module_path(py: Path) -> str | None:
    """Return the dotted module path for ``py`` within ``src/`` if possible."""
    try:
        rel = py.relative_to(SRC)
    except ValueError:
        return None
    if rel.suffix != ".py":
        return None
    return ".".join(rel.with_suffix("").parts)


def _validate_pep440(field_val: object) -> str | None:
    """Validate PEP 440 version strings and report an error message when invalid."""
    if field_val is None:
        return None
    if isinstance(field_val, str) and not field_val.strip():
        return None
    try:
        Version(str(field_val))
    except InvalidVersion:
        return f"non-PEP440 version: {field_val!r}"
    else:
        return None


def _inspect(py: Path) -> list[str]:
    """Validate a module at ``py`` and return collected violation messages."""
    errs: list[str] = []
    nav = _parse_navmap_dict(py)
    _, anchors_inline = _scan_inline(py)
    exports = _exports_for(py, nav)

    # If module exports anything, __navmap__ must exist
    if exports and not nav:
        errs.append(f"{py}: module exports symbols but has no __navmap__")
        return errs

    errs.extend(_validate_sections(py, nav.get("sections"), anchors_inline))
    errs.extend(_validate_exports_match(py, nav.get("exports"), exports))
    errs.extend(_validate_symbol_meta(py, exports, nav.get("symbols")))

    return errs


def main(argv: list[str] | None = None) -> int:
    """Compute main.

    Carry out the main operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    argv : List[str] | None
        Optional parameter default ``None``. Description for ``argv``.

    Returns
    -------
    int
        Description of return value.

    Examples
    --------
    >>> from tools.navmap.check_navmap import main
    >>> result = main()
    >>> result  # doctest: +ELLIPSIS
    """
    del argv
    errors = _collect_module_errors()

    if errors:
        LOGGER.error("\n".join(errors))
        return 1

    # Round-trip check: compare freshly built JSON to inline markers
    try:
        index = build_index(json_path=INDEX)
    except BuildNavmapError:
        LOGGER.exception("navmap check: build_navmap failed during round-trip")
        return 1
    rt_errs = _round_trip_errors(index)

    if rt_errs:
        LOGGER.error("\n".join(rt_errs))
        return 1

    LOGGER.info("navmap check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
