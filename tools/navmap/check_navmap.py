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
from collections.abc import Sequence
from pathlib import Path
from functools import singledispatch
from typing import TypedDict

from packaging.version import InvalidVersion, Version

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

    __slots__ = ("template",)

    def __repr__(self) -> str:
        return "AllPlaceholder()"


class AllDictTemplate:
    """Sentinel for ``{name: TEMPLATE for name in __all__}`` structures."""

    __slots__ = ("template",)

    def __init__(self, template: NavTree) -> None:
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
type ResolvedNavValue = NavPrimitive | list["ResolvedNavValue"] | dict[str, "ResolvedNavValue"] | set[str]

PLACEHOLDER_ALL = AllPlaceholder()


class SymbolMetaDict(TypedDict, total=False):
    """Metadata requirements for a single symbol."""

    owner: str
    stability: str
    since: str
    deprecated_in: str


class ModuleEntryDict(TypedDict, total=False):
    """Minimal navmap subset used by the checker."""

    exports: list[str]
    sections: list[dict[str, ResolvedNavValue]]
    symbols: dict[str, SymbolMetaDict]


@singledispatch
def _eval_nav_literal(node: ast.AST) -> NavTree:
    """Return the navmap literal represented by ``node``."""
    message = f"Unsupported navmap literal node: {ast.dump(node)}"
    raise NavmapLiteralError(message)


@_eval_nav_literal.register
def _(node: ast.Constant) -> NavTree:
    value = node.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    message = f"Unsupported constant in navmap literal: {value!r}"
    raise NavmapLiteralError(message)


@_eval_nav_literal.register
def _(node: ast.Name) -> NavTree:
    if node.id == "__all__":
        return PLACEHOLDER_ALL
    message = f"Unsupported name in navmap literal: {node.id!r}"
    raise NavmapLiteralError(message)


def _eval_sequence(nodes: Sequence[ast.AST]) -> list[NavTree]:
    """Return a list of evaluated navmap literals for ``nodes``."""
    return [_literal_eval_navmap(child) for child in nodes]


@_eval_nav_literal.register
def _(node: ast.List) -> NavTree:
    return _eval_sequence(node.elts)


@_eval_nav_literal.register
def _(node: ast.Tuple) -> NavTree:
    return _eval_sequence(node.elts)


@_eval_nav_literal.register
def _(node: ast.Set) -> NavTree:
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


@_eval_nav_literal.register
def _(node: ast.Dict) -> NavTree:
    return _eval_dict(node)


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


@_eval_nav_literal.register
def _(node: ast.DictComp) -> NavTree:
    return _eval_dict_comprehension(node)


def _literal_eval_navmap(node: ast.AST | None) -> NavTree:
    """Evaluate ``node`` into a navmap literal."""
    if node is None:
        message = "Navmap literal must not be empty."
        raise NavmapLiteralError(message)
    return _eval_nav_literal(node)


def _dedupe_str_list(items: Sequence[str]) -> list[str]:
    """Return ``items`` with original ordering and duplicates removed."""
    return list(dict.fromkeys(items))


def _expand_all_placeholder(exports: Sequence[str]) -> list[str]:
    """Return concrete export lists for ``__all__`` placeholders."""
    return _dedupe_str_list(exports)


def _expand_dict_template(
    template: NavTree, exports: Sequence[str]
) -> dict[str, ResolvedNavValue]:
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


def _read_text(py: Path) -> list[str]:
    """Return file contents as a list of lines, or an empty list on failure."""
    try:
        return py.read_text(encoding="utf-8").splitlines()
    except Exception:
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


def _parse_navmap_dict(py: Path) -> dict[str, Any]:
    """Parse a module's __navmap__ by a quick-and-safe literal eval.

    This checker only needs fields existence; deep evaluation is unnecessary.
    """
    import ast

    try:
        tree = ast.parse(py.read_text(encoding="utf-8"))
    except Exception:
        return {}
    result: dict[str, Any] = {}

    def _expand_placeholders(value: Any, exports: list[str]) -> Any:
        """Replace __all__ placeholders within navmap literals."""
        if isinstance(value, str) and value == "__all__":
            return list(dict.fromkeys(exports))
        if isinstance(value, list):
            expanded_list: list[Any] = []
            for entry in value:
                replaced = _expand_placeholders(entry, exports)
                if isinstance(replaced, list):
                    expanded_list.extend(replaced)
                else:
                    expanded_list.append(replaced)
            return expanded_list
        if isinstance(value, set):
            expanded_set: set[Any] = set()
            for entry in value:
                replaced = _expand_placeholders(entry, exports)
                if isinstance(replaced, list):
                    expanded_set.update(replaced)
                else:
                    expanded_set.add(replaced)
            return expanded_set
        if isinstance(value, dict):
            return {k: _expand_placeholders(v, exports) for k, v in value.items()}
        if isinstance(value, AllDictTemplate):
            template = value.template
            template_results: dict[str, Any] = {}
            for name in exports:
                mapped = _expand_placeholders(template, exports)
                if isinstance(mapped, dict):
                    template_results[name] = mapped
                else:
                    template_results[name] = mapped
            return template_results
        return value

    def _safe_eval(value: ast.AST | None) -> Any:
        """Safe eval.

        Parameters
        ----------
        value : ast.AST
            Description.

        Returns
        -------
        Any
            Description.

        Raises
        ------
        Exception
            Description.

        Examples
        --------
        >>> _safe_eval(...)
        """
        if value is None:
            raise ValueError("unsupported empty literal")
        if isinstance(value, ast.Constant):
            return value.value
        if isinstance(value, ast.Name) and value.id == "__all__":
            return "__all__"
        if isinstance(value, (ast.List, ast.Tuple)):
            return [_safe_eval(elt) for elt in value.elts]
        if isinstance(value, ast.Set):
            return {_safe_eval(elt) for elt in value.elts}
        if isinstance(value, ast.Dict):
            return {
                _safe_eval(k): _safe_eval(v) for k, v in zip(value.keys, value.values, strict=False)
            }
        if isinstance(value, ast.DictComp):
            if len(value.generators) != 1:
                raise ValueError("unsupported dict comprehension")
            comp = value.generators[0]
            if not isinstance(comp.target, ast.Name) or not isinstance(comp.iter, ast.Name):
                raise ValueError("unsupported dict comprehension")
            if comp.iter.id != "__all__":
                raise ValueError("unsupported dict comprehension iterator")
            template = _safe_eval(value.value)
            return AllDictTemplate(template)
        raise ValueError(ast.dump(value))

    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "__navmap__" for t in node.targets):
                try:
                    evaluated = _safe_eval(node.value)
                    if isinstance(evaluated, dict):
                        result = evaluated
                except Exception:
                    pass
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "__navmap__":
                if node.value is None:
                    continue
                try:
                    evaluated = _safe_eval(node.value)
                    if isinstance(evaluated, dict):
                        result = evaluated
                except Exception:
                    pass
    exports = _parse_all(py)
    if not exports and isinstance(result, dict):
        raw_exports = result.get("exports")
        if isinstance(raw_exports, list):
            exports = [x for x in raw_exports if isinstance(x, str)]
    if exports and result:
        result = _expand_placeholders(result, exports)
        if isinstance(result.get("exports"), list):
            result["exports"] = list(
                dict.fromkeys(x for x in result["exports"] if isinstance(x, str))
            )
    return result


def _parse_all(py: Path) -> list[str]:
    """Return the literal ``__all__`` sequence when it can be safely evaluated."""
    import ast

    try:
        tree = ast.parse(py.read_text(encoding="utf-8"))
    except Exception:
        return []

    def _literal(node: ast.AST | None) -> list[str] | None:
        """Literal.

        Parameters
        ----------
        node : ast.AST
            Description.

        Returns
        -------
        list[str] | None
            Description.

        Raises
        ------
        Exception
            Description.

        Examples
        --------
        >>> _literal(...)
        """
        if node is None:
            return None
        if isinstance(node, (ast.List, ast.Tuple)):
            vals: list[str] = []
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    vals.append(elt.value)
                elif isinstance(elt, ast.Name) and IDENT_RE.match(elt.id):
                    vals.append(elt.id)
                else:
                    return None
            return vals
        return None

    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                vals = _literal(node.value)
                if vals is not None:
                    return vals
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "__all__":
                vals = _literal(node.value)
                if vals is not None:
                    return vals
    return []


def _exports_for(py: Path, nav: dict[str, Any]) -> list[str]:
    """Derive the export list from ``__navmap__`` or ``__all__`` definitions."""
    nav_exports = nav.get("exports")
    if isinstance(nav_exports, list):
        vals = [x for x in nav_exports if isinstance(x, str)]
        if vals:
            return list(dict.fromkeys(vals))
    all_list = _parse_all(py)
    if all_list:
        return list(dict.fromkeys(all_list))
    return []


def _module_path(py: Path) -> str | None:
    """Return the dotted module path for ``py`` within ``src/`` if possible."""
    try:
        rel = py.relative_to(SRC)
    except Exception:
        return None
    if rel.suffix != ".py":
        return None
    return ".".join(rel.with_suffix("").parts)


def _validate_pep440(field_val: Any) -> str | None:
    """Validate PEP 440 version strings and report an error message when invalid."""
    if field_val is None or field_val == "":
        return None
    if Version is None:
        return "packaging not installed (cannot validate PEP 440)"
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

    # Sections: first must be public-api; kebab-case ids; anchors present for listed symbols
    sections = nav.get("sections", [])
    if sections:
        first = sections[0].get("id")
        if first != "public-api":
            errs.append(f"{py}: first navmap section must have id 'public-api'")
        for sec in sections:
            sid = sec.get("id", "")
            if sid and not SLUG_RE.match(sid):
                errs.append(f"{py}: section id '{sid}' is not kebab-case")
            for sym in sec.get("symbols", []):
                if not IDENT_RE.match(sym):
                    errs.append(f"{py}: invalid symbol name '{sym}' in section '{sid}'")
                elif sym not in anchors_inline:
                    errs.append(f"{py}: missing [nav:anchor] for section symbol '{sym}'")

    # Exports: if nav declares, it must match the actual exports setwise
    nav_exports = nav.get("exports", [])
    if isinstance(nav_exports, list):
        if set(x for x in nav_exports if isinstance(x, str)) != set(exports):
            errs.append(f"{py}: __navmap__['exports'] does not match __all__/exports set")

    # Per-export meta requirements (owner, stability, since/dep PEP 440)
    meta: dict[str, Any] = nav.get("symbols", {}) or {}
    for name in sorted(exports):
        m = meta.get(name, {})
        stab = m.get("stability")
        owner = m.get("owner")
        if stab not in STABILITY:
            errs.append(f"{py}: symbol '{name}' missing/invalid stability (got {stab!r})")
        if not owner:
            errs.append(f"{py}: symbol '{name}' missing owner (e.g., '@team')")
        # Versions
        since = m.get("since")
        deprec = m.get("deprecated_in")
        e1 = _validate_pep440(since)
        if e1:
            errs.append(f"{py}: symbol '{name}' since invalid: {e1}")
        e2 = _validate_pep440(deprec)
        if e2:
            errs.append(f"{py}: symbol '{name}' deprecated_in invalid: {e2}")
        if Version is not None and since and deprec:
            try:
                if Version(str(deprec)) < Version(str(since)):
                    errs.append(f"{py}: symbol '{name}' deprecated_in ({deprec}) < since ({since})")
            except InvalidVersion:
                pass

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
    ...
    """
    errors: list[str] = []
    for py in sorted(SRC.rglob("*.py")):
        errors.extend(_inspect(py))

    if errors:
        print("\n".join(errors))
        return 1

    # Round-trip check: compare freshly built JSON to inline markers
    try:
        # Local import to avoid creating a hard dependency in module scope
        from tools.navmap.build_navmap import build_index
    except Exception as e:  # pragma: no cover
        print(f"navmap check: unable to import build_navmap for round-trip ({e})", file=sys.stderr)
        return 1

    index = build_index(json_path=INDEX)  # also refreshes file on disk
    rt_errs: list[str] = []
    for mod, entry in (index.get("modules") or {}).items():
        p = REPO / entry.get("path", "")
        lines = _read_text(p)
        # Sections at recorded lines
        for sid, lineno in (entry.get("section_lines") or {}).items():
            if lineno < 1 or lineno > len(lines) or not SECTION_RE.match(lines[lineno - 1]):
                rt_errs.append(f"{p}: round-trip mismatch for section '{sid}' at line {lineno}")
        # Anchors at recorded lines
        for sym, lineno in (entry.get("anchors") or {}).items():
            if lineno < 1 or lineno > len(lines) or not ANCHOR_RE.match(lines[lineno - 1]):
                rt_errs.append(f"{p}: round-trip mismatch for anchor '{sym}' at line {lineno}")

    if rt_errs:
        print("\n".join(rt_errs))
        return 1

    print("navmap check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
