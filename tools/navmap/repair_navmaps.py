#!/usr/bin/env python
"""Overview of repair navmaps.

This module bundles repair navmaps logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from pprint import pformat
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"

try:
    from tools.navmap.build_navmap import ModuleInfo, _collect_module
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tools.navmap.build_navmap import ModuleInfo, _collect_module


def _collect_modules(root: Path) -> list[ModuleInfo]:
    """Collect modules.

    Parameters
    ----------
    root : Path
        Description.

    Returns
    -------
    list[ModuleInfo]
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _collect_modules(...)
    """
    modules: list[ModuleInfo] = []
    for py in sorted(root.rglob("*.py")):
        info = _collect_module(py)
        if info:
            modules.append(info)
    return modules


def _load_tree(path: Path) -> ast.Module:
    """Load tree.

    Parameters
    ----------
    path : Path
        Description.

    Returns
    -------
    ast.Module
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _load_tree(...)
    """
    text = path.read_text(encoding="utf-8")
    return ast.parse(text, filename=str(path))


def _definition_lines(tree: ast.Module) -> dict[str, int]:
    """Definition lines.

    Parameters
    ----------
    tree : ast.Module
        Description.

    Returns
    -------
    dict[str, int]
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _definition_lines(...)
    """
    lines: dict[str, int] = {}
    for node in tree.body:
        match node:
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                lines[node.name] = node.lineno
            case ast.ClassDef():
                lines[node.name] = node.lineno
            case ast.Assign(targets=targets):
                for target in targets:
                    if isinstance(target, ast.Name):
                        lines[target.id] = node.lineno
            case ast.AnnAssign(target=target):
                if isinstance(target, ast.Name):
                    lines[target.id] = node.lineno
    return lines


def _docstring_end(tree: ast.Module) -> int | None:
    """Docstring end.

    Parameters
    ----------
    tree : ast.Module
        Description.

    Returns
    -------
    int | None
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _docstring_end(...)
    """
    if not tree.body:
        return None
    node = tree.body[0]
    if (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ):
        return getattr(node, "end_lineno", node.lineno)
    return None


def _all_assignment_end(tree: ast.Module) -> int | None:
    """All assignment end.

    Parameters
    ----------
    tree : ast.Module
        Description.

    Returns
    -------
    int | None
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _all_assignment_end(...)
    """
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return getattr(node, "end_lineno", node.lineno)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        ):
            return getattr(node, "end_lineno", node.lineno)
    return None


def _navmap_assignment_span(tree: ast.Module) -> tuple[int, int] | None:
    """Navmap assignment span.

    Parameters
    ----------
    tree : ast.Module
        Description.

    Returns
    -------
    tuple[int, int] | None
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _navmap_assignment_span(...)
    """
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets: Iterable[ast.expr]
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "__navmap__":
                    start = node.lineno
                    end = getattr(node, "end_lineno", start)
                    return start, end
    return None


def _serialize_navmap(navmap: dict[str, Any]) -> list[str]:
    """Serialize navmap.

    Parameters
    ----------
    navmap : dict[str, Any]
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
    >>> _serialize_navmap(...)
    """
    literal = "__navmap__ = " + pformat(navmap, width=88, sort_dicts=True)
    return literal.splitlines()


def _ensure_navmap_structure(info: ModuleInfo) -> dict[str, Any]:
    """Ensure navmap structure.

    Parameters
    ----------
    info : ModuleInfo
        Description.

    Returns
    -------
    dict[str, Any]
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _ensure_navmap_structure(...)
    """
    raw_navmap = info.navmap_dict if info.navmap_dict else {}
    navmap: dict[str, Any] = dict(raw_navmap)
    exports = _normalize_exports(navmap.get("exports"), info.exports)
    navmap["exports"] = exports

    section_dicts = _collect_section_dicts(navmap.get("sections"))
    navmap["sections"] = _build_sections(section_dicts, exports)

    module_meta = _normalized_module_meta(navmap)
    if module_meta:
        navmap["module_meta"] = module_meta

    symbols_meta = _normalized_symbols(navmap.get("symbols"))
    _apply_symbol_defaults(symbols_meta, exports, module_meta)
    navmap["symbols"] = symbols_meta

    return navmap


def repair_module(info: ModuleInfo, apply: bool = False) -> list[str]:
    """Compute repair module.

    Carry out the repair module operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    info : ModuleInfo
        Description for ``info``.
    apply : bool | None
        Optional parameter default ``False``. Description for ``apply``.

    Returns
    -------
    List[str]
        Description of return value.

    Examples
    --------
    >>> from tools.navmap.repair_navmaps import repair_module
    >>> result = repair_module(...)
    >>> result  # doctest: +ELLIPSIS
    """
    path = info.path
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = _load_tree(path)

    current_navmap = info.navmap_dict or {}
    exports = _normalize_exports(current_navmap.get("exports"), info.exports)
    definition_lines = _definition_lines(tree)
    messages: list[str] = []

    insertions, anchor_messages = _collect_anchor_insertions(info, exports, definition_lines)
    messages.extend(anchor_messages)

    section_insertion = _public_api_insertion(info, tree)
    if section_insertion is not None:
        insertions.append(section_insertion)
        messages.append(
            f"{path}: inserted [nav:section public-api] after line {section_insertion[0]}"
        )

    changed = _apply_insertions(lines, insertions)

    nav_changed, nav_messages = _sync_navmap_literal(info, tree, text, lines, exports)
    changed = changed or nav_changed
    messages.extend(nav_messages)

    if changed and apply:
        new_text = "\n".join(lines)
        if not new_text.endswith("\n"):
            new_text += "\n"
        path.write_text(new_text, encoding="utf-8")

    return messages


def repair_all(root: Path, apply: bool) -> list[str]:
    """Compute repair all.

    Carry out the repair all operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    root : Path
        Description for ``root``.
    apply : bool
        Description for ``apply``.

    Returns
    -------
    List[str]
        Description of return value.

    Examples
    --------
    >>> from tools.navmap.repair_navmaps import repair_all
    >>> result = repair_all(..., ...)
    >>> result  # doctest: +ELLIPSIS
    """
    messages: list[str] = []
    for info in _collect_modules(root):
        messages.extend(repair_module(info, apply=apply))
    return messages


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse args.

    Parameters
    ----------
    argv : list[str] | None
        Description.

    Returns
    -------
    argparse.Namespace
        Description.


    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _parse_args(...)
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=SRC,
        help="Directory tree to scan for navmap metadata (default: %(default)s).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write fixes back to disk instead of printing suggested changes.",
    )
    return parser.parse_args(argv)


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
    >>> from tools.navmap.repair_navmaps import main
    >>> result = main()
    >>> result  # doctest: +ELLIPSIS
    """
    args = _parse_args(argv)
    root = args.root.resolve()
    messages = repair_all(root, apply=args.apply)
    if not messages:
        print("navmap repair: no issues detected")
        return 0
    print("\n".join(messages))
    if not args.apply:
        print("\nRe-run with --apply to write these fixes.")
    else:
        print("\nnavmap repair: applied fixes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _collect_section_dicts(raw: object) -> list[dict[str, Any]]:
    """Return section dictionaries extracted from ``raw`` when possible."""
    if not isinstance(raw, list):
        return []
    sections: list[dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, dict):
            sections.append(entry)
    return sections


def _build_sections(sections: Iterable[dict[str, Any]], exports: list[str]) -> list[dict[str, Any]]:
    """Return the canonical ``sections`` payload with the public API section first."""
    remaining = [section for section in sections if section.get("id") != "public-api"]
    return [{"id": "public-api", "symbols": exports}, *remaining]


def _collect_top_level_meta(navmap: Mapping[str, Any]) -> dict[str, Any]:
    """Return module metadata declared at the root of ``navmap``."""
    meta: dict[str, Any] = {}
    for key in ("owner", "stability", "since", "deprecated_in"):
        value = navmap.get(key)
        if value is not None:
            meta[key] = value
    return meta


def _normalized_module_meta(navmap: dict[str, Any]) -> dict[str, Any]:
    """Return module metadata after merging root-level defaults."""
    module_meta = _coerce_dict(navmap.get("module_meta"))
    top_level = _collect_top_level_meta(navmap)
    module_meta.update(top_level)
    for key in top_level:
        navmap.pop(key, None)
    return module_meta


def _normalized_symbols(raw: object) -> dict[str, dict[str, Any]]:
    """Return symbol metadata dictionaries keyed by symbol name."""
    if not isinstance(raw, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for name, meta in raw.items():
        if isinstance(name, str) and isinstance(meta, dict):
            result[name] = dict(meta)
    return result


def _apply_symbol_defaults(
    symbols_meta: dict[str, dict[str, Any]],
    exports: Iterable[str],
    module_meta: Mapping[str, Any],
) -> None:
    """Ensure every exported symbol inherits module-level defaults."""
    owner_default = module_meta.get("owner", "@todo-owner")
    stability_default = module_meta.get("stability", "experimental")
    since_default = module_meta.get("since", "0.0.0")
    deprecated_default = module_meta.get("deprecated_in")

    for name in exports:
        fields = symbols_meta.setdefault(name, {})
        fields.setdefault("owner", owner_default)
        fields.setdefault("stability", stability_default)
        fields.setdefault("since", since_default)
        if deprecated_default is not None:
            fields.setdefault("deprecated_in", deprecated_default)


def _normalize_exports(value: object, fallback: Iterable[str]) -> list[str]:
    """Return a deduplicated list of exports derived from ``value`` or ``fallback``."""
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        candidates = value
    else:
        candidates = fallback

    exports: list[str] = [item for item in candidates if isinstance(item, str)]
    # ``dict.fromkeys`` preserves order while deduplicating.
    return list(dict.fromkeys(exports))


def _coerce_dict(value: object) -> dict[str, Any]:
    """Return ``value`` as a shallow ``dict[str, Any]`` when possible."""
    if isinstance(value, dict):
        return dict(value)
    return {}


def _collect_anchor_insertions(
    info: ModuleInfo,
    exports: Iterable[str],
    definition_lines: Mapping[str, int],
) -> tuple[list[tuple[int, str]], list[str]]:
    """Return anchor insertion edits and messages for missing exports."""
    anchors = set(info.anchors)
    insertions: list[tuple[int, str]] = []
    messages: list[str] = []
    for name in exports:
        if name in anchors:
            continue
        line_no = definition_lines.get(name)
        if not line_no:
            messages.append(f"{info.path}: unable to locate definition for '{name}' to add anchor")
            continue
        insertions.append((line_no - 1, f"# [nav:anchor {name}]"))
        messages.append(f"{info.path}: inserted [nav:anchor {name}] at line {line_no}")
    return insertions, messages


def _public_api_insertion(info: ModuleInfo, tree: ast.Module) -> tuple[int, str] | None:
    """Return an insertion that ensures the public API section exists."""
    if "public-api" in set(info.sections):
        return None
    doc_end = _docstring_end(tree) or 0
    return doc_end, "# [nav:section public-api]"


def _apply_insertions(lines: list[str], insertions: list[tuple[int, str]]) -> bool:
    """Apply ``insertions`` to ``lines`` preserving relative order."""
    if not insertions:
        return False
    insertions.sort(key=lambda item: item[0])
    for offset, (index, content) in enumerate(insertions):
        lines.insert(index + offset, content)
    return True


def _sync_navmap_literal(
    info: ModuleInfo,
    tree: ast.Module,
    original_text: str,
    lines: list[str],
    exports: Sequence[str],
) -> tuple[bool, list[str]]:
    """Update the inline ``__navmap__`` literal when necessary."""
    messages: list[str] = []
    if not exports:
        return False, messages

    navmap_span = _navmap_assignment_span(tree)
    navmap_exists = navmap_span is not None and "__navmap__" in original_text
    updated_navmap = _ensure_navmap_structure(info)
    if navmap_exists and navmap_span is not None:
        start, end = navmap_span  # type: ignore[misc]
        navmap_lines = _serialize_navmap(updated_navmap)
        start_idx = start - 1
        end_idx = end
        if lines[start_idx:end_idx] != navmap_lines:
            lines[start_idx:end_idx] = navmap_lines
            messages.append(f"{info.path}: normalized __navmap__ literal")
            return True, messages
        return False, messages

    all_end = _all_assignment_end(tree) or 0
    navmap_lines = [*_serialize_navmap(updated_navmap), ""]
    lines[all_end:all_end] = navmap_lines
    messages.append(f"{info.path}: created __navmap__ stub with defaults")
    return True, messages
