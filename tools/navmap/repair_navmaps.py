#!/usr/bin/env python
"""Repair Navmaps utilities."""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable
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
    navmap = dict(info.navmap)
    exports = list(dict.fromkeys(navmap.get("exports", info.exports)))
    navmap["exports"] = exports

    sections = navmap.get("sections", [])
    remaining = [section for section in sections if section.get("id") != "public-api"]
    navmap["sections"] = [{"id": "public-api", "symbols": exports}] + remaining

    module_meta = {
        key: navmap.get(key)
        for key in ("owner", "stability", "since", "deprecated_in")
        if navmap.get(key) is not None
    }
    if module_meta:
        navmap["module_meta"] = module_meta

    symbols_meta = dict(navmap.get("symbols", {}))
    for name in exports:
        fields = symbols_meta.setdefault(name, {})
        fields.setdefault("owner", module_meta.get("owner", "@todo-owner"))
        fields.setdefault("stability", module_meta.get("stability", "experimental"))
        fields.setdefault("since", module_meta.get("since", "0.0.0"))
        if "deprecated_in" not in fields and module_meta.get("deprecated_in") is not None:
            fields["deprecated_in"] = module_meta["deprecated_in"]
    navmap["symbols"] = symbols_meta

    return navmap


def repair_module(info: ModuleInfo, apply: bool = False) -> list[str]:
    """Compute repair module.

    Carry out the repair module operation.

    Parameters
    ----------
    info : ModuleInfo
        Description for ``info``.
    apply : bool | None
        Description for ``apply``.

    Returns
    -------
    List[str]
        Description of return value.
    """
    
    
    
    
    
    
    
    path = info.path
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = _load_tree(path)

    exports = list(dict.fromkeys(info.navmap.get("exports", info.exports)))
    anchors = set(info.anchors)
    definition_lines = _definition_lines(tree)
    messages: list[str] = []
    insertions: list[tuple[int, str]] = []
    changed = False

    for name in exports:
        if name not in anchors:
            line_no = definition_lines.get(name)
            if not line_no:
                messages.append(f"{path}: unable to locate definition for '{name}' to add anchor")
                continue
            insertions.append((line_no - 1, f"# [nav:anchor {name}]"))
            messages.append(f"{path}: inserted [nav:anchor {name}] at line {line_no}")

    section_ids = {slug for slug in info.sections}
    if "public-api" not in section_ids:
        doc_end = _docstring_end(tree) or 0
        insertion_line = doc_end
        insertions.append((insertion_line, "# [nav:section public-api]"))
        messages.append(f"{path}: inserted [nav:section public-api] after line {insertion_line}")

    if insertions:
        insertions.sort(key=lambda item: item[0])
        offset = 0
        for index, content in insertions:
            lines.insert(index + offset, content)
            offset += 1
        changed = True

    updated_navmap: dict[str, Any] | None = None
    navmap_span = _navmap_assignment_span(tree)
    navmap_exists = navmap_span is not None and "__navmap__" in text

    if exports:
        if navmap_exists:
            start, end = navmap_span  # type: ignore[misc]
            updated_navmap = _ensure_navmap_structure(info)
            navmap_lines = _serialize_navmap(updated_navmap)
            start_idx = start - 1
            end_idx = end
            if lines[start_idx:end_idx] != navmap_lines:
                lines[start_idx:end_idx] = navmap_lines
                messages.append(f"{path}: normalized __navmap__ literal")
                changed = True
        else:
            all_end = _all_assignment_end(tree) or 0
            updated_navmap = _ensure_navmap_structure(info)
            navmap_lines = _serialize_navmap(updated_navmap) + [""]
            lines[all_end:all_end] = navmap_lines
            messages.append(f"{path}: created __navmap__ stub with defaults")
            changed = True

    if changed and apply:
        new_text = "\n".join(lines)
        if not new_text.endswith("\n"):
            new_text += "\n"
        path.write_text(new_text, encoding="utf-8")

    return messages


def repair_all(root: Path, apply: bool) -> list[str]:
    """Compute repair all.

    Carry out the repair all operation.

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

    Carry out the main operation.

    Parameters
    ----------
    argv : List[str] | None
        Description for ``argv``.

    Returns
    -------
    int
        Description of return value.
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
