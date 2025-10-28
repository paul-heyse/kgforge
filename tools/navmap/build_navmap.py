#!/usr/bin/env python
"""Build Navmap utilities."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
OUT = REPO / "site" / "_build" / "navmap"
ANCHOR_RE = re.compile(r"^\s*#\s*\[nav:anchor\s+([A-Za-z_]\w*)\]")
SECTION_RE = re.compile(r"^\s*#\s*\[nav:section\s+([a-z0-9]+(?:-[a-z0-9]+)*)\]")


@dataclass(slots=True)
class ModuleInfo:
    """Describe ModuleInfo."""

    path: Path
    module: str
    exports: list[str]
    navmap: dict[str, Any]
    anchors: dict[str, int]
    sections: dict[str, int]


def _extract_string(node: ast.AST) -> str | None:
    """Return extract string.

    Parameters
    ----------
    node : ast.AST
        Description for ``node``.

    Returns
    -------
    str | None
        Description of return value.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_strings(node: ast.AST) -> list[str]:
    """Return extract strings.

    Parameters
    ----------
    node : ast.AST
        Description for ``node``.

    Returns
    -------
    List[str]
        Description of return value.
    """
    match node:
        case ast.List(elts=elts) | ast.Tuple(elts=elts):
            return [s for elt in elts if (s := _extract_string(elt)) is not None]
        case _:
            return []


def _literal_eval(node: ast.AST, names: Mapping[str, object]) -> object:
    """Return literal eval.

    Parameters
    ----------
    node : ast.AST
        Description for ``node``.
    names : Mapping[str, object]
        Description for ``names``.

    Returns
    -------
    object
        Description of return value.

    Raises
    ------
    ValueError
        Raised when validation fails.
    """
    match node:
        case ast.Constant(value=value):
            return value
        case ast.List(elts=elts):
            return [_literal_eval(elt, names) for elt in elts]
        case ast.Tuple(elts=elts):
            return tuple(_literal_eval(elt, names) for elt in elts)
        case ast.Set(elts=elts):
            return {_literal_eval(elt, names) for elt in elts}
        case ast.Dict(keys=keys, values=values):
            return {
                _literal_eval(key, names): _literal_eval(value, names)
                for key, value in zip(keys, values, strict=True)
            }
        case ast.Name(id=name) if name in names:
            return names[name]
        case _:
            message = f"Unsupported expression in navmap literal: {ast.dump(node)}"
            raise ValueError(message)


def _module_name(py: Path) -> str | None:
    """Return module name.

    Parameters
    ----------
    py : Path
        Description for ``py``.

    Returns
    -------
    str | None
        Description of return value.
    """
    try:
        rel = py.relative_to(SRC)
    except ValueError:
        return None
    parts = rel.parent.parts if py.name == "__init__.py" else rel.with_suffix("").parts
    if not parts:
        return None
    return ".".join(parts)


def _rel(path: Path) -> str:
    """Return rel.

    Parameters
    ----------
    path : Path
        Description for ``path``.

    Returns
    -------
    str
        Description of return value.
    """
    return path.relative_to(REPO).as_posix()


def _git_sha() -> str:
    """Return git sha.

    Returns
    -------
    str
        Description of return value.
    """
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode("utf-8").strip()


def _parse_module(py: Path) -> ModuleInfo | None:  # noqa: C901, PLR0912
    """Return parse module.

    Parameters
    ----------
    py : Path
        Description for ``py``.

    Returns
    -------
    ModuleInfo | None
        Description of return value.
    """
    module = _module_name(py)
    if not module:
        return None

    text = py.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(py))

    exports: list[str] = []
    navmap: dict[str, Any] = {}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "__all__":
                    exports = _extract_strings(node.value)
                elif target.id == "__navmap__":
                    names = {"__all__": exports}
                    try:
                        navmap = _literal_eval(node.value, names)
                    except Exception:
                        navmap = {}
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__navmap__"
            and node.value is not None
        ):
            names = {"__all__": exports}
            try:
                navmap = _literal_eval(node.value, names)
            except Exception:
                navmap = {}

    anchors: dict[str, int] = {}
    sections: dict[str, int] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        if match := ANCHOR_RE.match(line):
            anchors[match.group(1)] = lineno
        if match := SECTION_RE.match(line):
            sections[match.group(1)] = lineno

    return ModuleInfo(
        path=py,
        module=module,
        exports=exports,
        navmap=navmap,
        anchors=anchors,
        sections=sections,
    )


def _collect_modules() -> list[ModuleInfo]:
    """Return collect modules.

    Returns
    -------
    List[ModuleInfo]
        Description of return value.
    """
    modules: list[ModuleInfo] = []
    for py in SRC.rglob("*.py"):
        info = _parse_module(py)
        if info:
            modules.append(info)
    return modules


def build_index() -> dict[str, Any]:
    """Return build index.

    Returns
    -------
    Mapping[str, Any]
        Description of return value.
    """
    sha = _git_sha()
    data: dict[str, Any] = {"commit": sha, "modules": {}}

    for info in _collect_modules():
        navmap = info.navmap or {}
        exports = navmap.get("exports", info.exports)
        data["modules"][info.module] = {
            "path": _rel(info.path),
            "exports": exports,
            "sections": navmap.get("sections", []),
            "section_lines": info.sections,
            "anchors": info.anchors,
            "links": {
                "source": f"vscode://file/{_rel(info.path)}",
            },
            "meta": navmap.get("symbols", {}),
            "tags": navmap.get("tags", []),
            "synopsis": navmap.get("synopsis", ""),
            "see_also": navmap.get("see_also", []),
            "deps": navmap.get("deps", []),
        }
    return data


def main() -> None:
    """Return main."""
    index = build_index()
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "navmap.json"
    out_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote {_rel(out_path)} @ {index['commit']}")


if __name__ == "__main__":
    main()
