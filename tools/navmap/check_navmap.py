#!/usr/bin/env python
"""Validate module navmap definitions and anchors."""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

try:
    from tools.navmap.build_navmap import ModuleInfo
except ModuleNotFoundError:  # pragma: no cover - fallback for direct script execution
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tools.navmap.build_navmap import ModuleInfo


SRC = REPO / "src"
ANCHOR_RE = re.compile(r"^\s*#\s*\[nav:anchor\s+([A-Za-z_]\w*)\]")
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SYMBOL_RE = re.compile(r"^[A-Za-z_]\w*$")


def _extract_string(node: ast.AST) -> str | None:
    """Return a string literal value when present."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_strings(node: ast.AST) -> list[str]:
    """Collect string literals from list or tuple AST nodes."""
    match node:
        case ast.List(elts=elts) | ast.Tuple(elts=elts):
            return [s for elt in elts if (s := _extract_string(elt)) is not None]
        case _:
            return []


def _literal_eval(node: ast.AST, names: Mapping[str, object]) -> object:
    """Evaluate a limited subset of literals used in navmap definitions."""
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
            message = f"Unsupported literal in navmap: {ast.dump(node)}"
            raise ValueError(message)


def _inspect(py: Path) -> ModuleInfo | None:  # noqa: C901, PLR0912
    """Inspect ``py`` and return a list of validation errors."""
    text = py.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(py))

    exports: set[str] = set()
    navmap: dict[str, Any] | None = None

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    exports = set(_extract_strings(node.value))
                if isinstance(target, ast.Name) and target.id == "__navmap__":
                    names = {"__all__": list(exports)}
                    try:
                        navmap = _literal_eval(node.value, names)
                    except Exception:
                        navmap = None
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__navmap__"
            and node.value is not None
        ):
            names = {"__all__": list(exports)}
            try:
                navmap = _literal_eval(node.value, names)
            except Exception:
                navmap = None

    anchors = {
        match.group(1) for match in (ANCHOR_RE.match(line) for line in text.splitlines()) if match
    }

    if navmap is None:
        return []

    errors: list[str] = []
    exports_list = list(exports)
    nav_exports = navmap.get("exports", exports_list)

    if set(nav_exports) != exports:
        errors.append(
            f"{py}: exports mismatch between __all__ and __navmap__ ({sorted(exports)} != {sorted(nav_exports)})"
        )

    sections = navmap.get("sections", [])
    if sections:
        first = sections[0].get("id")
        if first != "public-api":
            errors.append(f"{py}: first navmap section must have id 'public-api'")
        for section in sections:
            sid = section.get("id", "")
            if sid and not SLUG_RE.match(sid):
                errors.append(f"{py}: section id '{sid}' is not kebab-case")
            for sym in section.get("symbols", []):
                if not SYMBOL_RE.match(sym):
                    errors.append(f"{py}: invalid symbol name '{sym}' in section '{sid}'")
                elif sym not in anchors:
                    errors.append(f"{py}: missing [nav:anchor] for section symbol '{sym}'")

    return errors


def main() -> int:
    """Entry point used by pre-commit to validate navmap metadata."""
    errors: list[str] = []
    for py in SRC.rglob("*.py"):
        errors.extend(_inspect(py))

    if errors:
        print("\n".join(errors))
        return 1

    print("navmap check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
