"""Utilities to inject default module docstrings into the source tree."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def module_name(path: Path) -> str:
    """Return the dotted module path for a given source file."""
    rel = path.relative_to(SRC).with_suffix("")
    return str(rel).replace("/", ".")


def needs_docstring(text: str) -> bool:
    """Return True when the module lacks a top-level docstring."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return ast.get_docstring(tree, clean=False) is None


def insert_docstring(path: Path) -> bool:
    """Insert a module docstring if the file does not already have one."""
    text = path.read_text()
    if not needs_docstring(text):
        return False
    name = module_name(path)
    doc = f'"""Module for {name}."""\n\n'
    lines = text.splitlines(keepends=True)
    idx = 0
    if lines and lines[0].startswith("#!"):
        idx = 1
    if len(lines) > idx and lines[idx].startswith("#") and "coding" in lines[idx]:
        idx += 1
    lines.insert(idx, doc)
    path.write_text("".join(lines))
    return True


def main() -> None:
    """Entry point when invoked as a script."""
    for path in SRC.rglob("*.py"):
        insert_docstring(path)


if __name__ == "__main__":
    main()
