"""Utilities for inserting placeholder module docstrings into the codebase.

The helpers in this module scan the ``src`` tree, derive the dotted module path for
each file, and inject a minimal docstring when one is missing. This is primarily
used when bootstrapping new packages so that quality gates requiring docstrings
pass before more detailed documentation is added.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def module_name(path: Path) -> str:
    """Return the dotted module path for ``path`` relative to ``src``.

    Parameters
    ----------
    path
        Absolute path to a Python file inside the repository ``src`` tree.

    Returns
    -------
    str
        Importable module name with ``__init__`` files collapsed to the package
        path.
    """
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def needs_docstring(text: str) -> bool:
    """Return ``True`` when ``text`` parses without a module docstring.

    The file is parsed with :mod:`ast` so we accurately detect docstrings even
    when comments or encoding pragmas are present above the module body.

    Parameters
    ----------
    text
        Source code of the module to inspect.

    Returns
    -------
    bool
        ``True`` if the module lacks a top-level docstring, ``False`` otherwise.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return ast.get_docstring(tree, clean=False) is None


def insert_docstring(path: Path) -> bool:
    """Insert a minimal module docstring into ``path`` when one is missing.

    The helper respects shebangs and encoding pragmas, inserting the generated
    docstring immediately afterwards so file metadata stays intact.

    Parameters
    ----------
    path
        Absolute path to the Python module to update.

    Returns
    -------
    bool
        ``True`` if the file was modified, ``False`` when no change was required.
    """
    # Explicitly use UTF-8 so behavior matches the project's encoding assumptions across platforms.
    text = path.read_text(encoding="utf-8")
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
    path.write_text("".join(lines), encoding="utf-8")
    return True


def main() -> None:
    """Walk the source tree and inject docstrings for any bare modules found."""
    for path in SRC.rglob("*.py"):
        insert_docstring(path)


if __name__ == "__main__":
    main()
