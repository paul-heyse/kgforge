"""Insert placeholder module docstrings for files that are missing one."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def module_name(path: Path) -> str:
    """Return the dotted import path for ``path`` relative to ``src/``.

    Parameters
    ----------
    path : Path
        Python source file whose module name should be derived.

    Returns
    -------
    str
        Import path that represents ``path`` without the ``.py`` suffix.
    """

    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def needs_docstring(text: str) -> bool:
    """Return ``True`` when ``text`` does not define a module docstring."""

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return ast.get_docstring(tree, clean=False) is None


def insert_docstring(path: Path) -> bool:
    """Insert a placeholder module docstring if ``path`` does not have one.

    Parameters
    ----------
    path : Path
        Module that should be updated.

    Returns
    -------
    bool
        ``True`` when the file was modified, ``False`` otherwise.
    """

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
    """Ensure every module in ``src/`` has at least a placeholder docstring."""

    for path in SRC.rglob("*.py"):
        insert_docstring(path)


if __name__ == "__main__":
    main()
