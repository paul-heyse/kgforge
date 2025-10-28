"""Add Module Docstrings utilities."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def module_name(path: Path) -> str:
    """Compute module name.

    Carry out the module name operation.

    Parameters
    ----------
    path : Path
        Description for ``path``.

    Returns
    -------
    str
        Description of return value.

    Examples
    --------
    >>> from tools.add_module_docstrings import module_name
    >>> result = module_name(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def needs_docstring(text: str) -> bool:
    """Compute needs docstring.

    Carry out the needs docstring operation.

    Parameters
    ----------
    text : str
        Description for ``text``.

    Returns
    -------
    bool
        Description of return value.

    Examples
    --------
    >>> from tools.add_module_docstrings import needs_docstring
    >>> result = needs_docstring(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return ast.get_docstring(tree, clean=False) is None


def insert_docstring(path: Path) -> bool:
    """Compute insert docstring.

    Carry out the insert docstring operation.

    Parameters
    ----------
    path : Path
        Description for ``path``.

    Returns
    -------
    bool
        Description of return value.

    Examples
    --------
    >>> from tools.add_module_docstrings import insert_docstring
    >>> result = insert_docstring(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
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
    """Compute main.

    Carry out the main operation.

    Examples
    --------
    >>> from tools.add_module_docstrings import main
    >>> main()  # doctest: +ELLIPSIS
    """
    for path in SRC.rglob("*.py"):
        insert_docstring(path)


if __name__ == "__main__":
    main()
