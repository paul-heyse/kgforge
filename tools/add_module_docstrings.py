"""Provide utilities for module.

Auto-generated API documentation for the ``tools.add_module_docstrings`` module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
tools.add_module_docstrings
"""


from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def module_name(path: Path) -> str:
    """Return module name.

    Auto-generated reference for the ``module_name`` callable defined in ``tools.add_module_docstrings``.
    
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
    
    See Also
    --------
    tools.add_module_docstrings
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    rel = path.relative_to(SRC).with_suffix("")
    return str(rel).replace("/", ".")


def needs_docstring(text: str) -> bool:
    """Return needs docstring.

    Auto-generated reference for the ``needs_docstring`` callable defined in ``tools.add_module_docstrings``.
    
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
    
    See Also
    --------
    tools.add_module_docstrings
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return ast.get_docstring(tree, clean=False) is None


def insert_docstring(path: Path) -> bool:
    """Return insert docstring.

    Auto-generated reference for the ``insert_docstring`` callable defined in ``tools.add_module_docstrings``.
    
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
    
    See Also
    --------
    tools.add_module_docstrings
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
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
    """Return main.

    Auto-generated reference for the ``main`` callable defined in ``tools.add_module_docstrings``.
    
    Examples
    --------
    >>> from tools.add_module_docstrings import main
    >>> main()  # doctest: +ELLIPSIS
    
    See Also
    --------
    tools.add_module_docstrings
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    for path in SRC.rglob("*.py"):
        insert_docstring(path)


if __name__ == "__main__":
    main()
