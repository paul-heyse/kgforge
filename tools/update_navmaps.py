#!/usr/bin/env python
"""Provide utilities for module.

Auto-generated API documentation for the ``tools.update_navmaps`` module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
tools.update_navmaps
"""


from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def iter_python_files() -> list[Path]:
    """Return iter python files.

    Auto-generated reference for the ``iter_python_files`` callable defined in ``tools.update_navmaps``.
    
    Returns
    -------
    List[Path]
        Description of return value.
    
    Examples
    --------
    >>> from tools.update_navmaps import iter_python_files
    >>> result = iter_python_files()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.update_navmaps
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    search_root = SRC if SRC.exists() else ROOT
    return sorted(path for path in search_root.rglob("*.py") if path.is_file())


def module_docstring(path: Path) -> str | None:
    """Return module docstring.

    Auto-generated reference for the ``module_docstring`` callable defined in ``tools.update_navmaps``.
    
    Parameters
    ----------
    path : Path
        Description for ``path``.
    
    Returns
    -------
    str | None
        Description of return value.
    
    Examples
    --------
    >>> from tools.update_navmaps import module_docstring
    >>> result = module_docstring(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.update_navmaps
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return None
    return ast.get_docstring(tree)


def main() -> None:
    """Return main.

    Auto-generated reference for the ``main`` callable defined in ``tools.update_navmaps``.
    
    Raises
    ------
    SystemExit
        Raised when validation fails.
    
    Examples
    --------
    >>> from tools.update_navmaps import main
    >>> main()  # doctest: +ELLIPSIS
    
    See Also
    --------
    tools.update_navmaps
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    offenders: list[Path] = []
    for file_path in iter_python_files():
        doc = module_docstring(file_path)
        if doc and "NavMap:" in doc:
            offenders.append(file_path)
    if offenders:
        joined = "\n".join(str(path) for path in offenders)
        message = (
            "Module docstrings must not contain 'NavMap:' sections. "
            f"Found violations in:\n{joined}"
        )
        raise SystemExit(message)


if __name__ == "__main__":
    main()
