#!/usr/bin/env python
"""Provide utilities for module.

Auto-generated API documentation for the ``tools.navmap.repair_navmaps`` module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
tools.navmap.repair_navmaps
"""


from __future__ import annotations

import argparse
from pathlib import Path

from tools.navmap.check_navmap import _inspect

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"


def repair_module(path: Path) -> list[str]:
    """Return repair module.

    Auto-generated reference for the ``repair_module`` callable defined in ``tools.navmap.repair_navmaps``.
    
    Parameters
    ----------
    path : Path
        Description for ``path``.
    
    Returns
    -------
    List[str]
        Description of return value.
    
    Examples
    --------
    >>> from tools.navmap.repair_navmaps import repair_module
    >>> result = repair_module(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.navmap.repair_navmaps
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    return _inspect(path)


def repair_all(root: Path) -> list[str]:
    """Return repair all.

    Auto-generated reference for the ``repair_all`` callable defined in ``tools.navmap.repair_navmaps``.
    
    Parameters
    ----------
    root : Path
        Description for ``root``.
    
    Returns
    -------
    List[str]
        Description of return value.
    
    Examples
    --------
    >>> from tools.navmap.repair_navmaps import repair_all
    >>> result = repair_all(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.navmap.repair_navmaps
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    errors: list[str] = []
    for py in root.rglob("*.py"):
        errors.extend(repair_module(py))
    return errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Return parse args.

    Auto-generated reference for the ``_parse_args`` callable defined in ``tools.navmap.repair_navmaps``.
    
    Parameters
    ----------
    argv : List[str], optional
        Description for ``argv``.
    
    Returns
    -------
    argparse.Namespace
        Description of return value.
    
    Examples
    --------
    >>> from tools.navmap.repair_navmaps import _parse_args
    >>> result = _parse_args(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.navmap.repair_navmaps
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=SRC,
        help="Directory tree to scan for navmap metadata (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Return main.

    Auto-generated reference for the ``main`` callable defined in ``tools.navmap.repair_navmaps``.
    
    Parameters
    ----------
    argv : List[str], optional
        Description for ``argv``.
    
    Returns
    -------
    int
        Description of return value.
    
    Examples
    --------
    >>> from tools.navmap.repair_navmaps import main
    >>> result = main(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.navmap.repair_navmaps
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    args = _parse_args(argv)
    root = args.root.resolve()
    errors = repair_all(root)
    if not errors:
        print("navmap repair: no issues detected")
        return 0
    print("navmap repair: please address the issues below\n")
    print("\n".join(errors))
    print("\nHint: add [nav:anchor SymbolName] comments next to public definitions.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
