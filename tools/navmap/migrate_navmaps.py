#!/usr/bin/env python
"""Provide utilities for module.

Auto-generated API documentation for the ``tools.navmap.migrate_navmaps`` module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
tools.navmap.migrate_navmaps
"""


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final

from tools.navmap.build_navmap import build_index

DEFAULT_OUTPUT: Final[Path] = (
    Path(__file__).resolve().parents[2] / "site" / "_build" / "navmap.json"
)


def migrate_navmaps(output: Path | None = None, pretty: bool = True) -> dict[str, Any]:
    """Return migrate navmaps.

    Auto-generated reference for the ``migrate_navmaps`` callable defined in ``tools.navmap.migrate_navmaps``.
    
    Parameters
    ----------
    output : Path, optional
        Description for ``output``.
    pretty : bool, optional
        Description for ``pretty``.
    
    Returns
    -------
    Mapping[str, Any]
        Description of return value.
    
    Examples
    --------
    >>> from tools.navmap.migrate_navmaps import migrate_navmaps
    >>> result = migrate_navmaps(..., ...)
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.navmap.migrate_navmaps
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    index = build_index()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(index, indent=2 if pretty else None)
        output.write_text(text, encoding="utf-8")
    return index


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Return parse args.

    Auto-generated reference for the ``_parse_args`` callable defined in ``tools.navmap.migrate_navmaps``.
    
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
    >>> from tools.navmap.migrate_navmaps import _parse_args
    >>> result = _parse_args(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.navmap.migrate_navmaps
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination for the generated navmap JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit JSON without indentation to save space.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Return main.

    Auto-generated reference for the ``main`` callable defined in ``tools.navmap.migrate_navmaps``.
    
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
    >>> from tools.navmap.migrate_navmaps import main
    >>> result = main(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    tools.navmap.migrate_navmaps
    
    Notes
    -----
    Provide usage considerations, constraints, or complexity notes.
    """
    
    args = _parse_args(argv)
    migrate_navmaps(args.output, pretty=not args.compact)
    print(f"Wrote navmap index to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
