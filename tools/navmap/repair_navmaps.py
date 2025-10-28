#!/usr/bin/env python
"""Repair Navmaps utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.navmap.check_navmap import _inspect

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"


def repair_module(path: Path) -> list[str]:
    """Compute repair module.

    Carry out the repair module operation.

    Parameters
    ----------
    path : Path
        Description for ``path``.

    Returns
    -------
    List[str]
        Description of return value.
    """
    
    
    
    
    
    
    
    
    
    return _inspect(path)


def repair_all(root: Path) -> list[str]:
    """Compute repair all.

    Carry out the repair all operation.

    Parameters
    ----------
    root : Path
        Description for ``root``.

    Returns
    -------
    List[str]
        Description of return value.
    """
    
    
    
    
    
    
    
    
    
    errors: list[str] = []
    for py in root.rglob("*.py"):
        errors.extend(repair_module(py))
    return errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Compute parse args.

    Carry out the parse args operation.

    Parameters
    ----------
    argv : List[str] | None
        Description for ``argv``.

    Returns
    -------
    argparse.Namespace
        Description of return value.
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
    """Compute main.

    Carry out the main operation.

    Parameters
    ----------
    argv : List[str] | None
        Description for ``argv``.

    Returns
    -------
    int
        Description of return value.
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
