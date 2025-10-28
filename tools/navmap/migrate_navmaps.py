#!/usr/bin/env python
"""Migrate Navmaps utilities."""

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
    """Compute migrate navmaps.

    Carry out the migrate navmaps operation.

    Parameters
    ----------
    output : Path | None
        Description for ``output``.
    pretty : bool | None
        Description for ``pretty``.

    Returns
    -------
    Mapping[str, Any]
        Description of return value.
    """
    index = build_index()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(index, indent=2 if pretty else None)
        output.write_text(text, encoding="utf-8")
    return index


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
    migrate_navmaps(args.output, pretty=not args.compact)
    print(f"Wrote navmap index to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
