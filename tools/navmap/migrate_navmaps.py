#!/usr/bin/env python
"""Regenerate the navigation map JSON consumed by the documentation site."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

from tools._shared.logging import get_logger
from tools.navmap.build_navmap import NavIndexDict, build_index

LOGGER = get_logger(__name__)

DEFAULT_OUTPUT: Final[Path] = (
    Path(__file__).resolve().parents[2] / "site" / "_build" / "navmap.json"
)


def migrate_navmaps(output: Path | None = None, pretty: bool = True) -> NavIndexDict:
    """Rebuild the navigation map JSON file from the current source tree.

    Parameters
    ----------
    output
        Destination path for the generated JSON document. When ``None`` the
        navmap is only returned to the caller.
    pretty
        When ``True`` the JSON is emitted with indentation suitable for code
        reviews; otherwise a compact representation is written.

    Returns
    -------
    NavIndexDict
        Structured navigation metadata emitted by
        :func:`tools.navmap.build_navmap.build_index`.
    """
    index = build_index()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(index, indent=2 if pretty else None)
        output.write_text(text, encoding="utf-8")
    return index


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the navmap migration utility."""
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
    """CLI entry point for regenerating the navmap JSON asset.

    Parameters
    ----------
    argv
        Optional argument vector, primarily used by tests.

    Returns
    -------
    int
        ``0`` on success so the helper integrates cleanly with shell pipelines.
    """
    args = _parse_args(argv)
    migrate_navmaps(args.output, pretty=not args.compact)
    LOGGER.info("Wrote navmap index to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
