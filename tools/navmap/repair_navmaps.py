#!/usr/bin/env python
"""Report navmap validation issues and offer guidance for repairs."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.navmap.check_navmap import _inspect

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"


def repair_module(path: Path) -> list[str]:
    """Return navmap validation issues for ``path``."""
    return _inspect(path)


def repair_all(root: Path) -> list[str]:
    """Run navmap validation across ``root`` and collect all issues."""
    errors: list[str] = []
    for py in root.rglob("*.py"):
        errors.extend(repair_module(py))
    return errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the navmap repair utility."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=SRC,
        help="Directory tree to scan for navmap metadata (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point that surfaces actionable repair guidance."""
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
