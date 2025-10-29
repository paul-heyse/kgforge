"""Compatibility wrapper that invokes the docstring builder CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCFACTS = REPO / "docs" / "_build" / "docfacts.json"


def run_builder(extra_args: list[str] | None = None) -> None:
    """Execute the docstring builder CLI with optional arguments."""

    args = extra_args or []
    cmd = [sys.executable, "-m", "tools.docstring_builder.cli", "update", *args]
    print(f"[docstrings] Running docstring builder: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    """Entry point used by ``make docstrings`` and pre-commit hooks."""

    DOCFACTS.parent.mkdir(parents=True, exist_ok=True)
    run_builder()


if __name__ == "__main__":
    main()
