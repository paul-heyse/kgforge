"""Compatibility wrapper that invokes the docstring builder CLI."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCFACTS = REPO / "docs" / "_build" / "docfacts.json"

LOGGER = logging.getLogger("docstring_builder.generate_docstrings")


def run_builder(extra_args: list[str] | None = None) -> None:
    """Execute the docstring builder CLI with optional arguments."""
    args = extra_args or []
    cmd = [sys.executable, "-m", "tools.docstring_builder.cli", "update", *args]
    LOGGER.info("[docstrings] Running docstring builder: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    """Entry point used by ``make docstrings`` and pre-commit hooks."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    DOCFACTS.parent.mkdir(parents=True, exist_ok=True)
    run_builder()


if __name__ == "__main__":
    main()
