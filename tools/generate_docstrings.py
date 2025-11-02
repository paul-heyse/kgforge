"""Compatibility wrapper that invokes the docstring builder CLI."""

from __future__ import annotations

import sys
from pathlib import Path

from tools.shared.logging import get_logger
from tools.shared.proc import ToolExecutionError, run_tool

LOGGER = get_logger(__name__)


REPO = Path(__file__).resolve().parents[1]
DOCFACTS = REPO / "docs" / "_build" / "docfacts.json"


def run_builder(extra_args: list[str] | None = None) -> None:
    """Execute the docstring builder CLI with optional arguments."""
    args = extra_args or []
    cmd = [sys.executable, "-m", "tools.docstring_builder.cli", "update", *args]
    LOGGER.info("[docstrings] Running docstring builder: %s", " ".join(cmd))
    try:
        run_tool(cmd, timeout=20.0, check=True)
    except ToolExecutionError as exc:
        LOGGER.exception("Docstring builder failed")
        raise SystemExit(exc.returncode if exc.returncode is not None else 1) from exc


def main() -> None:
    """Entry point used by ``make docstrings`` and pre-commit hooks."""
    DOCFACTS.parent.mkdir(parents=True, exist_ok=True)
    run_builder()


if __name__ == "__main__":
    main()
