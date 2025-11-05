"""Compatibility wrapper that invokes the docstring builder CLI."""

from __future__ import annotations

import sys
from pathlib import Path

from tools._shared.error_codes import format_error_message
from tools._shared.logging import get_logger
from tools._shared.proc import ToolExecutionError, run_tool

LOGGER = get_logger(__name__)


REPO = Path(__file__).resolve().parents[1]
DOCFACTS = REPO / "docs" / "_build" / "docfacts.json"


def run_builder(extra_args: list[str] | None = None) -> None:
    """Execute the docstring builder CLI with optional arguments."""
    args = extra_args or []
    cmd = [sys.executable, "-m", "tools.docstring_builder.cli", "update", *args]
    LOGGER.info("[docstrings] Running docstring builder: %s", " ".join(cmd))
    try:
        run_tool(cmd, timeout=60.0, check=True)
    except ToolExecutionError as exc:
        stderr = getattr(exc, "stderr", "") or ""
        stdout = getattr(exc, "stdout", "") or ""
        combined_details = (
            "\n".join(part for part in (stderr.strip(), stdout.strip()) if part) or None
        )
        error_code = "KGF-DOC-BLD-001"
        message = "Docstring builder failed"
        if combined_details:
            lowered = combined_details.lower()
            if "schema validation failed" in lowered or "schema_docfacts" in lowered:
                error_code = "KGF-DOC-BLD-006"
                message = "DocFacts schema validation failed during docstring build"
            elif "missing canonical schema" in lowered:
                error_code = "KGF-DOC-ENV-002"
                message = "DocFacts schema not found"
        formatted = format_error_message(error_code, message, details=combined_details)
        LOGGER.exception(
            formatted,
            extra={
                "error_code": error_code,
                "command": cmd,
                "returncode": exc.returncode,
            },
        )
        raise SystemExit(exc.returncode if exc.returncode is not None else 1) from exc


def main() -> None:
    """Entry point used by ``make docstrings`` and pre-commit hooks."""
    DOCFACTS.parent.mkdir(parents=True, exist_ok=True)
    run_builder()


if __name__ == "__main__":
    main()
