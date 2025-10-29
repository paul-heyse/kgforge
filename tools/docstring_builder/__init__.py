"""Docstring builder package coordinating harvesting, synthesis, and
application.
"""

from __future__ import annotations

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the CLI entry point without eagerly importing it."""
    from tools.docstring_builder.cli import main as _main

    return _main(argv)
