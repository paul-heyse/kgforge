"""Docstring builder package coordinating harvesting, synthesis, and application."""

from __future__ import annotations

from importlib import import_module

BUILDER_VERSION = "2.0.0"

__all__ = ["BUILDER_VERSION", "main"]


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the CLI entry point without eagerly importing it."""
    cli_module = import_module("tools.docstring_builder.cli")
    cli_main = cli_module.main
    return cli_main(argv)
