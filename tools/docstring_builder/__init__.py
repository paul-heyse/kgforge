"""Docstring builder package coordinating harvesting, synthesis, and application."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import cast

from tools.docstring_builder import cache as cache
from tools.docstring_builder import cli as cli
from tools.docstring_builder import config as config
from tools.docstring_builder import docfacts as docfacts
from tools.docstring_builder import harvest as harvest
from tools.docstring_builder import ir as ir
from tools.docstring_builder import models as models
from tools.docstring_builder import observability as observability
from tools.docstring_builder import policy as policy
from tools.docstring_builder import render as render
from tools.docstring_builder import schema as schema
from tools.docstring_builder import semantics as semantics

BUILDER_VERSION = "2.0.0"

__all__ = [
    "BUILDER_VERSION",
    "cache",
    "cli",
    "config",
    "docfacts",
    "harvest",
    "ir",
    "main",
    "models",
    "observability",
    "policy",
    "render",
    "schema",
    "semantics",
]


CliMain = Callable[[list[str] | None], int]


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the CLI entry point without eagerly importing it."""
    cli_module = import_module("tools.docstring_builder.cli")
    cli_main = cast(CliMain, cli_module.main)
    return cli_main(argv)
