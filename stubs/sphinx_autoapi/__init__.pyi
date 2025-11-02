"""Stub for sphinx_autoapi optional dependency."""

from __future__ import annotations

from autoapi._parser import Parser as Parser

__all__ = ["Parser", "parse"]

def parse() -> None:
    """Parse and populate AutoAPI data."""
    ...
