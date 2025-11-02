"""Stub for sphinx_autoapi optional dependency."""

from __future__ import annotations

__all__ = ["Parser", "parse"]

class Parser:
    """Simplified parser interface exposed by sphinx-autoapi."""

    def __init__(self) -> None: ...
    def parse(self, node: object) -> object: ...
    def parse_file(self, file_path: str) -> object: ...
    def parse_file_in_namespace(self, file_path: str, dir_root: str) -> object: ...

def parse() -> None:
    """Parse and populate AutoAPI data."""
    ...
