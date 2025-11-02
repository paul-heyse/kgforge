"""Stub for astroid optional dependency (AST analysis)."""

from __future__ import annotations

class AstroidManager:
    """Manager for Astroid AST analysis."""

    def build_from_file(self, path: str) -> object:
        """Build an AST from a Python source file."""
        ...

MANAGER: AstroidManager
