from __future__ import annotations

class AstroidManager:
    """Minimal Astroid manager surface required for docs type-checking."""

    def __init__(self) -> None: ...
    def build_from_file(self, path: str) -> object:
        """Build an AST from ``path`` and return the module node."""
        ...

MANAGER: AstroidManager
"""Singleton manager instance exported by :mod:`astroid.manager`."""
