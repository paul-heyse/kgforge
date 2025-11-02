from __future__ import annotations

from astroid.manager import AstroidManager

class AstroidBuilder:
    """Partial Astroid builder surface invoked during docs configuration."""

    def __init__(self, manager: AstroidManager | None = None) -> None: ...
    def file_build(self, file_path: str, module_name: str) -> object:
        """Return an AST node representing the module at ``file_path``."""
        ...
