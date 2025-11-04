"""Tests for tools.docstring_builder.io helpers."""

from __future__ import annotations

from tools.docstring_builder.io import module_to_path
from tools.docstring_builder.paths import REPO_ROOT


def test_module_to_path_resolves_tools_package() -> None:
    """Ensure modules located under ``tools`` resolve to their actual file path."""
    expected = REPO_ROOT / "tools" / "docstring_builder" / "io.py"
    resolved = module_to_path("tools.docstring_builder.io")

    assert resolved == expected


def test_module_to_path_falls_back_to_src_when_missing() -> None:
    """Fall back to the src layout when the module cannot be resolved."""
    candidate = module_to_path("nonexistent.module")

    assert candidate == REPO_ROOT / "src" / "nonexistent" / "module.py"
