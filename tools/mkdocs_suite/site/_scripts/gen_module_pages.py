"""Reuse the docs-side module page generator for the packaged MkDocs site."""

from __future__ import annotations

from tools.mkdocs_suite.docs._scripts.gen_module_pages import (
    render_module_pages as _render_module_pages,
)

__all__ = ["render_module_pages"]


def render_module_pages() -> None:
    """Generate module pages for the packaged MkDocs site."""
    _render_module_pages()


if __name__ == "__main__":  # pragma: no cover - executed by mkdocs
    render_module_pages()
