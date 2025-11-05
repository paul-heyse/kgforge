"""MCP servers and bridges exposing Tree-sitter code intelligence tools."""

from __future__ import annotations

__all__ = ["__version__"]

try:  # pragma: no cover - populated when packaging
    from ._version import __version__  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - development fallback
    __version__ = "0.0.0-dev"
