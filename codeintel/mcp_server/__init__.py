"""MCP servers and bridges exposing Tree-sitter code intelligence tools."""

from __future__ import annotations

from importlib import import_module

__all__ = ["__version__"]

try:  # pragma: no cover - populated when packaging
    __version__ = import_module("codeintel.mcp_server._version").__version__
except ModuleNotFoundError:  # pragma: no cover - development fallback
    __version__ = "0.0.0-dev"
