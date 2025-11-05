"""Tree-sitter powered code-intelligence utilities."""

from __future__ import annotations

from importlib import import_module

__all__ = ["__version__"]

try:  # pragma: no cover - populated at build time
    __version__ = import_module("codeintel._version").__version__
except ModuleNotFoundError:  # pragma: no cover - development fallback
    __version__ = "0.0.0-dev"
