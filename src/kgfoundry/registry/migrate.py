"""Expose ``registry.migrate`` inside the ``kgfoundry`` namespace."""

from __future__ import annotations

from registry.migrate import apply, main

__all__ = ["apply", "main"]
