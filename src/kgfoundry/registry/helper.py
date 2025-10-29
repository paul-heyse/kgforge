"""Expose ``registry.helper`` inside the ``kgfoundry`` namespace."""

from __future__ import annotations

from registry.helper import DuckDBRegistryHelper

__all__ = ["DuckDBRegistryHelper"]
