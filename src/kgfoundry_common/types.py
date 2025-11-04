"""Type aliases for kgfoundry_common.

This module provides shared type definitions without any dependencies, avoiding circular imports
across the package.
"""

from __future__ import annotations

__all__ = ["JsonPrimitive", "JsonValue"]

# Primitive JSON types (leaf values)
type JsonPrimitive = str | int | float | bool | None

# JSON value can be primitive or nested (dict/list)
type JsonValue = JsonPrimitive | dict[str, JsonValue] | list[JsonValue]
