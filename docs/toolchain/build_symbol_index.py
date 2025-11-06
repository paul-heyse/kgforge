"""Build symbol index with typed configuration.

This module provides a new public API for building documentation symbol indices using typed
configuration objects instead of positional arguments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docs.toolchain.config import DocsSymbolIndexConfig


def build_symbol_index(*, config: DocsSymbolIndexConfig) -> dict[str, object]:
    """Build a symbol index with typed configuration.

    This is the new public API for symbol index building that accepts a typed
    configuration object instead of boolean positional arguments.

    Parameters
    ----------
    config : DocsSymbolIndexConfig
        Typed configuration controlling symbol index build behavior.

    Returns
    -------
    dict[str, object]
        Symbol index dictionary containing rows, by_file, and by_module mappings.

    Raises
    ------
    NotImplementedError
        This function is a placeholder for Phase 3.2 implementation.
        The return type annotation indicates the intended return value once
        this function is fully implemented.

    Examples
    --------
    >>> from docs.toolchain.config import DocsSymbolIndexConfig
    >>> config = DocsSymbolIndexConfig(include_private=False)
    >>> # index = build_symbol_index(config=config)
    """  # noqa: DOC202
    msg = "build_symbol_index is a placeholder for Phase 3.2 implementation"
    raise NotImplementedError(msg)


__all__ = [
    "build_symbol_index",
]
