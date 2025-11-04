"""Testing utilities for the lazy-loading docs script facades."""

from __future__ import annotations

from typing import TYPE_CHECKING

from docs.scripts import (
    build_symbol_index,
    mkdocs_gen_api,
    shared,
    symbol_delta,
    validate_artifacts,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

__all__ = ["clear_lazy_import_caches"]


def _clearers() -> Iterable[Callable[[], None]]:
    return (
        shared.clear_cache,
        build_symbol_index.clear_cache,
        mkdocs_gen_api.clear_cache,
        symbol_delta.clear_cache,
        validate_artifacts.clear_cache,
    )


def clear_lazy_import_caches() -> None:
    """Reset the lazy-import caches for docs script facades."""
    for clear in _clearers():
        clear()
