"""Persistent index for CodeIntel symbols and references."""

from __future__ import annotations

from codeintel.index.store import (
    EXCLUDES,
    EXT_TO_LANG,
    FileMeta,
    IndexStore,
    detect_lang,
    ensure_schema,
    find_references,
    index_incremental,
    needs_reindex,
    replace_file,
    search_symbols,
    stat_meta,
)

__all__ = [
    "EXCLUDES",
    "EXT_TO_LANG",
    "FileMeta",
    "IndexStore",
    "detect_lang",
    "ensure_schema",
    "find_references",
    "index_incremental",
    "needs_reindex",
    "replace_file",
    "search_symbols",
    "stat_meta",
]
