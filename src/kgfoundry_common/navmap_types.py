"""Module for kgfoundry_common.navmap_types.

NavMap:
- NavSection: Section of a module navmap.
- SymbolMeta: Optional metadata describing a documented symbol.
- NavMap: Structure describing a module navmap.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class NavSection(TypedDict):
    """Section of a module navmap."""

    id: str
    title: str
    symbols: list[str]


class SymbolMeta(TypedDict, total=False):
    """Optional metadata describing a documented symbol."""

    since: str
    stability: Literal["frozen", "stable", "experimental", "internal"]
    side_effects: list[Literal["none", "fs", "net", "gpu", "db"]]
    thread_safety: Literal["reentrant", "threadsafe", "not-threadsafe"]
    async_ok: bool
    perf_budget_ms: float
    tests: list[str]
    replaced_by: NotRequired[str]
    deprecated_msg: NotRequired[str]
    contracts: NotRequired[list[str]]
    coverage_target: NotRequired[float]


class NavMap(TypedDict, total=False):
    """Structure describing a module navmap."""

    title: str
    synopsis: str
    exports: list[str]
    sections: list[NavSection]
    see_also: list[str]
    tags: list[str]
    since: str
    deprecated: str
    symbols: dict[str, SymbolMeta]
    edit_scopes: dict[str, list[str]]
    deps: list[str]
