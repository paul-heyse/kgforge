"""Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
kgfoundry_common.navmap_types
"""


from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class NavSection(TypedDict):
    """Represent NavSection."""
    
    
    
    
    
    
    
    
    
    
    
    
    

    id: str
    title: str
    symbols: list[str]


class SymbolMeta(TypedDict, total=False):
    """Represent SymbolMeta."""
    
    
    
    
    
    
    
    
    
    
    
    
    

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
    """Represent NavMap."""
    
    
    
    
    
    
    
    
    
    
    
    
    

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
