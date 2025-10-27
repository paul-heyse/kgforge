"""
Provide utilities for module.

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
    """
    Represent NavSection.
    
    Attributes
    ----------
    id : str
        Attribute description.
    title : str
        Attribute description.
    symbols : List[str]
        Attribute description.
    
    Examples
    --------
    >>> from kgfoundry_common.navmap_types import NavSection
    >>> result = NavSection()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.navmap_types
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

    id: str
    title: str
    symbols: list[str]


class SymbolMeta(TypedDict, total=False):
    """
    Represent SymbolMeta.
    
    Attributes
    ----------
    since : str
        Attribute description.
    stability : Literal['frozen', 'stable', 'experimental', 'internal']
        Attribute description.
    side_effects : List[Literal['none', 'fs', 'net', 'gpu', 'db']]
        Attribute description.
    thread_safety : Literal['reentrant', 'threadsafe', 'not-threadsafe']
        Attribute description.
    async_ok : bool
        Attribute description.
    perf_budget_ms : float
        Attribute description.
    tests : List[str]
        Attribute description.
    replaced_by : NotRequired[str]
        Attribute description.
    deprecated_msg : NotRequired[str]
        Attribute description.
    contracts : NotRequired[List[str]]
        Attribute description.
    coverage_target : NotRequired[float]
        Attribute description.
    
    Examples
    --------
    >>> from kgfoundry_common.navmap_types import SymbolMeta
    >>> result = SymbolMeta()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.navmap_types
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

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
    """
    Represent NavMap.
    
    Attributes
    ----------
    title : str
        Attribute description.
    synopsis : str
        Attribute description.
    exports : List[str]
        Attribute description.
    sections : List[NavSection]
        Attribute description.
    see_also : List[str]
        Attribute description.
    tags : List[str]
        Attribute description.
    since : str
        Attribute description.
    deprecated : str
        Attribute description.
    symbols : Mapping[str, SymbolMeta]
        Attribute description.
    edit_scopes : Mapping[str, List[str]]
        Attribute description.
    deps : List[str]
        Attribute description.
    
    Examples
    --------
    >>> from kgfoundry_common.navmap_types import NavMap
    >>> result = NavMap()
    >>> result  # doctest: +ELLIPSIS
    ...
    
    See Also
    --------
    kgfoundry_common.navmap_types
    
    Notes
    -----
    Document class invariants and lifecycle details here.
    """
    
    
    
    

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
