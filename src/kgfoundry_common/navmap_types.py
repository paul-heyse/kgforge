"""Overview of navmap types.

This module bundles navmap types logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""
# [nav:section public-api]

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from kgfoundry_common.navmap_loader import load_nav_metadata

__all__ = [
    "ModuleMeta",
    "NavMap",
    "NavSection",
    "Stability",
    "SymbolMeta",
]
__navmap__ = load_nav_metadata(__name__, tuple(__all__))

# [nav:anchor Stability]
type Stability = Literal[
    "frozen",
    "stable",
    "experimental",
    "internal",
    "beta",
    "deprecated",
]


# [nav:anchor NavSection]
class NavSection(TypedDict):
    """Navigation section metadata.

    TypedDict representing a section in the navigation map. Sections
    group related symbols together for documentation organization.

    Parameters
    ----------
    id : str
        Unique section identifier.
    title : str
        Human-readable section title.
    symbols : list[str]
        List of symbol names included in this section.

    Attributes
    ----------
    id : str
        Unique section identifier.
    title : str
        Human-readable section title.
    symbols : list[str]
        List of symbol names included in this section.
    """

    id: str
    title: str
    symbols: list[str]


# [nav:anchor SymbolMeta]
class SymbolMeta(TypedDict, total=False):
    """Symbol metadata for documentation and tooling.

    TypedDict representing metadata about a symbol (function, class, etc.)
    in the navigation map. Used for documentation generation, testing,
    and tooling support.

    Parameters
    ----------
    since : str
        Version when the symbol was introduced.
    stability : Stability
        Stability level of the symbol.
    side_effects : list[Literal['none', 'fs', 'net', 'gpu', 'db']]
        List of side effects the symbol may have.
    thread_safety : Literal['reentrant', 'threadsafe', 'not-threadsafe']
        Thread safety guarantee of the symbol.
    async_ok : bool
        Whether the symbol is safe to use in async contexts.
    perf_budget_ms : float
        Performance budget in milliseconds.
    tests : list[str]
        List of test file paths or identifiers.
    owner : str, optional
        Owner or team responsible for the symbol.
    deprecated_in : str, optional
        Version when the symbol was deprecated.
    replaced_by : str, optional
        Symbol name that replaces this deprecated symbol.
    deprecated_msg : str, optional
        Deprecation message explaining the change.
    contracts : list[str], optional
        List of contract identifiers (e.g., JSON Schema URIs).
    coverage_target : float, optional
        Test coverage target percentage.

    Attributes
    ----------
    since : str
        Version when the symbol was introduced.
    stability : Stability
        Stability level of the symbol.
    side_effects : list[Literal['none', 'fs', 'net', 'gpu', 'db']]
        List of side effects the symbol may have.
    thread_safety : Literal['reentrant', 'threadsafe', 'not-threadsafe']
        Thread safety guarantee of the symbol.
    async_ok : bool
        Whether the symbol is safe to use in async contexts.
    perf_budget_ms : float
        Performance budget in milliseconds.
    tests : list[str]
        List of test file paths or identifiers.
    owner : NotRequired[str]
        Owner or team responsible for the symbol.
    deprecated_in : NotRequired[str]
        Version when the symbol was deprecated.
    replaced_by : NotRequired[str]
        Symbol name that replaces this deprecated symbol.
    deprecated_msg : NotRequired[str]
        Deprecation message explaining the change.
    contracts : NotRequired[list[str]]
        List of contract identifiers (e.g., JSON Schema URIs).
    coverage_target : NotRequired[float]
        Test coverage target percentage.
    """

    since: str
    stability: Stability
    side_effects: list[Literal["none", "fs", "net", "gpu", "db"]]
    thread_safety: Literal["reentrant", "threadsafe", "not-threadsafe"]
    async_ok: bool
    perf_budget_ms: float
    tests: list[str]
    owner: NotRequired[str]
    deprecated_in: NotRequired[str]
    replaced_by: NotRequired[str]
    deprecated_msg: NotRequired[str]
    contracts: NotRequired[list[str]]
    coverage_target: NotRequired[float]


# [nav:anchor ModuleMeta]
class ModuleMeta(TypedDict, total=False):
    """Module-level metadata for documentation.

    TypedDict representing metadata about a module in the navigation map.
    Used for documentation generation and module organization.

    Parameters
    ----------
    owner : str
        Owner or team responsible for the module.
    stability : Stability
        Stability level of the module.
    since : str
        Version when the module was introduced.
    deprecated_in : str, optional
        Version when the module was deprecated.

    Attributes
    ----------
    owner : str
        Owner or team responsible for the module.
    stability : Stability
        Stability level of the module.
    since : str
        Version when the module was introduced.
    deprecated_in : str
        Version when the module was deprecated.
    """

    owner: str
    stability: Stability
    since: str
    deprecated_in: str


# [nav:anchor NavMap]
class NavMap(TypedDict, total=False):
    """Navigation map metadata structure.

    TypedDict representing the complete navigation map for a module.
    Contains all metadata needed for documentation generation, including
    exports, sections, symbol metadata, and module information.

    Parameters
    ----------
    title : str
        Module title.
    synopsis : str
        Brief description of the module.
    exports : list[str]
        List of public symbol names exported by the module.
    sections : list[NavSection]
        List of navigation sections organizing the symbols.
    see_also : list[str], optional
        List of related modules or resources.
    tags : list[str], optional
        List of tags for categorization.
    since : str, optional
        Version when the module was introduced.
    deprecated : str, optional
        Version when the module was deprecated.
    symbols : dict[str, SymbolMeta], optional
        Dictionary mapping symbol names to their metadata.
    edit_scopes : dict[str, list[str]], optional
        Dictionary mapping edit scopes to symbol lists.
    deps : list[str], optional
        List of dependency module names.
    module_meta : ModuleMeta, optional
        Module-level metadata.

    Attributes
    ----------
    title : str
        Module title.
    synopsis : str
        Brief description of the module.
    exports : list[str]
        List of public symbol names exported by the module.
    sections : list[NavSection]
        List of navigation sections organizing the symbols.
    see_also : list[str]
        List of related modules or resources.
    tags : list[str]
        List of tags for categorization.
    since : str
        Version when the module was introduced.
    deprecated : str
        Version when the module was deprecated.
    symbols : dict[str, SymbolMeta]
        Dictionary mapping symbol names to their metadata.
    edit_scopes : dict[str, list[str]]
        Dictionary mapping edit scopes to symbol lists.
    deps : list[str]
        List of dependency module names.
    module_meta : ModuleMeta
        Module-level metadata.
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
    module_meta: ModuleMeta
