"""Overview of navmap types.

This module bundles navmap types logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

Stability = Literal["frozen", "stable", "experimental", "internal", "beta", "deprecated"]


class NavSection(TypedDict):
    """Model the NavSection.

    Represent the navsection data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    id: str
    title: str
    symbols: list[str]


class SymbolMeta(TypedDict, total=False):
    """Model the SymbolMeta.

    Represent the symbolmeta data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
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


class ModuleMeta(TypedDict, total=False):
    """Describe module-level metadata surfaced in navigation structures."""

    owner: str
    stability: Stability
    since: str
    deprecated_in: str


class NavMap(TypedDict, total=False):
    """Model the NavMap.

    Represent the navmap data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
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
