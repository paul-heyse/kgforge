from collections.abc import Callable
from typing import Any, TypedDict

NavIndex = Any
NavSection = Any
ModuleMeta = Any
SymbolMeta = Any
ModuleEntry = Any

class NavIndexDict(TypedDict):
    commit: str
    policy_version: str
    link_mode: str
    modules: dict[str, Any]

class NavSectionDict(TypedDict):
    id: str
    symbols: list[str]

class ModuleMetaDict(TypedDict, total=False):
    owner: str | None
    stability: str | None
    since: str | None
    deprecated_in: str | None

class SymbolMetaDict(TypedDict, total=False):
    owner: str | None
    stability: str | None
    since: str | None
    deprecated_in: str | None

class ModuleEntryDict(TypedDict):
    path: str
    exports: list[str]
    sections: list[NavSectionDict]
    section_lines: dict[str, int]
    anchors: dict[str, int]
    links: dict[str, str]
    meta: dict[str, SymbolMetaDict]
    module_meta: ModuleMetaDict
    tags: list[str]
    synopsis: str
    see_also: list[str]
    deps: list[str]

nav_index_from_dict: Callable[[NavIndexDict], NavIndex]
