"""Typed models for navmap documents.

This module provides typed dataclasses and TypedDict definitions for navmap
structures, ensuring type safety and schema compliance across navmap operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class NavSectionDict(TypedDict):
    """Represent a normalized nav section."""

    id: str
    symbols: list[str]


class SymbolMetaDict(TypedDict, total=False):
    """Represent metadata for a single symbol."""

    owner: str
    stability: str
    since: str
    deprecated_in: str


class ModuleMetaDict(TypedDict, total=False):
    """Represent module-level metadata defaults."""

    owner: str
    stability: str
    since: str
    deprecated_in: str


class ModuleEntryDict(TypedDict):
    """Represent the JSON entry for a module."""

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


class NavIndexDict(TypedDict):
    """Represent the persisted nav index."""

    commit: str
    policy_version: str
    link_mode: str
    modules: dict[str, ModuleEntryDict]


@dataclass(slots=True, frozen=True)
class NavSection:
    """Typed model for a nav section."""

    id: str
    symbols: list[str]

    def to_dict(self) -> NavSectionDict:
        """Convert to TypedDict representation."""
        return NavSectionDict(id=self.id, symbols=self.symbols)


@dataclass(slots=True, frozen=True)
class SymbolMeta:
    """Typed model for symbol metadata."""

    owner: str | None = None
    stability: str | None = None
    since: str | None = None
    deprecated_in: str | None = None

    def to_dict(self) -> SymbolMetaDict:
        """Convert to TypedDict representation."""
        result: SymbolMetaDict = {}
        if self.owner is not None:
            result["owner"] = self.owner
        if self.stability is not None:
            result["stability"] = self.stability
        if self.since is not None:
            result["since"] = self.since
        if self.deprecated_in is not None:
            result["deprecated_in"] = self.deprecated_in
        return result


@dataclass(slots=True, frozen=True)
class ModuleMeta:
    """Typed model for module-level metadata."""

    owner: str | None = None
    stability: str | None = None
    since: str | None = None
    deprecated_in: str | None = None

    def to_dict(self) -> ModuleMetaDict:
        """Convert to TypedDict representation."""
        result: ModuleMetaDict = {}
        if self.owner is not None:
            result["owner"] = self.owner
        if self.stability is not None:
            result["stability"] = self.stability
        if self.since is not None:
            result["since"] = self.since
        if self.deprecated_in is not None:
            result["deprecated_in"] = self.deprecated_in
        return result


@dataclass(slots=True, frozen=True)
class ModuleEntry:
    """Typed model for a module entry."""

    path: str
    exports: list[str]
    sections: list[NavSection]
    section_lines: dict[str, int]
    anchors: dict[str, int]
    links: dict[str, str]
    meta: dict[str, SymbolMeta]
    module_meta: ModuleMeta
    tags: list[str]
    synopsis: str
    see_also: list[str]
    deps: list[str]

    def to_dict(self) -> ModuleEntryDict:
        """Convert to TypedDict representation."""
        return ModuleEntryDict(
            path=self.path,
            exports=self.exports,
            sections=[section.to_dict() for section in self.sections],
            section_lines=self.section_lines,
            anchors=self.anchors,
            links=self.links,
            meta={name: meta.to_dict() for name, meta in self.meta.items()},
            module_meta=self.module_meta.to_dict(),
            tags=self.tags,
            synopsis=self.synopsis,
            see_also=self.see_also,
            deps=self.deps,
        )


@dataclass(slots=True, frozen=True)
class NavIndex:
    """Typed model for the nav index."""

    commit: str
    policy_version: str
    link_mode: str
    modules: dict[str, ModuleEntry]

    def to_dict(self) -> NavIndexDict:
        """Convert to TypedDict representation."""
        return NavIndexDict(
            commit=self.commit,
            policy_version=self.policy_version,
            link_mode=self.link_mode,
            modules={name: entry.to_dict() for name, entry in self.modules.items()},
        )


def nav_index_from_dict(data: NavIndexDict) -> NavIndex:
    """Create a NavIndex from a TypedDict.

    Parameters
    ----------
    data : NavIndexDict
        The dictionary representation of a nav index.

    Returns
    -------
    NavIndex
        The typed model instance.
    """
    modules: dict[str, ModuleEntry] = {}
    for module_name, module_data in data["modules"].items():
        sections = [
            NavSection(id=section["id"], symbols=section["symbols"])
            for section in module_data["sections"]
        ]
        meta = {
            name: SymbolMeta(
                owner=symbol_meta.get("owner"),
                stability=symbol_meta.get("stability"),
                since=symbol_meta.get("since"),
                deprecated_in=symbol_meta.get("deprecated_in"),
            )
            for name, symbol_meta in module_data["meta"].items()
        }
        module_meta = ModuleMeta(
            owner=module_data["module_meta"].get("owner"),
            stability=module_data["module_meta"].get("stability"),
            since=module_data["module_meta"].get("since"),
            deprecated_in=module_data["module_meta"].get("deprecated_in"),
        )
        modules[module_name] = ModuleEntry(
            path=module_data["path"],
            exports=module_data["exports"],
            sections=sections,
            section_lines=module_data["section_lines"],
            anchors=module_data["anchors"],
            links=module_data["links"],
            meta=meta,
            module_meta=module_meta,
            tags=module_data["tags"],
            synopsis=module_data["synopsis"],
            see_also=module_data["see_also"],
            deps=module_data["deps"],
        )
    return NavIndex(
        commit=data["commit"],
        policy_version=data["policy_version"],
        link_mode=data["link_mode"],
        modules=modules,
    )
