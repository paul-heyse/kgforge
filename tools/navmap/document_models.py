# ruff: noqa: N815
"""Navmap document models aligned with ``schema/tools/navmap_document.json``."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

import msgspec
from tools.navmap.models import NavIndex, SymbolMeta

NAVMAP_SCHEMA = "navmap_document.json"
NAVMAP_SCHEMA_ID = "https://kgfoundry.dev/schema/tools/navmap-document.json"
NAVMAP_SCHEMA_VERSION = "1.0.0"


class NavSectionDocument(msgspec.Struct, kw_only=True):
    """Serialized representation of a navigation section."""

    id: str
    symbols: list[str]


class SymbolMetaDocument(msgspec.Struct, kw_only=True):
    """Symbol metadata emitted in navmap documents."""

    owner: str | None = None
    stability: str | None = None
    since: str | None = None
    deprecatedIn: str | None = None


class ModuleMetaDocument(msgspec.Struct, kw_only=True):
    """Module-level metadata defaults."""

    owner: str | None = None
    stability: str | None = None
    since: str | None = None
    deprecatedIn: str | None = None


class ModuleEntryDocument(msgspec.Struct, kw_only=True):
    """Serialized module entry within the navmap document."""

    path: str
    exports: list[str]
    sections: list[NavSectionDocument]
    sectionLines: dict[str, int]
    anchors: dict[str, int]
    links: dict[str, str]
    meta: dict[str, SymbolMetaDocument]
    moduleMeta: ModuleMetaDocument
    tags: list[str]
    synopsis: str
    seeAlso: list[str]
    deps: list[str]


class NavmapDocument(msgspec.Struct, kw_only=True):
    """Top-level navmap document captured on disk."""

    schemaVersion: str = NAVMAP_SCHEMA_VERSION
    schemaId: str = NAVMAP_SCHEMA_ID
    generatedAt: str = msgspec.field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    commit: str = "HEAD"
    policyVersion: str = "1"
    linkMode: str = "editor"
    modules: dict[str, ModuleEntryDocument] = msgspec.field(default_factory=dict)


def _symbol_meta_to_document(meta: SymbolMeta) -> SymbolMetaDocument:
    return SymbolMetaDocument(
        owner=meta.owner,
        stability=meta.stability,
        since=meta.since,
        deprecatedIn=meta.deprecated_in,
    )


def _module_meta_to_document(module_meta: Mapping[str, str | None]) -> ModuleMetaDocument:
    return ModuleMetaDocument(
        owner=module_meta.get("owner"),
        stability=module_meta.get("stability"),
        since=module_meta.get("since"),
        deprecatedIn=module_meta.get("deprecated_in"),
    )


def navmap_document_from_index(
    index: NavIndex,
    *,
    commit: str,
    policy_version: str,
    link_mode: str,
) -> NavmapDocument:
    """Build a :class:`NavmapDocument` from a :class:`NavIndex` instance."""
    modules: dict[str, ModuleEntryDocument] = {}
    for name, entry in index.modules.items():
        sections = [
            NavSectionDocument(id=section.id, symbols=list(section.symbols))
            for section in entry.sections
        ]
        symbol_meta = {
            symbol: _symbol_meta_to_document(meta) for symbol, meta in entry.meta.items()
        }
        module_meta_doc = _module_meta_to_document(entry.module_meta.to_dict())
        modules[name] = ModuleEntryDocument(
            path=entry.path,
            exports=list(entry.exports),
            sections=sections,
            sectionLines=dict(entry.section_lines),
            anchors=dict(entry.anchors),
            links=dict(entry.links),
            meta=symbol_meta,
            moduleMeta=module_meta_doc,
            tags=list(entry.tags),
            synopsis=entry.synopsis,
            seeAlso=list(entry.see_also),
            deps=list(entry.deps),
        )

    return NavmapDocument(
        commit=commit,
        policyVersion=policy_version,
        linkMode=link_mode,
        modules=modules,
    )
