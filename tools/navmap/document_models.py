"""Navmap document models aligned with ``schema/tools/navmap_document.json``."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Final, cast

from pydantic import BaseModel, ConfigDict, Field

from tools.navmap.models import NavIndex, SymbolMeta

NAVMAP_SCHEMA: Final[str] = "navmap_document.json"
NAVMAP_SCHEMA_ID: Final[str] = "https://kgfoundry.dev/schema/tools/navmap-document.json"
NAVMAP_SCHEMA_VERSION: Final[str] = "1.0.0"


class NavSectionDocument(BaseModel):
    """Serialized representation of a navigation section."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    symbols: list[str]


class SymbolMetaDocument(BaseModel):
    """Symbol metadata emitted in navmap documents."""

    model_config = ConfigDict(populate_by_name=True)

    owner: str | None = None
    stability: str | None = None
    since: str | None = None
    deprecated_in: str | None = Field(None, alias="deprecatedIn")


class ModuleMetaDocument(BaseModel):
    """Module-level metadata defaults."""

    model_config = ConfigDict(populate_by_name=True)

    owner: str | None = None
    stability: str | None = None
    since: str | None = None
    deprecated_in: str | None = Field(None, alias="deprecatedIn")


class ModuleEntryDocument(BaseModel):
    """Serialized module entry within the navmap document."""

    model_config = ConfigDict(populate_by_name=True)

    path: str
    exports: list[str]
    sections: list[NavSectionDocument]
    section_lines: dict[str, int] = Field(alias="sectionLines")
    anchors: dict[str, int]
    links: dict[str, str]
    meta: dict[str, SymbolMetaDocument]
    module_meta: ModuleMetaDocument = Field(alias="moduleMeta")
    tags: list[str]
    synopsis: str
    see_also: list[str] = Field(alias="seeAlso")
    deps: list[str]


def _utc_iso_now() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(tz=UTC).isoformat()


class NavmapDocument(BaseModel):
    """Top-level navmap document captured on disk."""

    model_config = ConfigDict(populate_by_name=True)

    schema_version: str = Field(NAVMAP_SCHEMA_VERSION, alias="schemaVersion")
    schema_id: str = Field(NAVMAP_SCHEMA_ID, alias="schemaId")
    generated_at: str = Field(default_factory=_utc_iso_now, alias="generatedAt")
    commit: str = "HEAD"
    policy_version: str = Field("1", alias="policyVersion")
    link_mode: str = Field("editor", alias="linkMode")
    modules: dict[str, ModuleEntryDocument] = Field(default_factory=dict)


def _symbol_meta_to_document(meta: SymbolMeta) -> SymbolMetaDocument:
    return SymbolMetaDocument(
        owner=meta.owner,
        stability=meta.stability,
        since=meta.since,
        deprecated_in=meta.deprecated_in,
    )


def _module_meta_to_document(module_meta: Mapping[str, str | None]) -> ModuleMetaDocument:
    return ModuleMetaDocument(
        owner=module_meta.get("owner"),
        stability=module_meta.get("stability"),
        since=module_meta.get("since"),
        deprecated_in=module_meta.get("deprecated_in"),
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
        sections: list[NavSectionDocument] = [
            NavSectionDocument(id=section.id, symbols=list(section.symbols))
            for section in entry.sections
        ]
        symbol_meta: dict[str, SymbolMetaDocument] = {
            symbol: _symbol_meta_to_document(meta) for symbol, meta in entry.meta.items()
        }
        module_meta_doc = _module_meta_to_document(
            cast(Mapping[str, str | None], entry.module_meta.to_dict())
        )
        modules[name] = ModuleEntryDocument(
            path=entry.path,
            exports=list(entry.exports),
            sections=sections,
            section_lines=dict(entry.section_lines),
            anchors=dict(entry.anchors),
            links=dict(entry.links),
            meta=symbol_meta,
            module_meta=module_meta_doc,
            tags=list(entry.tags),
            synopsis=entry.synopsis,
            see_also=list(entry.see_also),
            deps=list(entry.deps),
        )

    return NavmapDocument(
        commit=commit,
        policy_version=policy_version,
        link_mode=link_mode,
        modules=modules,
    )
