# ruff: noqa: N815, PLR0913
"""Navmap document models aligned with ``schema/tools/navmap_document.json``."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import msgspec

from tools.navmap.models import NavIndex, SymbolMeta

NAVMAP_SCHEMA = "navmap_document.json"
NAVMAP_SCHEMA_ID = "https://kgfoundry.dev/schema/tools/navmap-document.json"
NAVMAP_SCHEMA_VERSION = "1.0.0"

if TYPE_CHECKING:

    class BaseStruct:
        """Typed placeholder for :class:`msgspec.Struct` during analysis."""

        def __init__(self, *args: object, **kwargs: object) -> None: ...

        def __init_subclass__(
            cls,
            *,
            kw_only: bool = False,
            **kwargs: object,
        ) -> None:
            """Accept struct keyword-only modifiers for type checking."""

else:
    BaseStruct = msgspec.Struct


class NavSectionDocument(BaseStruct, kw_only=True):
    """Serialized representation of a navigation section."""

    id: str
    symbols: list[str]

    if TYPE_CHECKING:

        def __init__(self, *, id: str, symbols: list[str]) -> None: ...  # noqa: A002


class SymbolMetaDocument(BaseStruct, kw_only=True):
    """Symbol metadata emitted in navmap documents."""

    owner: str | None = None
    stability: str | None = None
    since: str | None = None
    deprecatedIn: str | None = None

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            owner: str | None = None,
            stability: str | None = None,
            since: str | None = None,
            deprecatedIn: str | None = None,  # noqa: N803
        ) -> None: ...


class ModuleMetaDocument(BaseStruct, kw_only=True):
    """Module-level metadata defaults."""

    owner: str | None = None
    stability: str | None = None
    since: str | None = None
    deprecatedIn: str | None = None

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            owner: str | None = None,
            stability: str | None = None,
            since: str | None = None,
            deprecatedIn: str | None = None,  # noqa: N803
        ) -> None: ...


class ModuleEntryDocument(BaseStruct, kw_only=True):
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

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            path: str,
            exports: list[str],
            sections: list[NavSectionDocument],
            sectionLines: dict[str, int],  # noqa: N803
            anchors: dict[str, int],
            links: dict[str, str],
            meta: dict[str, SymbolMetaDocument],
            moduleMeta: ModuleMetaDocument,  # noqa: N803
            tags: list[str],
            synopsis: str,
            seeAlso: list[str],  # noqa: N803
            deps: list[str],
        ) -> None: ...


def _utc_iso_now() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(tz=UTC).isoformat()


def _empty_module_map() -> dict[str, ModuleEntryDocument]:
    """Return an empty module map for :class:`NavmapDocument`."""
    return {}


class NavmapDocument(BaseStruct, kw_only=True):
    """Top-level navmap document captured on disk."""

    schemaVersion: str = NAVMAP_SCHEMA_VERSION
    schemaId: str = NAVMAP_SCHEMA_ID
    generatedAt: str = cast(str, msgspec.field(default_factory=_utc_iso_now))
    commit: str = "HEAD"
    policyVersion: str = "1"
    linkMode: str = "editor"
    modules: dict[str, ModuleEntryDocument] = cast(
        dict[str, ModuleEntryDocument],
        msgspec.field(default_factory=_empty_module_map),
    )

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            schemaVersion: str = NAVMAP_SCHEMA_VERSION,  # noqa: N803
            schemaId: str = NAVMAP_SCHEMA_ID,  # noqa: N803
            generatedAt: str | None = None,  # noqa: N803
            commit: str = "HEAD",
            policyVersion: str = "1",  # noqa: N803
            linkMode: str = "editor",  # noqa: N803
            modules: dict[str, ModuleEntryDocument] | None = None,
        ) -> None: ...


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
