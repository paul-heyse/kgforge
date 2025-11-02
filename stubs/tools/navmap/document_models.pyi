# ruff: noqa: N803,N815
from __future__ import annotations

from tools.navmap.models import NavIndex

NAVMAP_SCHEMA: str
NAVMAP_SCHEMA_ID: str
NAVMAP_SCHEMA_VERSION: str

class NavSectionDocument:
    id: str
    symbols: list[str]

    def __init__(self, *, id: str, symbols: list[str]) -> None: ...  # noqa: A002 - schema field name

class SymbolMetaDocument:
    owner: str | None
    stability: str | None
    since: str | None
    deprecatedIn: str | None

    def __init__(
        self,
        *,
        owner: str | None = ...,
        stability: str | None = ...,
        since: str | None = ...,
        deprecatedIn: str | None = ...,
    ) -> None: ...

class ModuleMetaDocument:
    owner: str | None
    stability: str | None
    since: str | None
    deprecatedIn: str | None

    def __init__(
        self,
        *,
        owner: str | None = ...,
        stability: str | None = ...,
        since: str | None = ...,
        deprecatedIn: str | None = ...,
    ) -> None: ...

class ModuleEntryDocument:
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

    def __init__(
        self,
        *,
        path: str,
        exports: list[str],
        sections: list[NavSectionDocument],
        sectionLines: dict[str, int],
        anchors: dict[str, int],
        links: dict[str, str],
        meta: dict[str, SymbolMetaDocument],
        moduleMeta: ModuleMetaDocument,
        tags: list[str],
        synopsis: str,
        seeAlso: list[str],
        deps: list[str],
    ) -> None: ...

class NavmapDocument:
    schemaVersion: str
    schemaId: str
    generatedAt: str
    commit: str
    policyVersion: str
    linkMode: str
    modules: dict[str, ModuleEntryDocument]

    def __init__(
        self,
        *,
        schemaVersion: str = ...,
        schemaId: str = ...,
        generatedAt: str | None = ...,
        commit: str = ...,
        policyVersion: str = ...,
        linkMode: str = ...,
        modules: dict[str, ModuleEntryDocument] | None = ...,
    ) -> None: ...

def navmap_document_from_index(
    index: NavIndex,
    *,
    commit: str,
    policy_version: str,
    link_mode: str,
) -> NavmapDocument: ...
