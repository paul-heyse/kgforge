# ruff: noqa: N803,N815
from __future__ import annotations

ANALYTICS_SCHEMA: str
ANALYTICS_SCHEMA_ID: str
ANALYTICS_SCHEMA_VERSION: str

class RepoInfo:
    root: str

    def __init__(self, *, root: str) -> None: ...

class CatalogMetrics:
    packages: int
    modules: int
    symbols: int
    shards: int

    def __init__(
        self,
        *,
        packages: int,
        modules: int,
        symbols: int,
        shards: int,
    ) -> None: ...

class BrokenLinkDetail:
    module: str
    path: str | None
    page: str | None
    kind: str | None

    def __init__(
        self,
        *,
        module: str,
        path: str | None = ...,
        page: str | None = ...,
        kind: str | None = ...,
    ) -> None: ...

class AnalyticsErrors:
    broken_links: int
    details: list[BrokenLinkDetail]

    def __init__(
        self,
        *,
        broken_links: int,
        details: list[BrokenLinkDetail] | None = ...,
    ) -> None: ...

class PortalSessions:
    builds: int
    unique_users: int

    def __init__(self, *, builds: int, unique_users: int) -> None: ...

class PortalAnalytics:
    sessions: PortalSessions

    def __init__(self, *, sessions: PortalSessions) -> None: ...

class AgentAnalyticsDocument:
    schemaVersion: str
    schemaId: str
    generatedAt: str | None
    repo: RepoInfo
    catalog: CatalogMetrics
    portal: PortalAnalytics
    errors: AnalyticsErrors

    def __init__(
        self,
        *,
        schemaVersion: str = ...,
        schemaId: str = ...,
        generatedAt: str | None = ...,
        repo: RepoInfo | None = ...,
        catalog: CatalogMetrics | None = ...,
        portal: PortalAnalytics | None = ...,
        errors: AnalyticsErrors | None = ...,
    ) -> None: ...
