# ruff: noqa: N815
"""Typed models for Agent Catalog analytics artefacts."""

from __future__ import annotations

from datetime import UTC, datetime

import msgspec

ANALYTICS_SCHEMA = "doc_analytics.json"
ANALYTICS_SCHEMA_ID = "https://kgfoundry.dev/schema/tools/doc_analytics.json"
ANALYTICS_SCHEMA_VERSION = "1.0.0"


class RepoInfo(msgspec.Struct, kw_only=True):
    """Repository metadata."""

    root: str


class CatalogMetrics(msgspec.Struct, kw_only=True):
    """Counts summarising catalog composition."""

    packages: int
    modules: int
    symbols: int
    shards: int


class BrokenLinkDetail(msgspec.Struct, kw_only=True):
    """Details about a broken documentation link."""

    module: str
    path: str | None = None
    page: str | None = None
    kind: str | None = None


class AnalyticsErrors(msgspec.Struct, kw_only=True):
    """Aggregate error metrics for analytics runs."""

    broken_links: int
    details: list[BrokenLinkDetail]

    def __init__(
        self,
        *,
        broken_links: int,
        details: list[BrokenLinkDetail] | None = None,
    ) -> None:
        super().__init__(
            broken_links=broken_links,
            details=list(details) if details is not None else [],
        )


class PortalSessions(msgspec.Struct, kw_only=True):
    """Portal usage metrics tracked across runs."""

    builds: int
    unique_users: int


class PortalAnalytics(msgspec.Struct, kw_only=True):
    """Portal analytics namespace."""

    sessions: PortalSessions


class AgentAnalyticsDocument(msgspec.Struct, kw_only=True):
    """Top-level analytics document matching ``doc_analytics.json``."""

    schemaVersion: str
    schemaId: str
    generatedAt: str
    repo: RepoInfo
    catalog: CatalogMetrics
    portal: PortalAnalytics
    errors: AnalyticsErrors

    def __init__(
        self,
        *,
        generated_at: str | None = None,
        repo: RepoInfo | None = None,
        catalog: CatalogMetrics | None = None,
        portal: PortalAnalytics | None = None,
        errors: AnalyticsErrors | None = None,
    ) -> None:
        super().__init__(
            schemaVersion=ANALYTICS_SCHEMA_VERSION,
            schemaId=ANALYTICS_SCHEMA_ID,
            generatedAt=generated_at or datetime.now(tz=UTC).isoformat(),
            repo=repo or RepoInfo(root="."),
            catalog=catalog or CatalogMetrics(packages=0, modules=0, symbols=0, shards=0),
            portal=portal or PortalAnalytics(sessions=PortalSessions(builds=0, unique_users=0)),
            errors=errors or AnalyticsErrors(broken_links=0),
        )
