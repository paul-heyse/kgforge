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
    details: list[BrokenLinkDetail] = msgspec.field(default_factory=list)


class PortalSessions(msgspec.Struct, kw_only=True):
    """Portal usage metrics tracked across runs."""

    builds: int
    unique_users: int


class PortalAnalytics(msgspec.Struct, kw_only=True):
    """Portal analytics namespace."""

    sessions: PortalSessions


class AgentAnalyticsDocument(msgspec.Struct, kw_only=True):
    """Top-level analytics document matching ``doc_analytics.json``."""

    schemaVersion: str = ANALYTICS_SCHEMA_VERSION
    schemaId: str = ANALYTICS_SCHEMA_ID
    generatedAt: str = msgspec.field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    repo: RepoInfo = msgspec.field(default_factory=lambda: RepoInfo(root="."))
    catalog: CatalogMetrics = msgspec.field(
        default_factory=lambda: CatalogMetrics(packages=0, modules=0, symbols=0, shards=0)
    )
    portal: PortalAnalytics = msgspec.field(
        default_factory=lambda: PortalAnalytics(sessions=PortalSessions(builds=0, unique_users=0))
    )
    errors: AnalyticsErrors = msgspec.field(
        default_factory=lambda: AnalyticsErrors(broken_links=0, details=[])
    )
