"""Typed models for Agent Catalog analytics artefacts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

from pydantic import BaseModel, ConfigDict, Field


def _default_generated_at() -> str:
    return datetime.now(tz=UTC).isoformat()


def _default_repo_info() -> RepoInfo:
    return RepoInfo(root=".")


def _default_catalog_metrics() -> CatalogMetrics:
    return CatalogMetrics(packages=0, modules=0, symbols=0, shards=0)


def _default_portal_sessions() -> PortalSessions:
    return PortalSessions(builds=0, unique_users=0)


def _default_portal_analytics() -> PortalAnalytics:
    return PortalAnalytics(sessions=_default_portal_sessions())


def _empty_broken_link_details() -> list[BrokenLinkDetail]:
    return []


def _default_errors() -> AnalyticsErrors:
    return AnalyticsErrors(broken_links=0)


ANALYTICS_SCHEMA: Final[str] = "doc_analytics.json"
ANALYTICS_SCHEMA_ID: Final[str] = "https://kgfoundry.dev/schema/tools/doc_analytics.json"
ANALYTICS_SCHEMA_VERSION: Final[str] = "1.0.0"


class RepoInfo(BaseModel):
    """Repository metadata."""

    model_config = ConfigDict(populate_by_name=True)

    root: str


class CatalogMetrics(BaseModel):
    """Counts summarising catalog composition."""

    model_config = ConfigDict(populate_by_name=True)

    packages: int
    modules: int
    symbols: int
    shards: int


class BrokenLinkDetail(BaseModel):
    """Details about a broken documentation link."""

    model_config = ConfigDict(populate_by_name=True)

    module: str
    path: str | None = None
    page: str | None = None
    kind: str | None = None


class AnalyticsErrors(BaseModel):
    """Aggregate error metrics for analytics runs."""

    model_config = ConfigDict(populate_by_name=True)

    broken_links: int
    details: list[BrokenLinkDetail] = Field(default_factory=_empty_broken_link_details)


class PortalSessions(BaseModel):
    """Portal usage metrics tracked across runs."""

    model_config = ConfigDict(populate_by_name=True)

    builds: int
    unique_users: int


class PortalAnalytics(BaseModel):
    """Portal analytics namespace."""

    model_config = ConfigDict(populate_by_name=True)

    sessions: PortalSessions


class AgentAnalyticsDocument(BaseModel):
    """Top-level analytics document matching ``doc_analytics.json``."""

    model_config = ConfigDict(populate_by_name=True)

    schema_version: str = Field(ANALYTICS_SCHEMA_VERSION, alias="schemaVersion")
    schema_id: str = Field(ANALYTICS_SCHEMA_ID, alias="schemaId")
    generated_at: str = Field(default_factory=_default_generated_at, alias="generatedAt")
    repo: RepoInfo = Field(default_factory=_default_repo_info)
    catalog: CatalogMetrics = Field(default_factory=_default_catalog_metrics)
    portal: PortalAnalytics = Field(default_factory=_default_portal_analytics)
    errors: AnalyticsErrors = Field(default_factory=_default_errors)
