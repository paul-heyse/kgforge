# ruff: noqa: N815
"""Typed models for Agent Catalog analytics artefacts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import msgspec

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


ANALYTICS_SCHEMA = "doc_analytics.json"
ANALYTICS_SCHEMA_ID = "https://kgfoundry.dev/schema/tools/doc_analytics.json"
ANALYTICS_SCHEMA_VERSION = "1.0.0"


class RepoInfo(BaseStruct, kw_only=True):
    """Repository metadata."""

    root: str

    if TYPE_CHECKING:

        def __init__(self, *, root: str) -> None: ...


class CatalogMetrics(BaseStruct, kw_only=True):
    """Counts summarising catalog composition."""

    packages: int
    modules: int
    symbols: int
    shards: int

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            packages: int,
            modules: int,
            symbols: int,
            shards: int,
        ) -> None: ...


class BrokenLinkDetail(BaseStruct, kw_only=True):
    """Details about a broken documentation link."""

    module: str
    path: str | None = None
    page: str | None = None
    kind: str | None = None

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            module: str,
            path: str | None = None,
            page: str | None = None,
            kind: str | None = None,
        ) -> None: ...


class AnalyticsErrors(BaseStruct, kw_only=True):
    """Aggregate error metrics for analytics runs."""

    broken_links: int
    details: list[BrokenLinkDetail]

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            broken_links: int,
            details: list[BrokenLinkDetail] | None = None,
        ) -> None: ...

    else:
        details = msgspec.field(default_factory=_empty_broken_link_details)


class PortalSessions(BaseStruct, kw_only=True):
    """Portal usage metrics tracked across runs."""

    builds: int
    unique_users: int

    if TYPE_CHECKING:

        def __init__(self, *, builds: int, unique_users: int) -> None: ...


class PortalAnalytics(BaseStruct, kw_only=True):
    """Portal analytics namespace."""

    sessions: PortalSessions

    if TYPE_CHECKING:

        def __init__(self, *, sessions: PortalSessions) -> None: ...


class AgentAnalyticsDocument(BaseStruct, kw_only=True):
    """Top-level analytics document matching ``doc_analytics.json``."""

    schemaVersion: str = ANALYTICS_SCHEMA_VERSION
    schemaId: str = ANALYTICS_SCHEMA_ID
    generatedAt: str
    repo: RepoInfo
    catalog: CatalogMetrics
    portal: PortalAnalytics
    errors: AnalyticsErrors

    if TYPE_CHECKING:

        def __init__(  # noqa: PLR0913
            self,
            *,
            schemaVersion: str = ANALYTICS_SCHEMA_VERSION,  # noqa: N803
            schemaId: str = ANALYTICS_SCHEMA_ID,  # noqa: N803
            generatedAt: str | None = None,  # noqa: N803
            repo: RepoInfo | None = None,
            catalog: CatalogMetrics | None = None,
            portal: PortalAnalytics | None = None,
            errors: AnalyticsErrors | None = None,
        ) -> None: ...

    else:
        generatedAt = msgspec.field(default_factory=_default_generated_at)
        repo = msgspec.field(default_factory=_default_repo_info)
        catalog = msgspec.field(default_factory=_default_catalog_metrics)
        portal = msgspec.field(default_factory=_default_portal_analytics)
        errors = msgspec.field(default_factory=_default_errors)
