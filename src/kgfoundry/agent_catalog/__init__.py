"""Public interface for the agent catalog client and utilities."""
# [nav:section public-api]

from __future__ import annotations

from kgfoundry.agent_catalog import search
from kgfoundry.agent_catalog.audit import AuditLogger
from kgfoundry.agent_catalog.client import AgentCatalogClient, AgentCatalogClientError
from kgfoundry.agent_catalog.models import (
    AgentCatalogModel,
    load_catalog_model,
    load_catalog_payload,
)
from kgfoundry.agent_catalog.rbac import AccessController, Role
from kgfoundry.agent_catalog.session import (
    CatalogSession,
    CatalogSessionError,
)
from kgfoundry.agent_catalog.sqlite import load_catalog_from_sqlite, write_sqlite_catalog
from kgfoundry_common.navmap_loader import load_nav_metadata

__all__ = [
    "AccessController",
    "AgentCatalogClient",
    "AgentCatalogClientError",
    "AgentCatalogModel",
    "AuditLogger",
    "CatalogSession",
    "CatalogSessionError",
    "Role",
    "load_catalog_from_sqlite",
    "load_catalog_model",
    "load_catalog_payload",
    "search",
    "write_sqlite_catalog",
]

__navmap__ = load_nav_metadata(__name__, tuple(__all__))
