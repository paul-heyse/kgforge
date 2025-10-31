"""Public interface for the agent catalog client and utilities."""

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
    ProblemDetails,
)
from kgfoundry.agent_catalog.sqlite import load_catalog_from_sqlite, write_sqlite_catalog

__all__ = [
    "AccessController",
    "AgentCatalogClient",
    "AgentCatalogClientError",
    "AgentCatalogModel",
    "AuditLogger",
    "CatalogSession",
    "CatalogSessionError",
    "ProblemDetails",
    "Role",
    "load_catalog_from_sqlite",
    "load_catalog_model",
    "load_catalog_payload",
    "search",
    "write_sqlite_catalog",
]
