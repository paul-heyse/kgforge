"""Public interface for the agent catalog client and utilities."""

from kgfoundry.agent_catalog import search
from kgfoundry.agent_catalog.client import AgentCatalogClient, AgentCatalogClientError
from kgfoundry.agent_catalog.models import (
    AgentCatalogModel,
    load_catalog_model,
    load_catalog_payload,
)

__all__ = [
    "AgentCatalogClient",
    "AgentCatalogClientError",
    "AgentCatalogModel",
    "load_catalog_model",
    "load_catalog_payload",
    "search",
]
