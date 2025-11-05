"""Group knowledge-graph builder adapters under a single namespace."""

from __future__ import annotations

from kg_builder import mock_kg, neo4j_store
from kgfoundry_common.navmap_loader import load_nav_metadata
from kgfoundry_common.navmap_types import NavMap as _NavMap

NavMap = _NavMap

__all__ = [
    "NavMap",
    "mock_kg",
    "neo4j_store",
]
__navmap__ = load_nav_metadata(__name__, tuple(__all__))


# [nav:section public-api]
# [nav:anchor mock_kg]
# [nav:anchor neo4j_store]
