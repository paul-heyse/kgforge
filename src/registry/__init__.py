"""Expose registry persistence helpers via a cohesive namespace."""

from __future__ import annotations

from kgfoundry_common.navmap_loader import load_nav_metadata
from kgfoundry_common.navmap_types import NavMap as _NavMap
from registry import api, duckdb_helpers, duckdb_registry, helper, migrate

NavMap = _NavMap

__all__ = [
    "NavMap",
    "api",
    "duckdb_helpers",
    "duckdb_registry",
    "helper",
    "migrate",
]
__navmap__ = load_nav_metadata(__name__, tuple(__all__))


# [nav:section public-api]
# [nav:anchor api]
# [nav:anchor duckdb_helpers]
# [nav:anchor duckdb_registry]
# [nav:anchor helper]
# [nav:anchor migrate]
