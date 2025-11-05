"""Expose the public search client entry point and metadata helpers."""

from __future__ import annotations

from kgfoundry_common.navmap_loader import load_nav_metadata
from kgfoundry_common.navmap_types import NavMap as _NavMap
from search_client.client import KGFoundryClient as _KGFoundryClient

NavMap = _NavMap

__all__ = [
    "KGFoundryClient",
    "NavMap",
]
__navmap__ = load_nav_metadata(__name__, tuple(__all__))


# [nav:section public-api]
# [nav:anchor KGFoundryClient]
KGFoundryClient = _KGFoundryClient
