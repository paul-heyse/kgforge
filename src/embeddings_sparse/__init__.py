"""Expose sparse embedding integrations under a unified namespace."""

from __future__ import annotations

from embeddings_sparse import base, bm25, splade
from kgfoundry_common.navmap_loader import load_nav_metadata
from kgfoundry_common.navmap_types import NavMap as _NavMap

NavMap = _NavMap

__all__ = [
    "NavMap",
    "base",
    "bm25",
    "splade",
]
__navmap__ = load_nav_metadata(__name__, tuple(__all__))


# [nav:section public-api]
# [nav:anchor base]
# [nav:anchor bm25]
# [nav:anchor splade]
