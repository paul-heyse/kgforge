"""Expose FAISS-based vector store integrations and metadata."""

from __future__ import annotations

from kgfoundry_common.navmap_loader import load_nav_metadata
from kgfoundry_common.navmap_types import NavMap as _NavMap
from vectorstore_faiss import gpu

NavMap = _NavMap

__all__ = [
    "NavMap",
    "gpu",
]
__navmap__ = load_nav_metadata(__name__, tuple(__all__))


# [nav:section public-api]
# [nav:anchor gpu]
