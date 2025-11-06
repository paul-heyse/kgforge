"""Expose docling_kg pipelines and navigation metadata for documentation ingestion."""

from __future__ import annotations

from kgfoundry_common.navmap_loader import load_nav_metadata
from kgfoundry_common.navmap_types import NavMap as _NavMap

from . import canonicalizer, hybrid, vlm

NavMap = _NavMap

__all__ = [
    "NavMap",
    "canonicalizer",
    "hybrid",
    "vlm",
]
__navmap__ = load_nav_metadata(__name__, tuple(__all__))


# [nav:section public-api]
# [nav:anchor canonicalizer]
# [nav:anchor hybrid]
# [nav:anchor vlm]
