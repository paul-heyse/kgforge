"""Expose observability helpers and associated navigation metadata."""

from __future__ import annotations

from kgfoundry_common.navmap_loader import load_nav_metadata
from kgfoundry_common.navmap_types import NavMap as _NavMap
from observability import metrics

NavMap = _NavMap

__all__ = [
    "NavMap",
    "metrics",
]
__navmap__ = load_nav_metadata(__name__, tuple(__all__))


# [nav:section public-api]
# [nav:anchor metrics]
