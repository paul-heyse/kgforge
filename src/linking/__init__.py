"""Expose entity linking utilities and navigation metadata."""

from __future__ import annotations

from kgfoundry_common.navmap_loader import load_nav_metadata
from kgfoundry_common.navmap_types import NavMap as _NavMap
from linking import calibration, linker

NavMap = _NavMap

__all__ = [
    "NavMap",
    "calibration",
    "linker",
]
__navmap__ = load_nav_metadata(__name__, tuple(__all__))


# [nav:section public-api]
# [nav:anchor calibration]
# [nav:anchor linker]
