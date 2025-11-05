"""Expose download package utilities and nav metadata."""

from __future__ import annotations

from download import cli, harvester
from kgfoundry_common.navmap_loader import load_nav_metadata
from kgfoundry_common.navmap_types import NavMap as _NavMap

NavMap = _NavMap

__all__ = [
    "NavMap",
    "cli",
    "harvester",
]
__navmap__ = load_nav_metadata(__name__, tuple(__all__))

# [nav:section public-api]
# [nav:anchor cli]
# [nav:anchor harvester]
