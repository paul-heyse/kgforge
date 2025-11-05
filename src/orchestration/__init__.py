"""Expose orchestration flows and helpers for unified access."""

from __future__ import annotations

from kgfoundry_common.navmap_loader import load_nav_metadata
from kgfoundry_common.navmap_types import NavMap as _NavMap
from orchestration import cli, fixture_flow, flows

NavMap = _NavMap

__all__ = [
    "NavMap",
    "cli",
    "fixture_flow",
    "flows",
]
__navmap__ = load_nav_metadata(__name__, tuple(__all__))


# [nav:section public-api]
# [nav:anchor cli]
# [nav:anchor fixture_flow]
# [nav:anchor flows]
