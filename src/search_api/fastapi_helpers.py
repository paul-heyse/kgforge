"""Expose the typed FastAPI helpers surfaced by the search API layer."""

from __future__ import annotations

from kgfoundry_common.fastapi_helpers import (
    DEFAULT_TIMEOUT_SECONDS,
    typed_dependency,
    typed_exception_handler,
    typed_middleware,
)
from kgfoundry_common.navmap_loader import load_nav_metadata

__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "typed_dependency",
    "typed_exception_handler",
    "typed_middleware",
]
__navmap__ = load_nav_metadata(__name__, tuple(__all__))

# [nav:section public-api]
