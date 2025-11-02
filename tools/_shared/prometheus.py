"""Re-export typed Prometheus helpers for tooling packages.

Tooling code consumes the shared helpers from :mod:`kgfoundry_common.prometheus`
so we only maintain a single source of truth for fallbacks and type
annotations.
"""

from __future__ import annotations

from kgfoundry_common.prometheus import (
    HAVE_PROMETHEUS,
    CollectorRegistry,
    CounterLike,
    GaugeLike,
    HistogramLike,
    build_counter,
    build_gauge,
    build_histogram,
    get_default_registry,
)

__all__ = [
    "HAVE_PROMETHEUS",
    "CollectorRegistry",
    "CounterLike",
    "GaugeLike",
    "HistogramLike",
    "build_counter",
    "build_gauge",
    "build_histogram",
    "get_default_registry",
]
