"""Public wrapper for :mod:`tools._shared.prometheus`."""

from __future__ import annotations

from tools._shared.prometheus import (
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

__all__: tuple[str, ...] = (
    "HAVE_PROMETHEUS",
    "CollectorRegistry",
    "CounterLike",
    "GaugeLike",
    "HistogramLike",
    "build_counter",
    "build_gauge",
    "build_histogram",
    "get_default_registry",
)
