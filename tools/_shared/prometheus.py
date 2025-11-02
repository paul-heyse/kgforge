"""Typed wrappers around Prometheus metrics with graceful fallbacks.

This module centralises the logic for constructing Prometheus counters and
histograms while keeping type checkers happy when the optional dependency is not
installed.  Callers import the :func:`build_counter` and :func:`build_histogram`
helpers to obtain objects implementing the lightweight :class:`CounterLike` and
:class:`HistogramLike` protocols.

The helpers degrade to no-op implementations so instrumentation does not need to
sprout conditional imports throughout the codebase.  They also accept an
optional :class:`~prometheus_client.registry.CollectorRegistry` instance so unit
tests can inject their own registries.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, cast

try:  # pragma: no branch - import guarded for optional dependency
    from prometheus_client import Counter as _Counter
    from prometheus_client import Histogram as _Histogram
    from prometheus_client.registry import CollectorRegistry

    HAVE_PROMETHEUS = True
except ImportError:  # pragma: no cover - Prometheus not installed in minimal envs
    CollectorRegistry = object  # type: ignore[misc, assignment]
    HAVE_PROMETHEUS = False


class CounterLike(Protocol):
    """Subset of Prometheus counter behaviour relied on by tooling."""

    def labels(self, **kwargs: object) -> CounterLike: ...

    def inc(self, value: float = 1.0) -> None: ...


class HistogramLike(Protocol):
    """Subset of Prometheus histogram behaviour relied on by tooling."""

    def labels(self, **kwargs: object) -> HistogramLike: ...

    def observe(self, value: float) -> None: ...


class _NoopCounter:
    """Counter stub used when Prometheus is unavailable."""

    def labels(self, **kwargs: object) -> CounterLike:  # noqa: ARG002 - signature parity
        return self

    def inc(self, value: float = 1.0) -> None:  # noqa: ARG002 - signature parity
        return None


class _NoopHistogram:
    """Histogram stub used when Prometheus is unavailable."""

    def labels(self, **kwargs: object) -> HistogramLike:  # noqa: ARG002
        return self

    def observe(self, value: float) -> None:  # noqa: ARG002
        return None


def _as_counter_callable() -> Callable[..., CounterLike] | None:
    if not HAVE_PROMETHEUS:  # pragma: no cover - early exit when dependency missing
        return None
    counter_obj: object | None = _Counter  # type: ignore[name-defined] - guarded above
    if counter_obj is None or not callable(counter_obj):  # pragma: no cover - defensive guard
        return None
    return cast(Callable[..., CounterLike], counter_obj)


def _as_histogram_callable() -> Callable[..., HistogramLike] | None:
    if not HAVE_PROMETHEUS:  # pragma: no cover - early exit when dependency missing
        return None
    histogram_obj: object | None = _Histogram  # type: ignore[name-defined]
    if histogram_obj is None or not callable(histogram_obj):  # pragma: no cover
        return None
    return cast(Callable[..., HistogramLike], histogram_obj)


_COUNTER_CONSTRUCTOR = _as_counter_callable()
_HISTOGRAM_CONSTRUCTOR = _as_histogram_callable()


def build_counter(
    name: str,
    documentation: str,
    labelnames: Sequence[str] | None = None,
    *,
    registry: CollectorRegistry | None = None,
) -> CounterLike:
    """Return a counter metric or a no-op stub if Prometheus is unavailable."""
    constructor = _COUNTER_CONSTRUCTOR
    if constructor is None:
        return _NoopCounter()
    return constructor(name, documentation, labelnames or (), registry=registry)


def build_histogram(
    name: str,
    documentation: str,
    labelnames: Sequence[str] | None = None,
    *,
    buckets: Sequence[float] | None = None,
    registry: CollectorRegistry | None = None,
) -> HistogramLike:
    """Return a histogram metric or a no-op stub if Prometheus is unavailable."""
    constructor = _HISTOGRAM_CONSTRUCTOR
    if constructor is None:
        return _NoopHistogram()
    return constructor(
        name,
        documentation,
        labelnames or (),
        buckets=buckets,
        registry=registry,
    )


__all__ = [
    "CollectorRegistry",
    "CounterLike",
    "HistogramLike",
    "build_counter",
    "build_histogram",
]
