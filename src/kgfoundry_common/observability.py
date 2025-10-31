"""Prometheus metrics and OpenTelemetry tracing helpers.

This module provides structured observability with Prometheus metrics
and optional OpenTelemetry tracing. All metrics follow naming conventions
and include operation/status tags.

Examples
--------
>>> from kgfoundry_common.observability import get_metrics_registry, record_operation
>>> metrics = get_metrics_registry()
>>> with record_operation(metrics, "search", "success"):
...     # Operation code here
...     pass
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

from kgfoundry_common.navmap_types import NavMap

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "MetricsRegistry",
    "get_metrics_registry",
    "record_operation",
]

__navmap__: Final[NavMap] = {
    "title": "kgfoundry_common.observability",
    "synopsis": "Prometheus metrics and OpenTelemetry tracing helpers",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@kgfoundry-common",
        "stability": "stable",
        "since": "0.1.0",
    },
    "symbols": {
        name: {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        }
        for name in __all__
    },
}

# Prometheus client is optional (may not be installed)
try:
    from prometheus_client import Counter, Gauge, Histogram, Registry

    HAVE_PROMETHEUS = True
except ImportError:
    HAVE_PROMETHEUS = False
    # Stub types for type checking when prometheus_client is unavailable
    if TYPE_CHECKING:

        class Registry:  # type: ignore[no-redef]
            """Stub registry."""

        class Counter:  # type: ignore[no-redef]
            """Stub counter."""

            def inc(self, *args: object, **kwargs: object) -> None:
                """Stub increment."""

        class Gauge:  # type: ignore[no-redef]
            """Stub gauge."""

            def set(self, *args: object, **kwargs: object) -> None:
                """Stub set."""

        class Histogram:  # type: ignore[no-redef]
            """Stub histogram."""

            def observe(self, *args: object, **kwargs: object) -> None:
                """Stub observe."""


class MetricsRegistry:
    """Registry for Prometheus metrics with standard naming conventions.

    This class provides counters, gauges, and histograms following
    kgfoundry naming conventions (kgfoundry_requests_total, etc.).

    Examples
    --------
    >>> metrics = MetricsRegistry()
    >>> metrics.requests_total.labels(operation="search", status="success").inc()
    >>> metrics.request_duration_seconds.labels(operation="search").observe(0.123)
    """

    def __init__(self, registry: Registry | None = None) -> None:
        """Initialize metrics registry.

        Parameters
        ----------
        registry : Registry | None, optional
            Prometheus registry (defaults to default registry).
        """
        if not HAVE_PROMETHEUS:
            # Create stub metrics when prometheus_client is unavailable
            self.requests_total = _StubCounter()
            self.request_errors_total = _StubCounter()
            self.request_duration_seconds = _StubHistogram()
            return

        if registry is None:
            from prometheus_client import REGISTRY  # noqa: PLC0415

            registry = REGISTRY

        self.registry = registry

        # Standard metrics following naming conventions
        self.requests_total = Counter(
            "kgfoundry_requests_total",
            "Total number of requests",
            ["operation", "status"],
            registry=registry,
        )

        self.request_errors_total = Counter(
            "kgfoundry_request_errors_total",
            "Total number of request errors",
            ["operation", "status"],
            registry=registry,
        )

        self.request_duration_seconds = Histogram(
            "kgfoundry_request_duration_seconds",
            "Request duration in seconds",
            ["operation"],
            registry=registry,
        )


class _StubCounter:
    """Stub counter for when prometheus_client is unavailable."""

    def labels(self, **kwargs: object) -> _StubCounter:  # noqa: ARG002
        """Return self for chaining."""
        return self

    def inc(self, value: float = 1.0) -> None:
        """No-op increment."""


class _StubHistogram:
    """Stub histogram for when prometheus_client is unavailable."""

    def labels(self, **kwargs: object) -> _StubHistogram:  # noqa: ARG002
        """Return self for chaining."""
        return self

    def observe(self, value: float) -> None:
        """No-op observe."""


# Global metrics registry (lazy initialization)
_metrics_registry: MetricsRegistry | None = None


def get_metrics_registry() -> MetricsRegistry:
    """Get or create the global metrics registry.

    Returns
    -------
    MetricsRegistry
        Global metrics registry instance.

    Examples
    --------
    >>> metrics = get_metrics_registry()
    >>> metrics.requests_total.labels(operation="search", status="success").inc()
    """
    global _metrics_registry  # noqa: PLW0603
    if _metrics_registry is None:
        _metrics_registry = MetricsRegistry()
    return _metrics_registry


@contextmanager
def record_operation(
    metrics: MetricsRegistry | None = None,
    operation: str = "unknown",
    status: str = "success",
) -> Iterator[None]:
    """Context manager to record operation metrics and duration.

    Parameters
    ----------
    metrics : MetricsRegistry | None, optional
        Metrics registry (defaults to global registry).
    operation : str, optional
        Operation name (default: "unknown").
    status : str, optional
        Operation status (default: "success").

    Yields
    ------
    None
        Context manager yields control to the operation block.

    Examples
    --------
    >>> from kgfoundry_common.observability import record_operation, get_metrics_registry
    >>> metrics = get_metrics_registry()
    >>> with record_operation(metrics, "search", "success"):
    ...     # Perform search operation
    ...     pass
    """
    if metrics is None:
        metrics = get_metrics_registry()

    start_time = time.monotonic()

    try:
        yield
        final_status = status
    except Exception:
        final_status = "error"
        metrics.request_errors_total.labels(operation=operation, status=final_status).inc()
        raise
    finally:
        duration = time.monotonic() - start_time
        metrics.requests_total.labels(operation=operation, status=final_status).inc()
        metrics.request_duration_seconds.labels(operation=operation).observe(duration)
