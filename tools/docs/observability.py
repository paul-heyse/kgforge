"""Observability instrumentation for documentation pipelines.

This module provides Prometheus metrics and structured logging helpers for
documentation generation operations (catalog, graphs, test map, schemas, portal).
Metrics follow Prometheus naming conventions and include operation/status tags.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

    from prometheus_client import Counter as _CounterType
    from prometheus_client import Histogram as _HistogramType
    from prometheus_client.registry import CollectorRegistry as _RegistryType

    CounterFactory = Callable[..., _CounterType]
    HistogramFactory = Callable[..., _HistogramType]
    # CollectorRegistry is a class, but we use instances
    type CollectorRegistryType = _RegistryType
else:
    from collections.abc import Iterator

    CounterFactory = Callable[..., object]
    HistogramFactory = Callable[..., object]
    # Runtime fallback: use object as the base type
    type CollectorRegistryType = object

try:
    from prometheus_client import Counter as _PromCounter
    from prometheus_client import Histogram as _PromHistogram
    from prometheus_client.registry import CollectorRegistry

    HAVE_PROMETHEUS = True
except ImportError:
    HAVE_PROMETHEUS = False
    if TYPE_CHECKING:

        class CollectorRegistry:  # type: ignore[no-redef]
            """Stub registry."""

    class _NoopMetric:
        """No-op metric stub."""

        def labels(self, **kwargs: object) -> _NoopMetric:  # noqa: ARG002
            """Return self for chaining."""
            return self

        def inc(self, value: float = 1.0) -> None:
            """No-op increment."""

        def observe(self, value: float = 1.0) -> None:
            """No-op observe."""

    def _make_noop_metric(*args: object, **kwargs: object) -> _NoopMetric:  # noqa: ARG001
        """Create no-op metric instance."""
        return _NoopMetric()

    Counter = cast(CounterFactory, _make_noop_metric)
    Histogram = cast(HistogramFactory, _make_noop_metric)

# Assign Counter and Histogram from the real or stub implementations
if HAVE_PROMETHEUS:
    # Type narrowing: when HAVE_PROMETHEUS is True, _PromCounter and _PromHistogram are defined
    # pyrefly: ignore[unbound-name] - variables are bound in the try block when HAVE_PROMETHEUS is True
    Counter = cast(CounterFactory, _PromCounter)  # pyrefly: ignore[unbound-name]
    Histogram = cast(HistogramFactory, _PromHistogram)  # pyrefly: ignore[unbound-name]

# Type alias for registry (instance type, not class type)
type Registry = CollectorRegistryType


class CounterLike(Protocol):
    """Protocol for counter-like metrics."""

    def labels(self, **kwargs: object) -> CounterLike:
        """Return labeled counter instance."""
        ...

    def inc(self, value: float = 1.0) -> None:
        """Increment counter."""
        ...


class HistogramLike(Protocol):
    """Protocol for histogram-like metrics."""

    def labels(self, **kwargs: object) -> HistogramLike:
        """Return labeled histogram instance."""
        ...

    def observe(self, value: float) -> None:
        """Observe a value."""
        ...


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

    def observe(self, value: float = 1.0) -> None:
        """No-op observe."""


class DocumentationMetrics:
    """Metrics registry for documentation pipelines following Prometheus conventions.

    Metrics follow the naming pattern: ``docs_<operation>_runs_total`` for
    counters and ``docs_<operation>_duration_seconds`` for histograms.

    Examples
    --------
    >>> metrics = DocumentationMetrics()
    >>> metrics.catalog_runs_total.labels(status="success").inc()
    >>> metrics.graphs_duration_seconds.labels(status="success").observe(0.123)
    """

    catalog_runs_total: CounterLike | _StubCounter
    graphs_runs_total: CounterLike | _StubCounter
    test_map_runs_total: CounterLike | _StubCounter
    schemas_runs_total: CounterLike | _StubCounter
    portal_runs_total: CounterLike | _StubCounter
    analytics_runs_total: CounterLike | _StubCounter
    catalog_duration_seconds: HistogramLike | _StubHistogram
    graphs_duration_seconds: HistogramLike | _StubHistogram
    test_map_duration_seconds: HistogramLike | _StubHistogram
    schemas_duration_seconds: HistogramLike | _StubHistogram
    portal_duration_seconds: HistogramLike | _StubHistogram
    analytics_duration_seconds: HistogramLike | _StubHistogram

    def __init__(self, registry: CollectorRegistryType | None = None) -> None:
        """Initialize metrics registry.

        Parameters
        ----------
        registry : Registry | None, optional
            Prometheus registry (defaults to default registry).
        """
        if not HAVE_PROMETHEUS:
            self.catalog_runs_total = _StubCounter()
            self.graphs_runs_total = _StubCounter()
            self.test_map_runs_total = _StubCounter()
            self.schemas_runs_total = _StubCounter()
            self.portal_runs_total = _StubCounter()
            self.analytics_runs_total = _StubCounter()
            self.catalog_duration_seconds = _StubHistogram()
            self.graphs_duration_seconds = _StubHistogram()
            self.test_map_duration_seconds = _StubHistogram()
            self.schemas_duration_seconds = _StubHistogram()
            self.portal_duration_seconds = _StubHistogram()
            self.analytics_duration_seconds = _StubHistogram()
            return

        if registry is None:
            from prometheus_client import REGISTRY  # noqa: PLC0415

            registry = REGISTRY

        # registry is an instance, not a type
        self.registry: CollectorRegistryType = registry  # type: ignore[assignment]

        # Counter and Histogram are correctly typed at module level via cast()
        # pyrefly sees the correct types because we don't use TYPE_CHECKING stubs
        self.catalog_runs_total = Counter(  # type: ignore[assignment]
            "docs_catalog_runs_total",
            "Total number of catalog build operations",
            ["status"],
            registry=registry,
        )

        self.graphs_runs_total = Counter(  # type: ignore[assignment]
            "docs_graphs_runs_total",
            "Total number of graph build operations",
            ["status"],
            registry=registry,
        )

        self.test_map_runs_total = Counter(  # type: ignore[assignment]
            "docs_test_map_runs_total",
            "Total number of test map build operations",
            ["status"],
            registry=registry,
        )

        self.schemas_runs_total = Counter(  # type: ignore[assignment]
            "docs_schemas_runs_total",
            "Total number of schema export operations",
            ["status"],
            registry=registry,
        )

        self.portal_runs_total = Counter(  # type: ignore[assignment]
            "docs_portal_runs_total",
            "Total number of portal render operations",
            ["status"],
            registry=registry,
        )

        self.analytics_runs_total = Counter(  # type: ignore[assignment]
            "docs_analytics_runs_total",
            "Total number of analytics build operations",
            ["status"],
            registry=registry,
        )

        self.catalog_duration_seconds = Histogram(  # type: ignore[assignment]
            "docs_catalog_duration_seconds",
            "Duration of catalog build operations in seconds",
            ["status"],
            registry=registry,
        )

        self.graphs_duration_seconds = Histogram(  # type: ignore[assignment]
            "docs_graphs_duration_seconds",
            "Duration of graph build operations in seconds",
            ["status"],
            registry=registry,
        )

        self.test_map_duration_seconds = Histogram(  # type: ignore[assignment]
            "docs_test_map_duration_seconds",
            "Duration of test map build operations in seconds",
            ["status"],
            registry=registry,
        )

        self.schemas_duration_seconds = Histogram(  # type: ignore[assignment]
            "docs_schemas_duration_seconds",
            "Duration of schema export operations in seconds",
            ["status"],
            registry=registry,
        )

        self.portal_duration_seconds = Histogram(  # type: ignore[assignment]
            "docs_portal_duration_seconds",
            "Duration of portal render operations in seconds",
            ["status"],
            registry=registry,
        )

        self.analytics_duration_seconds = Histogram(  # type: ignore[assignment]
            "docs_analytics_duration_seconds",
            "Duration of analytics build operations in seconds",
            ["status"],
            registry=registry,
        )


_METRICS_REGISTRY: DocumentationMetrics | None = None


def get_metrics_registry() -> DocumentationMetrics:
    """Get or create the global metrics registry.

    Returns
    -------
    DocumentationMetrics
        Global metrics registry instance.

    Examples
    --------
    >>> metrics = get_metrics_registry()
    >>> metrics.catalog_runs_total.labels(status="success").inc()
    """
    global _METRICS_REGISTRY  # noqa: PLW0603
    if _METRICS_REGISTRY is None:
        _METRICS_REGISTRY = DocumentationMetrics()
    return _METRICS_REGISTRY


def get_correlation_id() -> str:
    """Generate a correlation ID for tracing operations across boundaries.

    Returns
    -------
    str
        Correlation ID in the format ``urn:docs:correlation:<uuid>``.

    Examples
    --------
    >>> corr_id = get_correlation_id()
    >>> assert corr_id.startswith("urn:docs:correlation:")
    """
    return f"urn:docs:correlation:{uuid.uuid4().hex}"


@contextmanager
def record_operation_metrics(
    operation: str,
    correlation_id: str | None = None,
    *,
    metrics: DocumentationMetrics | None = None,
    status: str = "success",
) -> Iterator[None]:
    """Context manager to record operation metrics and duration.

    Parameters
    ----------
    operation : str
        Operation name (e.g., "catalog", "graphs", "test_map", "schemas", "portal", "analytics").
    correlation_id : str | None, optional
        Correlation ID for tracing (default: auto-generated).
    metrics : DocumentationMetrics | None, optional
        Metrics registry (defaults to global registry).
    status : str, optional
        Initial status (default: "success"); updated to "error" on exception.

    Yields
    ------
    None
        Context manager yields control to the operation block.

    Examples
    --------
    >>> from tools.docs.observability import (
    ...     record_operation_metrics,
    ...     get_correlation_id,
    ... )
    >>> corr_id = get_correlation_id()
    >>> with record_operation_metrics("catalog", corr_id):
    ...     # Perform catalog build operation
    ...     pass
    """
    if metrics is None:
        metrics = get_metrics_registry()

    if correlation_id is None:
        correlation_id = get_correlation_id()

    start_time = time.monotonic()
    final_status = status

    try:
        yield
    except Exception:
        final_status = "error"
        raise
    finally:
        duration = time.monotonic() - start_time

        if operation == "catalog":
            metrics.catalog_runs_total.labels(status=final_status).inc()
            metrics.catalog_duration_seconds.labels(status=final_status).observe(duration)
        elif operation == "graphs":
            metrics.graphs_runs_total.labels(status=final_status).inc()
            metrics.graphs_duration_seconds.labels(status=final_status).observe(duration)
        elif operation == "test_map":
            metrics.test_map_runs_total.labels(status=final_status).inc()
            metrics.test_map_duration_seconds.labels(status=final_status).observe(duration)
        elif operation == "schemas":
            metrics.schemas_runs_total.labels(status=final_status).inc()
            metrics.schemas_duration_seconds.labels(status=final_status).observe(duration)
        elif operation == "portal":
            metrics.portal_runs_total.labels(status=final_status).inc()
            metrics.portal_duration_seconds.labels(status=final_status).observe(duration)
        elif operation == "analytics":
            metrics.analytics_runs_total.labels(status=final_status).inc()
            metrics.analytics_duration_seconds.labels(status=final_status).observe(duration)


__all__ = [
    "DocumentationMetrics",
    "get_correlation_id",
    "get_metrics_registry",
    "record_operation_metrics",
]
