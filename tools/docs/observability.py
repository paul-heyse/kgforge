"""Observability instrumentation for documentation pipelines.

This module provides Prometheus metrics and structured logging helpers for
documentation generation operations (catalog, graphs, test map, schemas, portal).
Metrics follow Prometheus naming conventions and include operation/status tags.
All metrics are instantiated via the typed helpers defined in
``tools/_shared/observability_facade.md`` so they fall back to no-ops when
``prometheus_client`` is missing (see the "Eliminate Pyrefly Suppressions"
scenario in the code-quality spec).
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING

from tools.shared.prometheus import (
    CollectorRegistry,
    CounterLike,
    HistogramLike,
    build_counter,
    build_histogram,
    get_default_registry,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


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

    catalog_runs_total: CounterLike
    graphs_runs_total: CounterLike
    test_map_runs_total: CounterLike
    schemas_runs_total: CounterLike
    portal_runs_total: CounterLike
    analytics_runs_total: CounterLike
    catalog_duration_seconds: HistogramLike
    graphs_duration_seconds: HistogramLike
    test_map_duration_seconds: HistogramLike
    schemas_duration_seconds: HistogramLike
    portal_duration_seconds: HistogramLike
    analytics_duration_seconds: HistogramLike

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Initialize metrics registry.

        Parameters
        ----------
        registry : CollectorRegistry | None, optional
            Prometheus registry (defaults to default registry).
        """
        resolved_registry = registry if registry is not None else get_default_registry()
        self.registry = resolved_registry

        self.catalog_runs_total = build_counter(
            "docs_catalog_runs_total",
            "Total number of catalog build operations",
            ["status"],
            registry=resolved_registry,
        )

        self.graphs_runs_total = build_counter(
            "docs_graphs_runs_total",
            "Total number of graph build operations",
            ["status"],
            registry=resolved_registry,
        )

        self.test_map_runs_total = build_counter(
            "docs_test_map_runs_total",
            "Total number of test map build operations",
            ["status"],
            registry=resolved_registry,
        )

        self.schemas_runs_total = build_counter(
            "docs_schemas_runs_total",
            "Total number of schema export operations",
            ["status"],
            registry=resolved_registry,
        )

        self.portal_runs_total = build_counter(
            "docs_portal_runs_total",
            "Total number of portal render operations",
            ["status"],
            registry=resolved_registry,
        )

        self.analytics_runs_total = build_counter(
            "docs_analytics_runs_total",
            "Total number of analytics build operations",
            ["status"],
            registry=resolved_registry,
        )

        self.catalog_duration_seconds = build_histogram(
            "docs_catalog_duration_seconds",
            "Duration of catalog build operations in seconds",
            ["status"],
            registry=resolved_registry,
        )

        self.graphs_duration_seconds = build_histogram(
            "docs_graphs_duration_seconds",
            "Duration of graph build operations in seconds",
            ["status"],
            registry=resolved_registry,
        )

        self.test_map_duration_seconds = build_histogram(
            "docs_test_map_duration_seconds",
            "Duration of test map build operations in seconds",
            ["status"],
            registry=resolved_registry,
        )

        self.schemas_duration_seconds = build_histogram(
            "docs_schemas_duration_seconds",
            "Duration of schema export operations in seconds",
            ["status"],
            registry=resolved_registry,
        )

        self.portal_duration_seconds = build_histogram(
            "docs_portal_duration_seconds",
            "Duration of portal render operations in seconds",
            ["status"],
            registry=resolved_registry,
        )

        self.analytics_duration_seconds = build_histogram(
            "docs_analytics_duration_seconds",
            "Duration of analytics build operations in seconds",
            ["status"],
            registry=resolved_registry,
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
    global _METRICS_REGISTRY
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
