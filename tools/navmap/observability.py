"""Observability instrumentation for navmap utilities.

This module provides Prometheus metrics and structured logging helpers for
navmap operations (build, check, repair, migrate). Metrics follow Prometheus
naming conventions and include operation/status tags. The metrics originate from
the typed facade documented in ``tools/_shared/observability_facade.md`` and
comply with the "Eliminate Pyrefly Suppressions" spec so they become no-ops if
``prometheus_client`` is not installed.
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


class NavmapMetrics:
    """Metrics registry for navmap utilities following Prometheus conventions.

    Metrics follow the naming pattern: ``navmap_<operation>_total`` for
    counters and ``navmap_<operation>_duration_seconds`` for histograms.

    Examples
    --------
    >>> metrics = NavmapMetrics()
    >>> metrics.build_runs_total.labels(status="success").inc()
    >>> metrics.repair_duration_seconds.labels(status="success").observe(0.123)
    """

    build_runs_total: CounterLike
    check_runs_total: CounterLike
    repair_runs_total: CounterLike
    migrate_runs_total: CounterLike
    build_duration_seconds: HistogramLike
    check_duration_seconds: HistogramLike
    repair_duration_seconds: HistogramLike
    migrate_duration_seconds: HistogramLike

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Initialize metrics registry.

        Parameters
        ----------
        registry : CollectorRegistry | None, optional
            Prometheus registry (defaults to default registry).
        """
        resolved_registry = registry if registry is not None else get_default_registry()
        self.registry = resolved_registry

        self.build_runs_total = build_counter(
            "navmap_build_runs_total",
            "Total number of navmap build operations",
            ["status"],
            registry=resolved_registry,
        )

        self.check_runs_total = build_counter(
            "navmap_check_runs_total",
            "Total number of navmap check operations",
            ["status"],
            registry=resolved_registry,
        )

        self.repair_runs_total = build_counter(
            "navmap_repair_runs_total",
            "Total number of navmap repair operations",
            ["status"],
            registry=resolved_registry,
        )

        self.migrate_runs_total = build_counter(
            "navmap_migrate_runs_total",
            "Total number of navmap migrate operations",
            ["status"],
            registry=resolved_registry,
        )

        self.build_duration_seconds = build_histogram(
            "navmap_build_duration_seconds",
            "Duration of navmap build operations in seconds",
            ["status"],
            registry=resolved_registry,
        )

        self.check_duration_seconds = build_histogram(
            "navmap_check_duration_seconds",
            "Duration of navmap check operations in seconds",
            ["status"],
            registry=resolved_registry,
        )

        self.repair_duration_seconds = build_histogram(
            "navmap_repair_duration_seconds",
            "Duration of navmap repair operations in seconds",
            ["status"],
            registry=resolved_registry,
        )

        self.migrate_duration_seconds = build_histogram(
            "navmap_migrate_duration_seconds",
            "Duration of navmap migrate operations in seconds",
            ["status"],
            registry=resolved_registry,
        )


_METRICS_REGISTRY: NavmapMetrics | None = None


def get_metrics_registry() -> NavmapMetrics:
    """Get or create the global metrics registry.

    Returns
    -------
    NavmapMetrics
        Global metrics registry instance.

    Examples
    --------
    >>> metrics = get_metrics_registry()
    >>> metrics.build_runs_total.labels(status="success").inc()
    """
    global _METRICS_REGISTRY
    if _METRICS_REGISTRY is None:
        _METRICS_REGISTRY = NavmapMetrics()
    return _METRICS_REGISTRY


def get_correlation_id() -> str:
    """Generate a correlation ID for tracing operations across boundaries.

    Returns
    -------
    str
        Correlation ID in the format ``urn:navmap:correlation:<uuid>``.

    Examples
    --------
    >>> cid = get_correlation_id()
    >>> assert cid.startswith("urn:navmap:correlation:")
    """
    return f"urn:navmap:correlation:{uuid.uuid4().hex}"


@contextmanager
def record_operation_metrics(
    operation: str,
    status: str = "success",
    correlation_id: str | None = None,
) -> Iterator[None]:
    """Context manager to record operation metrics.

    Parameters
    ----------
    operation : str
        Operation name (e.g., "build", "check", "repair", "migrate").
    status : str, optional
        Status label ("success" or "error"), by default "success".
    correlation_id : str | None, optional
        Correlation ID for tracing, by default None (auto-generated).

    Yields
    ------
    None
        Context manager yields nothing.

    Examples
    --------
    >>> with record_operation_metrics("build", status="success"):
    ...     # Build operation
    ...     pass
    """
    if correlation_id is None:
        correlation_id = get_correlation_id()

    metrics = get_metrics_registry()
    start_time = time.monotonic()

    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.monotonic() - start_time

        if operation == "build":
            metrics.build_runs_total.labels(status=status).inc()
            metrics.build_duration_seconds.labels(status=status).observe(duration)
        elif operation == "check":
            metrics.check_runs_total.labels(status=status).inc()
            metrics.check_duration_seconds.labels(status=status).observe(duration)
        elif operation == "repair":
            metrics.repair_runs_total.labels(status=status).inc()
            metrics.repair_duration_seconds.labels(status=status).observe(duration)
        elif operation == "migrate":
            metrics.migrate_runs_total.labels(status=status).inc()
            metrics.migrate_duration_seconds.labels(status=status).observe(duration)
