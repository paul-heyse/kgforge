"""Observability instrumentation for the docstring builder.

This module provides Prometheus metrics, structured logging helpers, and optional
OpenTelemetry tracing for the docstring builder pipeline. Metrics follow Prometheus
naming conventions and include operation/status tags.

Examples
--------
>>> from tools.docstring_builder.observability import (
...     get_metrics_registry,
...     record_operation_metrics,
...     get_correlation_id,
... )
>>> metrics = get_metrics_registry()
>>> correlation_id = get_correlation_id()
>>> with record_operation_metrics("harvest", correlation_id):
...     # Harvest operation
...     pass
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

try:
    from prometheus_client import Counter, Histogram, Registry

    HAVE_PROMETHEUS = True
except ImportError:
    HAVE_PROMETHEUS = False
    if TYPE_CHECKING:

        class Registry:  # type: ignore[no-redef]
            """Stub registry."""

        class Counter:  # type: ignore[no-redef]
            """Stub counter."""

            def labels(self, *args: object, **kwargs: object) -> _StubCounter:
                """Stub labels."""

        class Histogram:  # type: ignore[no-redef]
            """Stub histogram."""

            def labels(self, *args: object, **kwargs: object) -> _StubHistogram:
                """Stub labels."""

        class _StubCounter:
            """Stub counter."""

            def inc(self, value: float = 1.0) -> None:
                """No-op increment."""

        class _StubHistogram:
            """Stub histogram."""

            def observe(self, value: float) -> None:
                """No-op observe."""


class DocstringBuilderMetrics:
    """Metrics registry for the docstring builder following Prometheus conventions.

    Metrics follow the naming pattern: ``docbuilder_<operation>_<unit>_total`` for
    counters and ``docbuilder_<operation>_duration_seconds`` for histograms.

    Examples
    --------
    >>> metrics = DocstringBuilderMetrics()
    >>> metrics.runs_total.labels(status="success").inc()
    >>> metrics.harvest_duration_seconds.labels(status="success").observe(0.123)
    """

    def __init__(self, registry: Registry | None = None) -> None:
        """Initialize metrics registry.

        Parameters
        ----------
        registry : Registry | None, optional
            Prometheus registry (defaults to default registry).
        """
        if not HAVE_PROMETHEUS:
            self.runs_total = _StubCounter()
            self.plugin_failures_total = _StubCounter()
            self.harvest_duration_seconds = _StubHistogram()
            self.policy_duration_seconds = _StubHistogram()
            self.render_duration_seconds = _StubHistogram()
            self.cli_duration_seconds = _StubHistogram()
            return

        if registry is None:
            from prometheus_client import REGISTRY  # noqa: PLC0415

            registry = REGISTRY

        self.registry = registry

        self.runs_total = Counter(
            "docbuilder_runs_total",
            "Total number of docstring builder runs",
            ["status"],
            registry=registry,
        )

        self.plugin_failures_total = Counter(
            "docbuilder_plugin_failures_total",
            "Total number of plugin execution failures",
            ["plugin_name", "error_type"],
            registry=registry,
        )

        self.harvest_duration_seconds = Histogram(
            "docbuilder_harvest_duration_seconds",
            "Duration of harvest operations in seconds",
            ["status"],
            registry=registry,
        )

        self.policy_duration_seconds = Histogram(
            "docbuilder_policy_duration_seconds",
            "Duration of policy engine operations in seconds",
            ["status"],
            registry=registry,
        )

        self.render_duration_seconds = Histogram(
            "docbuilder_render_duration_seconds",
            "Duration of rendering operations in seconds",
            ["status"],
            registry=registry,
        )

        self.cli_duration_seconds = Histogram(
            "docbuilder_cli_duration_seconds",
            "Duration of CLI operations in seconds",
            ["command", "status"],
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


_METRICS_REGISTRY: DocstringBuilderMetrics | None = None


def get_metrics_registry() -> DocstringBuilderMetrics:
    """Get or create the global metrics registry.

    Returns
    -------
    DocstringBuilderMetrics
        Global metrics registry instance.

    Examples
    --------
    >>> metrics = get_metrics_registry()
    >>> metrics.runs_total.labels(status="success").inc()
    """
    global _METRICS_REGISTRY  # noqa: PLW0603
    if _METRICS_REGISTRY is None:
        _METRICS_REGISTRY = DocstringBuilderMetrics()
    return _METRICS_REGISTRY


def get_correlation_id() -> str:
    """Generate a correlation ID for tracing operations across boundaries.

    Returns
    -------
    str
        Correlation ID in the format ``urn:docbuilder:correlation:<uuid>``.

    Examples
    --------
    >>> corr_id = get_correlation_id()
    >>> assert corr_id.startswith("urn:docbuilder:correlation:")
    """
    return f"urn:docbuilder:correlation:{uuid.uuid4().hex}"


@contextmanager
def record_operation_metrics(
    operation: str,
    correlation_id: str | None = None,
    *,
    metrics: DocstringBuilderMetrics | None = None,
    status: str = "success",
) -> Iterator[None]:
    """Context manager to record operation metrics and duration.

    Parameters
    ----------
    operation : str
        Operation name (e.g., "harvest", "policy", "render", "cli").
    correlation_id : str | None, optional
        Correlation ID for tracing (default: auto-generated).
    metrics : DocstringBuilderMetrics | None, optional
        Metrics registry (defaults to global registry).
    status : str, optional
        Initial status (default: "success"); updated to "error" on exception.

    Yields
    ------
    None
        Context manager yields control to the operation block.

    Examples
    --------
    >>> from tools.docstring_builder.observability import (
    ...     record_operation_metrics,
    ...     get_correlation_id,
    ... )
    >>> corr_id = get_correlation_id()
    >>> with record_operation_metrics("harvest", corr_id):
    ...     # Perform harvest operation
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

        if operation == "harvest":
            metrics.harvest_duration_seconds.labels(status=final_status).observe(duration)
        elif operation == "policy":
            metrics.policy_duration_seconds.labels(status=final_status).observe(duration)
        elif operation == "render":
            metrics.render_duration_seconds.labels(status=final_status).observe(duration)
        elif operation == "cli":
            # CLI status is determined by the command, not the operation
            # This is a simplified version; CLI should pass command explicitly
            metrics.cli_duration_seconds.labels(command="unknown", status=final_status).observe(
                duration
            )


__all__ = [
    "DocstringBuilderMetrics",
    "get_correlation_id",
    "get_metrics_registry",
    "record_operation_metrics",
]
