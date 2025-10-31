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


class MetricLabels(Protocol):
    """Protocol for metric objects that support labels()."""

    def labels(self, **kwargs: object) -> MetricLabels:
        """Return labeled metric instance."""
        ...


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

    runs_total: CounterLike | _StubCounter
    plugin_failures_total: CounterLike | _StubCounter
    harvest_duration_seconds: HistogramLike | _StubHistogram
    policy_duration_seconds: HistogramLike | _StubHistogram
    render_duration_seconds: HistogramLike | _StubHistogram
    cli_duration_seconds: HistogramLike | _StubHistogram

    def __init__(self, registry: CollectorRegistryType | None = None) -> None:
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

        # pyrefly: ignore[bad-argument-type] - registry type is narrowed by HAVE_PROMETHEUS
        self.runs_total = Counter(  # type: ignore[assignment]
            "docbuilder_runs_total",
            "Total number of docstring builder runs",
            ["status"],
            registry=registry,  # pyrefly: ignore[bad-argument-type]
        )

        self.plugin_failures_total = Counter(  # type: ignore[assignment]
            "docbuilder_plugin_failures_total",
            "Total number of plugin execution failures",
            ["plugin_name", "error_type"],
            registry=registry,  # pyrefly: ignore[bad-argument-type]
        )

        self.harvest_duration_seconds = Histogram(  # type: ignore[assignment]
            "docbuilder_harvest_duration_seconds",
            "Duration of harvest operations in seconds",
            ["status"],
            registry=registry,  # pyrefly: ignore[bad-argument-type]
        )

        self.policy_duration_seconds = Histogram(  # type: ignore[assignment]
            "docbuilder_policy_duration_seconds",
            "Duration of policy engine operations in seconds",
            ["status"],
            registry=registry,  # pyrefly: ignore[bad-argument-type]
        )

        self.render_duration_seconds = Histogram(  # type: ignore[assignment]
            "docbuilder_render_duration_seconds",
            "Duration of rendering operations in seconds",
            ["status"],
            registry=registry,  # pyrefly: ignore[bad-argument-type]
        )

        self.cli_duration_seconds = Histogram(  # type: ignore[assignment]
            "docbuilder_cli_duration_seconds",
            "Duration of CLI operations in seconds",
            ["command", "status"],
            registry=registry,  # pyrefly: ignore[bad-argument-type]
        )


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
