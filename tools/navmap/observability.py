"""Observability instrumentation for navmap utilities.

This module provides Prometheus metrics and structured logging helpers for
navmap operations (build, check, repair, migrate). Metrics follow Prometheus
naming conventions and include operation/status tags.
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
else:
    Counter = cast(CounterFactory, _PromCounter)
    Histogram = cast(HistogramFactory, _PromHistogram)

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

    build_runs_total: CounterLike | _StubCounter
    check_runs_total: CounterLike | _StubCounter
    repair_runs_total: CounterLike | _StubCounter
    migrate_runs_total: CounterLike | _StubCounter
    build_duration_seconds: HistogramLike | _StubHistogram
    check_duration_seconds: HistogramLike | _StubHistogram
    repair_duration_seconds: HistogramLike | _StubHistogram
    migrate_duration_seconds: HistogramLike | _StubHistogram

    def __init__(self, registry: CollectorRegistryType | None = None) -> None:
        """Initialize metrics registry.

        Parameters
        ----------
        registry : Registry | None, optional
            Prometheus registry (defaults to default registry).
        """
        if not HAVE_PROMETHEUS:
            self.build_runs_total = _StubCounter()
            self.check_runs_total = _StubCounter()
            self.repair_runs_total = _StubCounter()
            self.migrate_runs_total = _StubCounter()
            self.build_duration_seconds = _StubHistogram()
            self.check_duration_seconds = _StubHistogram()
            self.repair_duration_seconds = _StubHistogram()
            self.migrate_duration_seconds = _StubHistogram()
            return

        if registry is None:
            from prometheus_client import REGISTRY  # noqa: PLC0415

            registry = REGISTRY

        # registry is an instance, not a type
        self.registry: CollectorRegistryType = registry  # type: ignore[assignment]

        # Counter and Histogram are correctly typed at module level via cast()
        # pyrefly sees the correct types because we don't use TYPE_CHECKING stubs
        self.build_runs_total = Counter(  # type: ignore[assignment]
            "navmap_build_runs_total",
            "Total number of navmap build operations",
            ["status"],
            registry=registry,
        )

        self.check_runs_total = Counter(  # type: ignore[assignment]
            "navmap_check_runs_total",
            "Total number of navmap check operations",
            ["status"],
            registry=registry,
        )

        self.repair_runs_total = Counter(  # type: ignore[assignment]
            "navmap_repair_runs_total",
            "Total number of navmap repair operations",
            ["status"],
            registry=registry,
        )

        self.migrate_runs_total = Counter(  # type: ignore[assignment]
            "navmap_migrate_runs_total",
            "Total number of navmap migrate operations",
            ["status"],
            registry=registry,
        )

        self.build_duration_seconds = Histogram(  # type: ignore[assignment]
            "navmap_build_duration_seconds",
            "Duration of navmap build operations in seconds",
            ["status"],
            registry=registry,
        )

        self.check_duration_seconds = Histogram(  # type: ignore[assignment]
            "navmap_check_duration_seconds",
            "Duration of navmap check operations in seconds",
            ["status"],
            registry=registry,
        )

        self.repair_duration_seconds = Histogram(  # type: ignore[assignment]
            "navmap_repair_duration_seconds",
            "Duration of navmap repair operations in seconds",
            ["status"],
            registry=registry,
        )

        self.migrate_duration_seconds = Histogram(  # type: ignore[assignment]
            "navmap_migrate_duration_seconds",
            "Duration of navmap migrate operations in seconds",
            ["status"],
            registry=registry,
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
    global _METRICS_REGISTRY  # noqa: PLW0603
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
