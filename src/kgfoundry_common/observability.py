"""Prometheus metrics and OpenTelemetry tracing helpers.

This module provides structured observability with Prometheus metrics
and optional OpenTelemetry tracing. All metrics follow naming conventions
and include operation/status tags. Gracefully degrades when Prometheus
or OpenTelemetry are unavailable.

Examples
--------
>>> from kgfoundry_common.observability import MetricsProvider, observe_duration
>>> metrics = MetricsProvider.default()
>>> with observe_duration(metrics, "search") as obs:
...     # Operation code here
...     obs.success()
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final, Protocol

from kgfoundry_common.logging import get_logger
from kgfoundry_common.navmap_types import NavMap

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "MetricsProvider",
    "observe_duration",
    "start_span",
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

logger = get_logger(__name__)

# Prometheus client is optional (may not be installed)
if TYPE_CHECKING:
    from prometheus_client import Counter, Histogram
    from prometheus_client.registry import CollectorRegistry
else:
    CollectorRegistry = object
    Counter = object
    Histogram = object

try:
    from prometheus_client import Counter, Histogram
    from prometheus_client.registry import CollectorRegistry

    HAVE_PROMETHEUS = True
    _PROMETHEUS_VERSION: str | None = None
    try:
        import prometheus_client

        _PROMETHEUS_VERSION = getattr(prometheus_client, "__version__", None)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not detect Prometheus version: %s", exc)
except ImportError:
    HAVE_PROMETHEUS = False
    _PROMETHEUS_VERSION = None

# OpenTelemetry is optional (may not be installed)
try:
    from opentelemetry import trace

    HAVE_OPENTELEMETRY = True
except ImportError:
    HAVE_OPENTELEMETRY = False
    trace = None  # type: ignore[assignment]


# Protocol definitions for typed metric interfaces
class CounterLike(Protocol):
    """Protocol for counter-like metrics.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    *args : inspect._empty
        Describe ``args``.
    **kwargs : inspect._empty
        Describe ``kwargs``.

    Returns
    -------
    inspect._empty
        Describe return value.
"""

    def labels(self, **kwargs: object) -> CounterLike:
        """Return labeled counter instance.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        **kwargs : object
            Describe ``kwargs``.

        Returns
        -------
        CounterLike
            Describe return value.
"""
        ...

    def inc(self, *args: object, **kwargs: object) -> None:
        """Increment counter.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        *args : object
            Describe ``args``.
        **kwargs : object
            Describe ``kwargs``.
"""
        ...


class HistogramLike(Protocol):
    """Protocol for histogram-like metrics.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    *args : inspect._empty
        Describe ``args``.
    **kwargs : inspect._empty
        Describe ``kwargs``.

    Returns
    -------
    inspect._empty
        Describe return value.
"""

    def labels(self, **kwargs: object) -> HistogramLike:
        """Return labeled histogram instance.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        **kwargs : object
            Describe ``kwargs``.

        Returns
        -------
        HistogramLike
            Describe return value.
"""
        ...

    def observe(self, *args: object, **kwargs: object) -> None:
        """Observe a value.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        *args : object
            Describe ``args``.
        **kwargs : object
            Describe ``kwargs``.
"""
        ...


class GaugeLike(Protocol):
    """Protocol for gauge-like metrics.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    *args : inspect._empty
        Describe ``args``.
    **kwargs : inspect._empty
        Describe ``kwargs``.

    Returns
    -------
    inspect._empty
        Describe return value.
"""

    def labels(self, **kwargs: object) -> GaugeLike:
        """Return labeled gauge instance.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        **kwargs : object
            Describe ``kwargs``.

        Returns
        -------
        GaugeLike
            Describe return value.
"""
        ...

    def set(self, value: float) -> None:
        """Set gauge value.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        value : float
            Describe ``value``.
"""
        ...


# Stub implementations that satisfy type checkers
class _StubCounter:
    """Stub counter for when prometheus_client is unavailable.

    <!-- auto:docstring-builder v1 -->

    Returns
    -------
    inspect._empty
        Describe return value.
"""

    def labels(self, **kwargs: object) -> _StubCounter:  # noqa: ARG002
        """Return self for chaining.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        **kwargs : object
            Describe ``kwargs``.

        Returns
        -------
        _StubCounter
            Describe return value.
"""
        return self

    def inc(self, value: float = 1.0) -> None:
        """No-op increment.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        value : float, optional
            Describe ``value``.
            Defaults to ``1.0``.
"""


class _StubHistogram:
    """Stub histogram for when prometheus_client is unavailable.

    <!-- auto:docstring-builder v1 -->

    Returns
    -------
    inspect._empty
        Describe return value.
"""

    def labels(self, **kwargs: object) -> _StubHistogram:  # noqa: ARG002
        """Return self for chaining.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        **kwargs : object
            Describe ``kwargs``.

        Returns
        -------
        _StubHistogram
            Describe return value.
"""
        return self

    def observe(self, value: float) -> None:
        """No-op observe.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        value : float
            Describe ``value``.
"""


class _StubGauge:
    """Stub gauge for when prometheus_client is unavailable.

    <!-- auto:docstring-builder v1 -->

    Returns
    -------
    inspect._empty
        Describe return value.
"""

    def labels(self, **kwargs: object) -> _StubGauge:  # noqa: ARG002
        """Return self for chaining.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        **kwargs : object
            Describe ``kwargs``.

        Returns
        -------
        _StubGauge
            Describe return value.
"""
        return self

    def set(self, value: float) -> None:
        """No-op set.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        value : float
            Describe ``value``.
"""


_DEFAULT_PROVIDERS: dict[type[MetricsProvider], MetricsProvider] = {}


class MetricsProvider:
    """Metrics provider with Prometheus-compatible counters and histograms.

    <!-- auto:docstring-builder v1 -->

    This class provides typed wrappers for Prometheus metrics with safe
    fallbacks when Prometheus is unavailable. All stub implementations
    return `self` from `.labels()` to allow chaining.

    Parameters
    ----------
    registry : CollectorRegistry | None, optional
        Describe ``registry``.
        Defaults to ``None``.

    Examples
    --------
    >>> metrics = MetricsProvider.default()
    >>> metrics.runs_total.labels(component="search", status="success").inc()
    >>> metrics.operation_duration_seconds.labels(component="search", operation="query").observe(
    ...     0.123
    ... )
"""

    runs_total: CounterLike | _StubCounter
    operation_duration_seconds: HistogramLike | _StubHistogram

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Initialize metrics provider.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        registry : CollectorRegistry | NoneType, optional
            Prometheus registry (defaults to default registry if Prometheus available).
            If None and Prometheus unavailable, stub metrics are used.
            Defaults to ``None``.
"""
        if not HAVE_PROMETHEUS:
            logger.debug("Prometheus not available; using stub metrics")
            self.runs_total = _StubCounter()
            self.operation_duration_seconds = _StubHistogram()
            return

        if registry is None:
            from prometheus_client import REGISTRY  # noqa: PLC0415

            registry = REGISTRY

        # Type narrowing: HAVE_PROMETHEUS is True, so Counter/Histogram are imported
        # Prometheus Counter/Histogram match Protocol interface (labels() returns self, inc/observe methods exist)
        self.runs_total = Counter(  # type: ignore[assignment]  # Prometheus Counter matches Protocol interface
            "kgfoundry_runs_total",
            "Total number of operations",
            ["component", "status"],
            registry=registry,
        )

        self.operation_duration_seconds = Histogram(  # type: ignore[assignment]  # Prometheus Histogram matches Protocol interface
            "kgfoundry_operation_duration_seconds",
            "Operation duration in seconds",
            ["component", "operation", "status"],
            registry=registry,
        )

    @classmethod
    def default(cls) -> MetricsProvider:
        """Create default metrics provider instance.

        <!-- auto:docstring-builder v1 -->

        Returns
        -------
        MetricsProvider
            Default metrics provider (uses Prometheus if available, otherwise stubs).

        Examples
        --------
        >>> metrics = MetricsProvider.default()
        >>> metrics.runs_total.labels(component="search", status="success").inc()
"""
        provider = _DEFAULT_PROVIDERS.get(cls)
        if provider is None:
            provider = cls()
            _DEFAULT_PROVIDERS[cls] = provider
        return provider


class _DurationObserver:
    """Context manager helper for observing operation duration.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    metrics : MetricsProvider
        Describe ``metrics``.
    component : str
        Describe ``component``.
    operation : str
        Describe ``operation``.
    start_time : float
        Describe ``start_time``.
"""

    def __init__(
        self,
        metrics: MetricsProvider,
        component: str,
        operation: str,
        start_time: float,
    ) -> None:
        """Initialize duration observer.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        metrics : MetricsProvider
            Metrics provider instance.
        component : str
            Component name (e.g., "search", "index").
        operation : str
            Operation name (e.g., "query", "index").
        start_time : float
            Start time from `time.monotonic()`.
"""
        self.metrics = metrics
        self.component = component
        self.operation = operation
        self.start_time = start_time
        self.status = "success"

    def success(self) -> None:
        """Mark operation as successful.

        <!-- auto:docstring-builder v1 -->
"""
        self.status = "success"

    def error(self) -> None:
        """Mark operation as failed.

        <!-- auto:docstring-builder v1 -->
"""
        self.status = "error"

    def __enter__(self) -> _DurationObserver:
        """Enter context manager.

        <!-- auto:docstring-builder v1 -->

        Returns
        -------
        _DurationObserver
            Describe return value.
"""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object
    ) -> None:
        """Exit context manager and record metrics.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        exc_type : BaseException | NoneType
            Describe ``exc_type``.
        exc : BaseException | NoneType
            Describe ``exc``.
        tb : object
            Describe ``tb``.
"""
        duration = time.monotonic() - self.start_time

        # Update status if exception occurred
        if exc_type is not None:
            self.status = "error"

        # Record metrics
        self.metrics.runs_total.labels(component=self.component, status=self.status).inc()
        self.metrics.operation_duration_seconds.labels(
            component=self.component,
            operation=self.operation,
            status=self.status,
        ).observe(duration)

        # Log structured entry
        logger.info(
            "Operation completed",
            extra={
                "operation": self.operation,
                "status": self.status,
                "duration_ms": duration * 1000,
                "component": self.component,
            },
        )


@contextmanager
def observe_duration(
    metrics: MetricsProvider,
    operation: str,
    component: str = "unknown",
) -> Iterator[_DurationObserver]:
    """Context manager to observe operation duration and record metrics.

    <!-- auto:docstring-builder v1 -->

    This context manager records operation duration, increments counters,
    and emits structured logs with correlation IDs.

    Parameters
    ----------
    metrics : MetricsProvider
        Metrics provider instance.
    operation : str
        Operation name (e.g., "query", "index").
    component : str, optional
        Component name (e.g., "search", "index"). Defaults to "unknown".
        Defaults to ``'unknown'``.

    Yields
    ------
    _DurationObserver
        Observer instance with `success()` and `error()` methods.

    Examples
    --------
    >>> metrics = MetricsProvider.default()
    >>> with observe_duration(metrics, "search", component="search") as obs:
    ...     # Perform search operation
    ...     obs.success()

    Returns
    -------
    Iterator[_DurationObserver]
        Describe return value.
"""
    start_time = time.monotonic()
    observer = _DurationObserver(metrics, component, operation, start_time)
    yield observer


@contextmanager
def start_span(
    name: str,
    attributes: dict[str, str | int | float | bool] | None = None,
) -> Iterator[None]:
    """Start an OpenTelemetry span with safe fallback.

    <!-- auto:docstring-builder v1 -->

    This context manager creates an OpenTelemetry span if available,
    otherwise performs no operation. This allows code to use tracing
    without requiring OpenTelemetry to be installed.

    Parameters
    ----------
    name : str
        Span name (e.g., "search.query", "index.build").
    attributes : dict[str, str | int | float | bool] | None, optional
        Span attributes for tracing. Defaults to None.
        Defaults to ``None``.

    Yields
    ------
    None
        Context manager yields control to the operation block.

    Examples
    --------
    >>> with start_span("search.query", attributes={"query_id": "abc123"}):
    ...     # Operation code here
    ...     pass

    Returns
    -------
    Iterator[None]
        Describe return value.
"""
    if not HAVE_OPENTELEMETRY or trace is None:
        yield
        return

    # Type narrowing: HAVE_OPENTELEMETRY is True and trace is not None
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        try:
            yield
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            raise
