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
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final, Protocol, cast

from kgfoundry_common.logging import get_logger
from kgfoundry_common.navmap_types import NavMap

__all__ = [
    "MetricsProvider",
    "MetricsRegistry",
    "get_metrics_registry",
    "observe_duration",
    "record_operation",
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

if TYPE_CHECKING:
    from prometheus_client.metrics import Counter as PromCounterType
    from prometheus_client.metrics import Histogram as PromHistogramType
    from prometheus_client.registry import CollectorRegistry as PromCollectorRegistryType
    CollectorRegistryType = PromCollectorRegistryType
else:
    PromCounterType = PromHistogramType = PromCollectorRegistryType = object
    CollectorRegistryType = object  # type: ignore[assignment]

_RuntimeCounter: object | None = None
_RuntimeHistogram: object | None = None
_RuntimeCollectorRegistry: object | None = None

try:
    from prometheus_client import Counter as _RuntimeCounter
    from prometheus_client import Histogram as _RuntimeHistogram
    from prometheus_client.registry import CollectorRegistry as _RuntimeCollectorRegistry

    HAVE_PROMETHEUS = True
    _PROMETHEUS_VERSION: str | None
    try:
        import prometheus_client

        _PROMETHEUS_VERSION = getattr(prometheus_client, "__version__", None)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not detect Prometheus version: %s", exc)
        _PROMETHEUS_VERSION = None
except ImportError:
    HAVE_PROMETHEUS = False
    _PROMETHEUS_VERSION = None
    _RuntimeCounter = None
    _RuntimeHistogram = None
    _RuntimeCollectorRegistry = None


class _StubCollectorRegistry:
    """Stub collector registry used when Prometheus is unavailable."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._args = args
        self._kwargs = kwargs


class _StubCounter:
    """Stub counter used when Prometheus is unavailable."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._args = args
        self._kwargs = kwargs

    def labels(self, **kwargs: object) -> _StubCounter:
        return self

    def inc(self, *args: object, **kwargs: object) -> None:
        return None


class _StubHistogram:
    """Stub histogram used when Prometheus is unavailable."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._args = args
        self._kwargs = kwargs

    def labels(self, **kwargs: object) -> _StubHistogram:
        return self

    def observe(self, *args: object, **kwargs: object) -> None:
        return None


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


# Stub gauge for compatibility (unchanged behavior when Prometheus missing)
class _StubGauge:
    """Stub gauge for when prometheus_client is unavailable."""

    def labels(self, **kwargs: object) -> _StubGauge:  # noqa: ARG002
        return self

    def set(self, value: float) -> None:  # noqa: ARG002
        return None


_DEFAULT_PROVIDERS: dict[type[MetricsProvider], MetricsProvider] = {}


def _build_counter(
    name: str,
    documentation: str,
    labelnames: list[str],
    registry: CollectorRegistryType | None,
) -> CounterLike | _StubCounter:
    if not HAVE_PROMETHEUS or _RuntimeCounter is None:
        return _StubCounter()
    runtime_counter = cast(type[PromCounterType], _RuntimeCounter)
    prom_registry: PromCollectorRegistryType | None
    prom_registry = cast(PromCollectorRegistryType | None, registry)
    counter = runtime_counter(name, documentation, labelnames, registry=prom_registry)
    return cast(CounterLike, counter)


def _build_histogram(
    name: str,
    documentation: str,
    labelnames: list[str],
    registry: CollectorRegistryType | None,
) -> HistogramLike | _StubHistogram:
    if not HAVE_PROMETHEUS or _RuntimeHistogram is None:
        return _StubHistogram()
    runtime_histogram = cast(type[PromHistogramType], _RuntimeHistogram)
    prom_registry: PromCollectorRegistryType | None
    prom_registry = cast(PromCollectorRegistryType | None, registry)
    histogram = runtime_histogram(name, documentation, labelnames, registry=prom_registry)
    return cast(HistogramLike, histogram)


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

    def __init__(self, registry: CollectorRegistryType | None = None) -> None:
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
            if _RuntimeCollectorRegistry is None:
                msg = "Prometheus registry is unavailable despite HAVE_PROMETHEUS=True"
                raise RuntimeError(msg)
            from prometheus_client import REGISTRY  # noqa: PLC0415

            registry = REGISTRY

        # Type narrowing: HAVE_PROMETHEUS is True, so Counter/Histogram are imported
        # Prometheus Counter/Histogram match Protocol interface (labels() returns self, inc/observe methods exist)
        self.runs_total = _build_counter(
            "kgfoundry_runs_total",
            "Total number of operations",
            ["component", "status"],
            registry,
        )

        self.operation_duration_seconds = _build_histogram(
            "kgfoundry_operation_duration_seconds",
            "Operation duration in seconds",
            ["component", "operation", "status"],
            registry,
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


class MetricsRegistry:
    """Prometheus-style registry for high-level request metrics."""

    requests_total: CounterLike | _StubCounter
    request_errors_total: CounterLike | _StubCounter
    request_duration_seconds: HistogramLike | _StubHistogram

    def __init__(
        self,
        *,
        namespace: str = "kgfoundry",
        registry: CollectorRegistryType | None = None,
    ) -> None:
        if HAVE_PROMETHEUS:
            if registry is None:
                if _RuntimeCollectorRegistry is None:
                    msg = "Prometheus registry construction failed"
                    raise RuntimeError(msg)
                registry = _RuntimeCollectorRegistry()
        else:
            if registry is None:
                registry = _StubCollectorRegistry()

        metric_prefix = namespace.replace("-", "_")
        labels_counter = ["operation", "status"]
        labels_hist = ["operation"]

        self._registry = registry
        self.requests_total = _build_counter(
            f"{metric_prefix}_requests_total",
            "Total number of operations processed",
            labels_counter,
            registry,
        )
        self.request_errors_total = _build_counter(
            f"{metric_prefix}_request_errors_total",
            "Total number of failed operations",
            labels_counter,
            registry,
        )
        self.request_duration_seconds = _build_histogram(
            f"{metric_prefix}_request_duration_seconds",
            "Operation duration in seconds",
            labels_hist,
            registry,
        )

    @property
    def registry(self) -> CollectorRegistryType | None:
        """Return the underlying Prometheus registry when available."""
        return self._registry


_METRICS_REGISTRY: MetricsRegistry | None = None


def get_metrics_registry() -> MetricsRegistry:
    """Return the process-wide metrics registry singleton."""
    global _METRICS_REGISTRY
    if _METRICS_REGISTRY is None:
        _METRICS_REGISTRY = MetricsRegistry()
    return _METRICS_REGISTRY


@contextmanager
def record_operation(
    metrics: MetricsRegistry | None = None,
    operation: str = "unknown",
    status: str = "success",
) -> Iterator[None]:
    """Record request counters and durations around an operation."""
    registry = metrics or get_metrics_registry()
    start_time = time.monotonic()
    error_status = "error"

    try:
        yield
    except Exception:
        registry.requests_total.labels(operation=operation, status=error_status).inc()
        registry.request_errors_total.labels(operation=operation, status=error_status).inc()
        duration = time.monotonic() - start_time
        registry.request_duration_seconds.labels(operation=operation).observe(duration)

        logger.warning(
            "Operation failed",
            extra={
                "operation": operation,
                "status": error_status,
                "duration_ms": duration * 1000,
            },
        )
        raise
    else:
        duration = time.monotonic() - start_time
        registry.requests_total.labels(operation=operation, status=status).inc()
        registry.request_duration_seconds.labels(operation=operation).observe(duration)

        logger.info(
            "Operation completed",
            extra={
                "operation": operation,
                "status": status,
                "duration_ms": duration * 1000,
            },
        )


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
