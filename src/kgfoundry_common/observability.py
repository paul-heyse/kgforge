"""Prometheus metrics and OpenTelemetry tracing helpers.

Examples
--------
>>> from kgfoundry_common.logging import set_correlation_id
>>> from kgfoundry_common.observability import MetricsProvider, observe_duration
>>> set_correlation_id("req-123")
>>> provider = MetricsProvider.default()
>>> with observe_duration(provider, "search", component="search_api") as observer:
...     observer.success()
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, cast

from kgfoundry_common.logging import get_logger, with_fields
from kgfoundry_common.navmap_types import NavMap
from kgfoundry_common.prometheus import (
    CollectorRegistry,
    CounterLike,
    HistogramLike,
    build_counter,
    build_histogram,
    get_default_registry,
)


class _TraceAPI(Protocol):
    def get_tracer(self, name: str) -> Tracer: ...


if TYPE_CHECKING:  # pragma: no cover - typing only
    from opentelemetry.trace import Status as OTStatus
    from opentelemetry.trace import StatusCode as OTStatusCode
    from opentelemetry.trace import Tracer
else:  # pragma: no cover - runtime fallback when dependency missing
    OTStatus = Any
    OTStatusCode = Any
    Tracer = Any


trace_api: _TraceAPI | None
Status: type[OTStatus] | None
StatusCode: type[OTStatusCode] | None
try:  # pragma: no cover - optional dependency guard
    from opentelemetry import trace as trace_api
    from opentelemetry.trace import Status, StatusCode
except ImportError:  # pragma: no cover - optional dependency guard
    trace_api = None
    Status = None
    StatusCode = None


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

LOGGER = get_logger(__name__)
StatusLiteral = Literal["success", "error"]


@dataclass(slots=True)
class _ObservabilityCache:
    provider: MetricsProvider | None = None
    registry: MetricsRegistry | None = None


_OBS_CACHE = _ObservabilityCache()


@dataclass(slots=True)
class MetricsProvider:
    """Provide component-level metrics for long-running operations."""

    runs_total: CounterLike
    operation_duration_seconds: HistogramLike
    _registry: CollectorRegistry | None = field(default=None, repr=False)

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Initialise the metrics provider."""
        resolved_registry = cast(CollectorRegistry | None, registry or get_default_registry())
        self._registry = resolved_registry
        self.runs_total = build_counter(
            "kgfoundry_runs_total",
            "Total number of operations executed by a component.",
            ("component", "status"),
            registry=resolved_registry,
        )
        self.operation_duration_seconds = build_histogram(
            "kgfoundry_operation_duration_seconds",
            "Operation duration in seconds for each component/operation pair.",
            ("component", "operation", "status"),
            registry=resolved_registry,
        )

    @property
    def registry(self) -> CollectorRegistry | None:
        """Return the underlying Prometheus registry (may be ``None``)."""
        return self._registry

    @classmethod
    def default(cls) -> MetricsProvider:
        """Return a cached provider instance suitable for process-wide use."""
        if _OBS_CACHE.provider is None:
            _OBS_CACHE.provider = MetricsProvider()
        return _OBS_CACHE.provider


@dataclass(slots=True)
class MetricsRegistry:
    """Expose request-level counters and histograms for HTTP surfaces."""

    requests_total: CounterLike
    request_errors_total: CounterLike
    request_duration_seconds: HistogramLike
    _registry: CollectorRegistry | None = field(default=None, repr=False)

    def __init__(
        self,
        *,
        namespace: str = "kgfoundry",
        registry: CollectorRegistry | None = None,
    ) -> None:
        """Initialise a metrics registry scoped to ``namespace``."""
        resolved_registry = cast(CollectorRegistry | None, registry or get_default_registry())
        metric_prefix = namespace.replace("-", "_")
        labels = ("operation", "status")
        self._registry = resolved_registry
        self.requests_total = build_counter(
            f"{metric_prefix}_requests_total",
            "Total number of processed operations.",
            labels,
            registry=resolved_registry,
        )
        self.request_errors_total = build_counter(
            f"{metric_prefix}_request_errors_total",
            "Total number of failed operations.",
            labels,
            registry=resolved_registry,
        )
        self.request_duration_seconds = build_histogram(
            f"{metric_prefix}_request_duration_seconds",
            "Operation latency in seconds.",
            ("operation",),
            registry=resolved_registry,
        )

    @property
    def registry(self) -> CollectorRegistry | None:
        """Return the underlying Prometheus registry (may be ``None``)."""
        return self._registry


def get_metrics_registry() -> MetricsRegistry:
    """Return the process-wide metrics registry singleton."""
    if _OBS_CACHE.registry is None:
        _OBS_CACHE.registry = MetricsRegistry()
    return _OBS_CACHE.registry


@dataclass(slots=True)
class _DurationObserver:
    """Track the state of an in-flight operation for structured recording."""

    metrics: MetricsProvider
    operation: str
    component: str
    correlation_id: str | None
    status: StatusLiteral = "success"
    _start: float = field(default_factory=time.monotonic)

    def success(self) -> None:
        """Mark the operation as successful."""
        self.status = "success"

    def error(self) -> None:
        """Mark the operation as failed."""
        self.status = "error"

    def duration_seconds(self) -> float:
        """Return the elapsed duration in seconds."""
        return time.monotonic() - self._start


@contextmanager
def observe_duration(
    metrics: MetricsProvider,
    operation: str,
    *,
    component: str = "unknown",
    correlation_id: str | None = None,
) -> Iterator[_DurationObserver]:
    """Record metrics and structured logs for a component operation."""
    observer = _DurationObserver(metrics, operation, component, correlation_id)
    try:
        yield observer
    except Exception:
        observer.status = "error"
        _finalise_observation(observer)
        raise
    else:
        _finalise_observation(observer)


def _finalise_observation(observer: _DurationObserver) -> None:
    duration = observer.duration_seconds()
    observer.metrics.runs_total.labels(
        component=observer.component,
        status=observer.status,
    ).inc()
    observer.metrics.operation_duration_seconds.labels(
        component=observer.component,
        operation=observer.operation,
        status=observer.status,
    ).observe(duration)
    extra = {
        "component": observer.component,
        "duration_ms": duration * 1000,
    }
    with with_fields(
        LOGGER,
        correlation_id=observer.correlation_id,
        operation=observer.operation,
        status=observer.status,
    ) as adapter:
        adapter.info("Operation completed", extra=extra)


@contextmanager
def record_operation(
    metrics: MetricsRegistry | None = None,
    *,
    operation: str = "unknown",
    correlation_id: str | None = None,
) -> Iterator[None]:
    """Track request counters and latency for an externally visible operation."""
    registry = metrics or get_metrics_registry()
    status: StatusLiteral = "success"
    start = time.monotonic()
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.monotonic() - start
        registry.requests_total.labels(operation=operation, status=status).inc()
        if status == "error":
            registry.request_errors_total.labels(operation=operation, status=status).inc()
        registry.request_duration_seconds.labels(operation=operation).observe(duration)
        extra = {"duration_ms": duration * 1000}
        with with_fields(
            LOGGER,
            correlation_id=correlation_id,
            operation=operation,
            status=status,
        ) as adapter:
            if status == "error":
                adapter.error("Operation failed", extra=extra)
            else:
                adapter.info("Operation completed", extra=extra)


@contextmanager
def start_span(
    name: str,
    attributes: Mapping[str, str | int | float | bool] | None = None,
) -> Iterator[None]:
    """Start an OpenTelemetry span when the optional dependency is available."""
    if trace_api is None:
        yield
        return

    tracer = trace_api.get_tracer(__name__)
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        try:
            yield
        except Exception as exc:  # pragma: no cover - requires OpenTelemetry at runtime
            span.record_exception(exc)
            if Status is not None and StatusCode is not None:
                span.set_status(Status(StatusCode.ERROR))
            raise
