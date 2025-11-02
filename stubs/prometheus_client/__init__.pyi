from __future__ import annotations

from collections.abc import Sequence

from prometheus_client.registry import CollectorRegistry

class _MetricBase:
    def labels(self, *labelvalues: str, **labelkwargs: str) -> _MetricBase: ...

class Counter(_MetricBase):
    """Prometheus counter metric."""

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: Sequence[str] | None = ...,
        *,
        namespace: str | None = ...,
        subsystem: str | None = ...,
        unit: str | None = ...,
        registry: CollectorRegistry | None = ...,
        **kwargs: object,
    ) -> None: ...
    def labels(self, *labelvalues: str, **labelkwargs: str) -> Counter: ...
    def inc(self, amount: float = ...) -> None: ...

class Gauge(_MetricBase):
    """Prometheus gauge metric."""

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: Sequence[str] | None = ...,
        *,
        namespace: str | None = ...,
        subsystem: str | None = ...,
        unit: str | None = ...,
        registry: CollectorRegistry | None = ...,
        **kwargs: object,
    ) -> None: ...
    def labels(self, *labelvalues: str, **labelkwargs: str) -> Gauge: ...
    def set(self, value: float) -> None: ...

class Histogram(_MetricBase):
    """Prometheus histogram metric."""

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: Sequence[str] | None = ...,
        *,
        buckets: Sequence[float] | None = ...,
        namespace: str | None = ...,
        subsystem: str | None = ...,
        unit: str | None = ...,
        registry: CollectorRegistry | None = ...,
        **kwargs: object,
    ) -> None: ...
    def labels(self, *labelvalues: str, **labelkwargs: str) -> Histogram: ...
    def observe(self, amount: float) -> None: ...

def start_http_server(
    port: int, addr: str = ..., registry: CollectorRegistry | None = ...
) -> None: ...
def generate_latest(registry: CollectorRegistry | None = ...) -> bytes: ...

REGISTRY: CollectorRegistry
