"""Typed facades for optional OpenTelemetry integrations.

This module centralises the small portion of the OpenTelemetry surface that the
codebase relies on.  We define lightweight ``Protocol`` interfaces that mirror
the runtime behaviour we exercise and provide helpers for loading the optional
dependency safely at runtime without leaking ``Any`` into the type graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast

SpanAttributeValue = str | int | float | bool
Attributes = Mapping[str, SpanAttributeValue]


class SpanProtocol(Protocol):
    """Minimal span surface exercised by the observability helpers."""

    def set_attribute(self, key: str, value: SpanAttributeValue) -> None: ...

    def record_exception(self, exception: BaseException) -> None: ...

    def set_status(self, status: object) -> None: ...


class TracerProtocol(Protocol):
    """Tracer facade returned by ``opentelemetry.trace.get_tracer``."""

    def start_as_current_span(self, name: str) -> AbstractContextManager[SpanProtocol]: ...


class TraceAPIProtocol(Protocol):
    """Subset of the OpenTelemetry trace module used by the codebase."""

    def get_tracer(self, name: str) -> TracerProtocol: ...


class SpanProcessorProtocol(Protocol):
    """Span processor lifecycle hooks invoked by the tests."""

    def on_start(self, span: SpanProtocol, parent_context: object | None = ...) -> None: ...

    def on_end(self, span: SpanProtocol) -> None: ...

    def shutdown(self) -> None: ...

    def force_flush(self, timeout_millis: int = ...) -> bool: ...


class TracerProviderProtocol(Protocol):
    """Tracer provider constructor and helpers used in fixtures."""

    def __init__(self, *args: object, **kwargs: object) -> None: ...

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: str | None = ...,
        schema_url: str | None = ...,
        attributes: Attributes | None = ...,
    ) -> TracerProtocol: ...

    def add_span_processor(self, processor: SpanProcessorProtocol) -> None: ...

    def shutdown(self) -> None: ...

    def force_flush(self, timeout_millis: int = ...) -> bool: ...


class SpanExporterProtocol(Protocol):
    """Minimal exporter interface used by the OpenTelemetry fixtures."""

    def export(self, spans: Sequence[object]) -> None: ...


class StatusCodeProtocol(Protocol):
    """Subset of ``opentelemetry.trace.StatusCode`` used in observability."""

    ERROR: StatusCodeProtocol


class StatusFactory(Protocol):
    """Factory callable for instantiating ``Status`` instances."""

    def __call__(
        self,
        status_code: StatusCodeProtocol,
        description: str | None = ...,
    ) -> object: ...


@dataclass(slots=True)
class TraceRuntime:
    """Container for optional OpenTelemetry runtime handles."""

    api: TraceAPIProtocol | None
    status_factory: StatusFactory | None
    status_codes: StatusCodeProtocol | None


def _safe_getattr(module: object, name: str) -> object | None:
    try:
        return cast(object, getattr(module, name))
    except AttributeError:  # pragma: no cover - defensive
        return None


def load_trace_runtime() -> TraceRuntime:
    """Return the OpenTelemetry trace runtime handles if available."""
    try:
        trace_module = import_module("opentelemetry.trace")
    except ImportError:
        return TraceRuntime(api=None, status_factory=None, status_codes=None)

    api = cast(TraceAPIProtocol, trace_module)
    status_factory = _safe_getattr(trace_module, "Status")
    status_codes = _safe_getattr(trace_module, "StatusCode")
    return TraceRuntime(
        api=api,
        status_factory=cast(StatusFactory | None, status_factory),
        status_codes=cast(StatusCodeProtocol | None, status_codes),
    )


def load_tracer_provider_cls() -> type[TracerProviderProtocol] | None:
    """Return the OpenTelemetry ``TracerProvider`` class if present."""
    try:
        sdk_trace_module = import_module("opentelemetry.sdk.trace")
    except ImportError:
        return None

    provider_cls = _safe_getattr(sdk_trace_module, "TracerProvider")
    if provider_cls is None:
        return None
    return cast(type[TracerProviderProtocol], provider_cls)


def load_in_memory_span_exporter_cls() -> type[SpanExporterProtocol] | None:
    """Return the in-memory span exporter class if the SDK is installed."""
    try:
        exporter_module = import_module("opentelemetry.sdk.trace.export.in_memory_span_exporter")
    except ImportError:
        return None

    exporter_cls = _safe_getattr(exporter_module, "InMemorySpanExporter")
    if exporter_cls is None:
        return None
    return cast(type[SpanExporterProtocol], exporter_cls)
