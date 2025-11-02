"""OpenTelemetry SDK trace stubs."""

from __future__ import annotations

from opentelemetry.trace import Span as APITraceSpan
from opentelemetry.trace import Tracer as APITracer
from opentelemetry.util.types import Attributes

__all__ = ["SpanProcessor", "TracerProvider"]

class TracerProvider:
    """OpenTelemetry SDK TracerProvider."""

    def __init__(self, *args: object, **kwargs: object) -> None: ...

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: str | None = ...,
        schema_url: str | None = ...,
        attributes: Attributes | None = ...,
    ) -> APITracer: ...

    def add_span_processor(self, processor: SpanProcessor) -> None: ...

    def shutdown(self) -> None: ...

    def force_flush(self, timeout_millis: int = ...) -> bool: ...

class SpanProcessor:
    """OpenTelemetry SpanProcessor base class."""

    def on_start(self, span: APITraceSpan, parent_context: object | None = ...) -> None: ...

    def on_end(self, span: APITraceSpan) -> None: ...

    def shutdown(self) -> None: ...

    def force_flush(self, timeout_millis: int = ...) -> bool: ...
