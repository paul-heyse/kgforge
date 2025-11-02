"""OpenTelemetry SDK trace stubs."""

from typing import Any

__all__ = ["SpanProcessor", "TracerProvider"]

class TracerProvider:
    """OpenTelemetry SDK TracerProvider."""

    def get_tracer(self, name: str, /, *args: object, **kwargs: object) -> Any: ...  # noqa: ANN401
    def add_span_processor(self, processor: Any) -> None: ...  # noqa: ANN401

class SpanProcessor:
    """OpenTelemetry SpanProcessor base class."""

    ...
