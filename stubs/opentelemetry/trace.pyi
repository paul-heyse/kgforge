"""OpenTelemetry trace stubs matching the subset used by observability helpers."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from enum import Enum
from typing import Any

__all__ = [
    "NoOpSpanProcessor",
    "Span",
    "Status",
    "StatusCode",
    "Tracer",
    "TracerProvider",
    "get_tracer",
    "get_tracer_provider",
    "set_tracer_provider",
]

class StatusCode(Enum):
    """Enumeration of span status codes."""

    UNSET = 0
    OK = 1
    ERROR = 2

class Status:
    """Span status container."""

    def __init__(self, status_code: StatusCode, description: str | None = None) -> None: ...

class Span:
    """Active span handle."""

    def set_attribute(self, key: str, value: str | int | float | bool) -> None: ...
    def record_exception(
        self,
        exception: BaseException,
        attributes: Mapping[str, object] | None = None,
        /,
    ) -> None: ...
    def set_status(self, status: Status) -> None: ...

class Tracer:
    """Tracer capable of creating spans."""

    def start_as_current_span(
        self,
        name: str,
        /,
        *args: object,
        **kwargs: object,
    ) -> AbstractContextManager[Span]: ...

class TracerProvider:
    """Tracer provider factory."""

    def get_tracer(self, name: str, /, *args: object, **kwargs: object) -> Tracer: ...

class NoOpSpanProcessor:
    """Span processor that performs no operations."""

    def __init__(self) -> None: ...

def get_tracer(name: str, /, *args: object, **kwargs: object) -> Tracer: ...
def get_tracer_provider() -> Any: ...  # noqa: ANN401
def set_tracer_provider(provider: Any) -> None: ...  # noqa: ANN401
