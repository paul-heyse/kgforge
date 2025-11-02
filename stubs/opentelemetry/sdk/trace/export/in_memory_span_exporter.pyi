"""OpenTelemetry InMemorySpanExporter stubs."""

from typing import Any

__all__ = ["InMemorySpanExporter"]

class InMemorySpanExporter:
    """OpenTelemetry InMemorySpanExporter."""

    def get_finished_spans(self) -> list[Any]: ...
    def clear(self) -> None: ...
