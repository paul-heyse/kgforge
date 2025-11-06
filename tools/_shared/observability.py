"""Observer interfaces used by the CLI runtime helpers."""

from __future__ import annotations

from typing import Protocol


class MetricEmitterError(RuntimeError):
    """Raised when the configured metric emitter fails."""


class MetricEmitter(Protocol):
    """Protocol describing the observer responsible for emitting CLI metrics."""

    def emit_cli_run(self, *, operation: str, status: str, duration_s: float) -> None:
        """Emit an observation describing a CLI execution."""


class _NoopEmitter:
    """Default emitter used when no concrete metrics backend is installed."""

    @staticmethod
    def emit_cli_run(*, operation: str, status: str, duration_s: float) -> None:
        """Do nothing; placeholder for optional observability integration."""
        del operation, status, duration_s


# Replace at import time inside the host service if a metrics backend is available.
emitter: MetricEmitter = _NoopEmitter()

__all__ = ["MetricEmitter", "MetricEmitterError", "emitter"]
