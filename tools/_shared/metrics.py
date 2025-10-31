"""Observability helpers for tooling subprocess execution.

This module defines Prometheus metrics and OpenTelemetry span helpers that are
consumed by :mod:`tools._shared.proc`. The helpers degrade gracefully when the
optional Prometheus or OpenTelemetry dependencies are not installed so the
calling code never needs to guard imports. Metrics are registered once at import
time using well-known names to make it easy for dashboards to scrape the values
exposed by tooling processes.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Final, Protocol, cast

from kgfoundry_common.observability import start_span
from tools._shared.logging import StructuredLoggerAdapter, get_logger, with_fields

LOGGER = get_logger(__name__)


class CounterLike(Protocol):
    """Subset of Prometheus counter behaviour used by tooling."""

    def labels(self, **kwargs: object) -> CounterLike: ...

    def inc(self, value: float = 1.0) -> None: ...


class HistogramLike(Protocol):
    """Subset of Prometheus histogram behaviour used by tooling."""

    def labels(self, **kwargs: object) -> HistogramLike: ...

    def observe(self, value: float) -> None: ...


class _NoopCounter:
    """Counter stub used when Prometheus is unavailable."""

    def labels(self, **kwargs: object) -> CounterLike:  # noqa: ARG002
        return self

    def inc(self, value: float = 1.0) -> None:  # noqa: ARG002
        return None


class _NoopHistogram:
    """Histogram stub used when Prometheus is unavailable."""

    def labels(self, **kwargs: object) -> HistogramLike:  # noqa: ARG002
        return self

    def observe(self, value: float) -> None:  # noqa: ARG002
        return None


CounterFactory = Callable[[str, str, Sequence[str]], CounterLike]
HistogramFactory = Callable[[str, str, Sequence[str]], HistogramLike]


def _build_counter_factory() -> CounterFactory | None:
    try:
        prometheus = import_module("prometheus_client")
    except ImportError:  # pragma: no cover - optional dependency
        return None

    counter_obj: object | None = getattr(prometheus, "Counter", None)
    if counter_obj is None or not callable(counter_obj):  # pragma: no cover - defensive check
        return None

    counter_callable = cast(Callable[[str, str, Sequence[str]], object], counter_obj)

    def factory(name: str, documentation: str, labelnames: Sequence[str]) -> CounterLike:
        return cast(CounterLike, counter_callable(name, documentation, list(labelnames)))

    return factory


def _build_histogram_factory() -> HistogramFactory | None:
    try:
        prometheus = import_module("prometheus_client")
    except ImportError:  # pragma: no cover - optional dependency
        return None

    histogram_obj: object | None = getattr(prometheus, "Histogram", None)
    if histogram_obj is None or not callable(histogram_obj):  # pragma: no cover - defensive check
        return None

    histogram_callable = cast(Callable[[str, str, Sequence[str]], object], histogram_obj)

    def factory(name: str, documentation: str, labelnames: Sequence[str]) -> HistogramLike:
        return cast(HistogramLike, histogram_callable(name, documentation, list(labelnames)))

    return factory


_PROM_COUNTER_FACTORY: CounterFactory | None = _build_counter_factory()
_PROM_HISTOGRAM_FACTORY: HistogramFactory | None = _build_histogram_factory()


def _make_counter(name: str, documentation: str, labelnames: Sequence[str]) -> CounterLike:
    factory = _PROM_COUNTER_FACTORY
    if factory is None:
        return _NoopCounter()
    return factory(name, documentation, labelnames)


def _make_histogram(name: str, documentation: str, labelnames: Sequence[str]) -> HistogramLike:
    factory = _PROM_HISTOGRAM_FACTORY
    if factory is None:
        return _NoopHistogram()
    return factory(name, documentation, labelnames)


TOOL_RUNS_TOTAL: CounterLike = _make_counter(
    "tool_runs_total",
    "Total tooling subprocess invocations",
    ["tool", "status"],
)

TOOL_FAILURES_TOTAL: CounterLike = _make_counter(
    "tool_failures_total",
    "Count of tooling subprocess failures grouped by reason",
    ["tool", "reason"],
)

TOOL_DURATION_SECONDS: HistogramLike = _make_histogram(
    "tool_duration_seconds",
    "Tooling subprocess duration in seconds",
    ["tool", "status"],
)


@dataclass(slots=True)
class ToolRunObservation:
    """Captures runtime details for a single subprocess invocation."""

    command: Sequence[str]
    cwd: Path | None
    timeout: float | None
    tool: str = field(init=False)
    status: str = field(default="success", init=False)
    failure_reason: str | None = field(default=None, init=False)
    returncode: int | None = field(default=None, init=False)
    timed_out: bool = field(default=False, init=False)
    start_time: float = field(default_factory=time.monotonic, init=False)

    def __post_init__(self) -> None:
        self.tool = Path(self.command[0]).name if self.command else "<unknown>"

    def success(self, returncode: int) -> None:
        """Record successful completion with ``returncode``."""
        self.status = "success"
        self.returncode = returncode
        self.failure_reason = None
        self.timed_out = False

    def failure(
        self,
        reason: str,
        *,
        returncode: int | None = None,
        timed_out: bool = False,
    ) -> None:
        """Record failed completion with context metadata."""
        self.status = "error"
        self.failure_reason = reason
        self.returncode = returncode
        self.timed_out = timed_out

    def duration_seconds(self) -> float:
        """Return the elapsed duration in seconds."""
        return time.monotonic() - self.start_time


@contextmanager
def observe_tool_run(
    command: Sequence[str],
    *,
    cwd: Path | None,
    timeout: float | None,
) -> Iterator[ToolRunObservation]:
    """Context manager that records metrics and tracing for a subprocess."""
    observation = ToolRunObservation(command=command, cwd=cwd, timeout=timeout)
    command_parts: list[str] = list(command)
    logger = with_fields(
        LOGGER,
        tool=observation.tool,
        command=command_parts,
        cwd=str(cwd) if cwd else None,
        timeout_seconds=timeout,
    )
    span_name = f"tools.run.{observation.tool}"
    span_attributes: dict[str, str | int | float | bool | None]
    span_attributes = {
        "tool": observation.tool,
        "cwd": str(cwd) if cwd else "",
        "timeout_s": timeout if timeout is not None else -1.0,
    }

    with start_span(
        span_name, attributes={k: v for k, v in span_attributes.items() if v is not None}
    ):
        try:
            yield observation
        except Exception:
            if observation.status == "success":
                observation.failure("exception")
            _record(observation, logger)
            raise
        else:
            _record(observation, logger)


def _record(
    observation: ToolRunObservation,
    logger: StructuredLoggerAdapter,
) -> None:
    """Persist metrics and structured logs for the finished observation."""
    duration = observation.duration_seconds()
    status = observation.status
    TOOL_RUNS_TOTAL.labels(tool=observation.tool, status=status).inc()
    TOOL_DURATION_SECONDS.labels(tool=observation.tool, status=status).observe(duration)
    extra: dict[str, object] = {
        "duration_ms": duration * 1000,
        "tool": observation.tool,
        "status": status,
        "returncode": observation.returncode,
        "timed_out": observation.timed_out,
    }
    if status == "error":
        reason = observation.failure_reason or "unknown"
        TOOL_FAILURES_TOTAL.labels(tool=observation.tool, reason=reason).inc()
        extra["reason"] = reason
        logger.error("Tool run failed", extra=extra)
    else:
        logger.info("Tool run succeeded", extra=extra)


__all__: Final[list[str]] = [
    "ToolRunObservation",
    "observe_tool_run",
]
