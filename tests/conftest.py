"""Shared pytest fixtures for table-driven testing and observability validation.

This module provides reusable fixtures for:
- Search options and configuration factories
- Problem Details payload loading and validation
- CLI command execution with captured output
- Logging and metrics capture (logs, Prometheus, OpenTelemetry)
- Idempotency and retry simulation
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import TYPE_CHECKING, ParamSpec, Protocol, TypeVar, cast

import pytest
from _pytest.logging import LogCaptureFixture
from prometheus_client.registry import CollectorRegistry

from kgfoundry_common.opentelemetry_types import (
    SpanExporterProtocol,
    SpanProtocol,
    TracerProviderProtocol,
    load_in_memory_span_exporter_cls,
    load_tracer_provider_cls,
)
from kgfoundry_common.problem_details import JsonValue

repo_root = Path(__file__).parent.parent

P = ParamSpec("P")
R = TypeVar("R")

if TYPE_CHECKING:  # pragma: no cover - typing support only

    def fixture(*args: object, **kwargs: object) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

else:
    fixture = pytest.fixture

# Type aliases
ProblemDetailsDict = dict[str, JsonValue]


class MetricSample(Protocol):
    value: float


class MetricFamily(Protocol):
    name: str
    samples: Sequence[MetricSample] | Iterable[MetricSample]


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest by setting up Python path for src packages.

    Parameters
    ----------
    config
        Pytest configuration object (required by pytest hook).
    """
    _ = config  # Unused but required by pytest hook signature
    repo_root = Path(__file__).parent.parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


@fixture
def temp_index_dir() -> Iterator[Path]:
    """Provide a temporary directory for index operations.

    Yields
    ------
    Path
        Temporary directory that is cleaned up after the test.
    """
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@fixture
def caplog_records(caplog: LogCaptureFixture) -> dict[str, list[logging.LogRecord]]:
    """Capture logs by operation name for structured assertions.

    Returns
    -------
    dict[str, list[LogRecord]]
        Mapping of operation → list of log records for that operation.
    """

    def _collect_records() -> dict[str, list[logging.LogRecord]]:
        """Collect records grouped by operation."""
        records_by_op: dict[str, list[logging.LogRecord]] = {}
        records = [record for record in caplog.records if isinstance(record, logging.LogRecord)]
        for record in records:
            record_dict = cast(dict[str, object], record.__dict__)
            op_obj = record_dict.get("operation", "unknown")
            op = op_obj if isinstance(op_obj, str) else "unknown"
            records_by_op.setdefault(op, []).append(record)
        return records_by_op

    return _collect_records()


@fixture
def prometheus_registry() -> CollectorRegistry:
    """Provide an isolated Prometheus registry for metrics capture.

    Returns
    -------
    CollectorRegistry
        Fresh registry for this test; cleaned up after test completes.
    """
    return CollectorRegistry()


# Optional OpenTelemetry fixtures


@fixture
def otel_span_exporter() -> SpanExporterProtocol:
    """Provide an in-memory OpenTelemetry span exporter for testing."""
    exporter_cls = load_in_memory_span_exporter_cls()
    if exporter_cls is None:
        pytest.skip("OpenTelemetry span exporter required for observability tests")
        raise RuntimeError("OpenTelemetry span exporter unavailable")
    exporter: SpanExporterProtocol = exporter_cls()
    return exporter


@fixture
def otel_tracer_provider(
    otel_span_exporter: SpanExporterProtocol,
) -> Iterator[TracerProviderProtocol]:
    """Provide an OpenTelemetry tracer provider configured with in-memory exporter."""
    tracer_provider_cls = load_tracer_provider_cls()
    if tracer_provider_cls is None:
        pytest.skip("OpenTelemetry SDK required for observability tests")
        raise RuntimeError("OpenTelemetry SDK unavailable")

    otel_trace_mod = pytest.importorskip(
        "opentelemetry.trace",
        reason="OpenTelemetry API required for observability tests",
    )
    assert isinstance(otel_trace_mod, ModuleType)

    provider: TracerProviderProtocol = tracer_provider_cls()
    span_processor = _SimpleSpanProcessor(otel_span_exporter)
    provider.add_span_processor(span_processor)
    set_tracer_provider = cast(
        Callable[[TracerProviderProtocol], None], otel_trace_mod.set_tracer_provider
    )
    get_tracer_provider = cast(
        Callable[[], TracerProviderProtocol], otel_trace_mod.get_tracer_provider
    )
    set_tracer_provider(provider)
    try:
        yield provider
    finally:
        set_tracer_provider(get_tracer_provider())


def load_problem_details_example(example_name: str) -> ProblemDetailsDict:
    """Load a Problem Details example from schema/examples.

    Parameters
    ----------
    example_name : str
        Name of the example file (e.g., "search-missing-index").

    Returns
    -------
    dict
        Parsed Problem Details JSON.

    Raises
    ------
    FileNotFoundError
        If example file does not exist.
    json.JSONDecodeError
        If example JSON is malformed.
    """
    example_path = (
        Path(__file__).parent.parent / "schema/examples/problem_details" / f"{example_name}.json"
    )
    if not example_path.exists():
        msg = f"Problem Details example not found: {example_path}"
        raise FileNotFoundError(msg)

    return cast(ProblemDetailsDict, json.loads(example_path.read_text(encoding="utf-8")))


@fixture
def problem_details_loader() -> Callable[[str], ProblemDetailsDict]:
    """Fixture providing access to Problem Details examples.

    Yields
    ------
    callable
        Function load_problem_details_example() bound to this fixture.
    """
    return load_problem_details_example


@fixture
def structured_log_asserter() -> Callable[[logging.LogRecord, set[str]], None]:
    """Provide helpers for asserting structured log fields.

    Yields
    ------
    callable
        Function to assert log record has required fields.
    """

    def assert_log_has_fields(
        record: logging.LogRecord,
        required_fields: set[str],
    ) -> None:
        """Assert log record includes all required fields.

        Parameters
        ----------
        record : LogRecord
            The log record to check.
        required_fields : set[str]
            Field names that must be present.

        Raises
        ------
        AssertionError
            If any required field is missing.
        """
        record_dict = cast(dict[str, object], record.__dict__)
        missing = required_fields - set(record_dict.keys())
        if missing:
            msg = f"Missing fields in log record: {missing}"
            raise AssertionError(msg)

    return assert_log_has_fields


@fixture
def metrics_asserter(
    prometheus_registry: CollectorRegistry,
) -> Callable[[str, int | float | None], None]:
    """Provide helpers for asserting Prometheus metrics.

    Yields
    ------
    callable
        Function to assert metric exists and has expected value.
    """

    def assert_metric(name: str, value: int | float | None = None) -> None:
        """Assert Prometheus metric exists and optionally has expected value.

        Parameters
        ----------
        name : str
            Metric name to check.
        value : int | float | None, optional
            Expected value (if None, only checks existence).

        Raises
        ------
        AssertionError
            If metric not found or value mismatch.
        """
        # Collect all families and samples
        families = [cast(MetricFamily, family) for family in prometheus_registry.collect()]
        for family in families:
            if family.name == name:
                samples_raw = list(family.samples)
                if samples_raw and value is not None:
                    sample = samples_raw[0]
                    actual = float(sample.value)
                    if actual != value:
                        msg = f"Metric {name}: expected {value}, got {actual}"
                        raise AssertionError(msg)
                return

        msg = f"Metric not found: {name}"
        raise AssertionError(msg)

    return assert_metric


class _SimpleSpanProcessor:
    """Simple span processor for in-memory collection during tests."""

    def __init__(self, exporter: SpanExporterProtocol) -> None:
        """Initialize with exporter.

        Parameters
        ----------
        exporter : SpanExporterProtocol
            OTEL span exporter.
        """
        self.exporter = exporter

    def on_start(self, span: SpanProtocol, parent_context: object | None = None) -> None:
        """No-op start hook to satisfy span processor protocol."""

    def force_flush(self, timeout_millis: int = 0) -> bool:
        """Flush spans immediately (noop)."""
        return True

    def on_end(self, span: SpanProtocol) -> None:
        """Process ended span.

        Parameters
        ----------
        span : object
            The ended span.
        """
        self.exporter.export([span])

    def shutdown(self) -> None:
        """Allow graceful shutdown calls (noop)."""


__all__ = [
    "caplog_records",
    "load_problem_details_example",
    "metrics_asserter",
    "otel_span_exporter",
    "otel_tracer_provider",
    "problem_details_loader",
    "prometheus_registry",
    "structured_log_asserter",
    "temp_index_dir",
]
