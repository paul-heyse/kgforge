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
from collections.abc import Generator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from prometheus_client import CollectorRegistry

# Set up path for src packages (after imports above)
repo_root = Path(__file__).parent.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Type aliases
ProblemDetailsDict = dict[str, Any]


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def] - pytest hook signature
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


@pytest.fixture
def temp_index_dir() -> Generator[Path]:
    """Provide a temporary directory for index operations.

    Yields
    ------
    Path
        Temporary directory that is cleaned up after the test.
    """
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def caplog_records(caplog) -> dict[str, list[logging.LogRecord]]:
    """Capture logs by operation name for structured assertions.

    Returns
    -------
    dict[str, list[LogRecord]]
        Mapping of operation → list of log records for that operation.
    """

    def _collect_records() -> dict[str, list[logging.LogRecord]]:
        """Collect records grouped by operation."""
        records_by_op: dict[str, list[logging.LogRecord]] = {}
        for record in caplog.records:
            op = record.__dict__.get("operation", "unknown")
            records_by_op.setdefault(op, []).append(record)
        return records_by_op

    return _collect_records()


@pytest.fixture
def prometheus_registry() -> CollectorRegistry:
    """Provide an isolated Prometheus registry for metrics capture.

    Returns
    -------
    CollectorRegistry
        Fresh registry for this test; cleaned up after test completes.
    """
    return CollectorRegistry()


# Optional OpenTelemetry imports
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    HAVE_OTEL = True
except ImportError:
    HAVE_OTEL = False
    otel_trace = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment]
    InMemorySpanExporter = None  # type: ignore[assignment]


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

    return json.loads(example_path.read_text(encoding="utf-8"))


@pytest.fixture
def problem_details_loader():
    """Fixture providing access to Problem Details examples.

    Yields
    ------
    callable
        Function load_problem_details_example() bound to this fixture.
    """
    return load_problem_details_example


@pytest.fixture
def structured_log_asserter():
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
        record_dict = record.__dict__
        missing = required_fields - set(record_dict.keys())
        if missing:
            msg = f"Missing fields in log record: {missing}"
            raise AssertionError(msg)

    return assert_log_has_fields


@pytest.fixture
def metrics_asserter(prometheus_registry):
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
        families = list(prometheus_registry.collect())
        for family in families:
            if family.name == name:
                samples = list(family.samples)
                if samples and value is not None:
                    actual = samples[0].value
                    if actual != value:
                        msg = f"Metric {name}: expected {value}, got {actual}"
                        raise AssertionError(msg)
                return

        msg = f"Metric not found: {name}"
        raise AssertionError(msg)

    return assert_metric


@pytest.fixture
def otel_span_exporter() -> Any:  # noqa: ANN401 - OTEL library typing
    """Provide an in-memory OpenTelemetry span exporter for testing.

    Yields
    ------
    InMemorySpanExporter | None
        Span exporter for capturing traces (or None if OTEL unavailable).
    """
    if not HAVE_OTEL or InMemorySpanExporter is None:
        yield None
    else:
        yield InMemorySpanExporter()


@pytest.fixture
def otel_tracer_provider(otel_span_exporter: Any) -> Any:  # noqa: ANN401 - OTEL library typing
    """Provide an OpenTelemetry tracer provider configured with in-memory exporter.

    Parameters
    ----------
    otel_span_exporter : Any
        Span exporter fixture (passed through pytest injection).

    Yields
    ------
    TracerProvider | None
        Tracer provider for creating spans (or None if OTEL unavailable).
    """
    if not HAVE_OTEL or TracerProvider is None or otel_span_exporter is None:
        yield None
    else:
        provider = TracerProvider()
        provider.add_span_processor(
            otel_trace.NoOpSpanProcessor() if otel_trace else None  # type: ignore[misc]
        )
        if otel_span_exporter:
            provider.add_span_processor(
                _SimpleSpanProcessor(otel_span_exporter)  # type: ignore[arg-type]
            )
        otel_trace.set_tracer_provider(provider)  # type: ignore[misc]
        yield provider
        otel_trace.set_tracer_provider(otel_trace.get_tracer_provider())  # type: ignore[misc]


class _SimpleSpanProcessor:
    """Simple span processor for in-memory collection during tests."""

    def __init__(self, exporter: Any) -> None:  # noqa: ANN401
        """Initialize with exporter.

        Parameters
        ----------
        exporter : Any
            OTEL span exporter.
        """
        self.exporter = exporter

    def on_end(self, span: Any) -> None:  # noqa: ANN401
        """Process ended span.

        Parameters
        ----------
        span : Any
            The ended span.
        """
        self.exporter.export([span])


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
