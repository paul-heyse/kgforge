"""Tests for structured logging with correlation IDs and observability."""

from __future__ import annotations

import json
import logging
from io import StringIO
from typing import cast

import pytest

from kgfoundry_common.logging import (
    CorrelationContext,
    JsonFormatter,
    LogContextExtra,
    LoggerAdapter,
    get_correlation_id,
    get_logger,
    set_correlation_id,
    with_fields,
)


def test_format_with_structured_fields() -> None:
    """Test JSON formatter includes structured fields."""
    formatter = JsonFormatter()
    handler = logging.StreamHandler(StringIO())
    logger = logging.getLogger("test")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Create a log record with structured fields
    record = logger.makeRecord(
        "test",
        logging.INFO,
        __file__,
        42,
        "Test message",
        (),
        None,
        extra={"operation": "test_op", "status": "success"},
    )
    formatted = formatter.format(record)

    # Parse JSON and verify fields
    data = json.loads(formatted)
    assert data["message"] == "Test message"
    assert data["operation"] == "test_op"
    assert data["status"] == "success"
    assert data["level"] == "INFO"


def test_correlation_id_from_context() -> None:
    """Test correlation ID is injected from context."""
    formatter = JsonFormatter()
    logger = logging.getLogger("test_context")
    logger.handlers.clear()

    set_correlation_id("req-123")
    try:
        record = logger.makeRecord(
            "test_context",
            logging.INFO,
            __file__,
            42,
            "Test",
            (),
            None,
        )
        formatted = formatter.format(record)
        data = json.loads(formatted)
        assert data["correlation_id"] == "req-123"
    finally:
        set_correlation_id(None)


def test_get_logger_attaches_null_handler() -> None:
    """Test that get_logger attaches a NullHandler."""
    logger = get_logger("test.module")
    assert isinstance(logger, LoggerAdapter)
    # Get the underlying logger
    base_logger = logger.logger
    assert any(isinstance(h, logging.NullHandler) for h in base_logger.handlers)


def test_log_success_method(caplog: pytest.LogCaptureFixture) -> None:
    """Test log_success helper method."""
    logger = get_logger("test.success")
    caplog.set_level(logging.INFO)

    logger.log_success(
        "Operation completed",
        operation="build_index",
        duration_ms=1234.5,
    )

    assert len(caplog.records) == 1
    record = cast(LogContextExtra, caplog.records[0].__dict__)
    assert record["status"] == "success"
    assert record["operation"] == "build_index"


def test_log_failure_method(caplog: pytest.LogCaptureFixture) -> None:
    """Test log_failure helper method."""
    logger = get_logger("test.failure")
    caplog.set_level(logging.ERROR)

    exc = ValueError("Test error")
    logger.log_failure(
        "Operation failed",
        exception=exc,
        operation="save_data",
    )

    assert len(caplog.records) == 1
    record = cast(LogContextExtra, caplog.records[0].__dict__)
    assert record["status"] == "error"
    assert record["operation"] == "save_data"


def test_log_io_method(caplog: pytest.LogCaptureFixture) -> None:
    """Test log_io helper method."""
    logger = get_logger("test.io")
    caplog.set_level(logging.INFO)

    logger.log_io(
        "I/O operation completed",
        operation="read_file",
        duration_ms=500.0,
        input_bytes=1024,
        output_bytes=2048,
    )

    assert len(caplog.records) == 1
    record = cast(LogContextExtra, caplog.records[0].__dict__)
    assert record["duration_ms"] == 500.0


@pytest.mark.parametrize(
    ("log_level", "expected_status"),
    [
        (logging.INFO, "success"),
        (logging.WARNING, "warning"),
        (logging.ERROR, "error"),
    ],
)
def test_status_inferred_from_level(
    log_level: int,
    expected_status: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that status is inferred from log level."""
    logger = get_logger("test.status")
    caplog.set_level(log_level)

    logger.log(log_level, "Test message")

    assert len(caplog.records) == 1
    record = cast(LogContextExtra, caplog.records[0].__dict__)
    assert record["status"] == expected_status


def test_correlation_context_sets_and_clears_id() -> None:
    """Test that CorrelationContext sets and clears correlation ID."""
    assert get_correlation_id() is None

    with CorrelationContext(correlation_id="req-456"):
        assert get_correlation_id() == "req-456"

    assert get_correlation_id() is None


def test_correlation_context_nesting() -> None:
    """Test that CorrelationContext nesting works correctly."""
    with CorrelationContext(correlation_id="outer"):
        assert get_correlation_id() == "outer"

        with CorrelationContext(correlation_id="inner"):
            assert get_correlation_id() == "inner"

        assert get_correlation_id() == "outer"

    assert get_correlation_id() is None


def test_with_fields_context_manager(caplog: pytest.LogCaptureFixture) -> None:
    """Test with_fields context manager for temporary field injection."""
    logger = get_logger("test.fields")
    caplog.set_level(logging.INFO)

    with with_fields(logger, correlation_id="req789", operation="search") as adapter:
        adapter.info("Inside context")

    assert len(caplog.records) == 1
    record = cast(LogContextExtra, caplog.records[0].__dict__)
    assert record["correlation_id"] == "req789"
    assert record["operation"] == "search"


def test_correlation_id_propagates_through_logger(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that correlation ID from context is automatically injected."""
    logger = get_logger("test.propagate")
    caplog.set_level(logging.INFO)

    set_correlation_id("req-prop-123")
    try:
        logger.info("Test message")
        record = cast(LogContextExtra, caplog.records[0].__dict__)
        assert record["correlation_id"] == "req-prop-123"
    finally:
        set_correlation_id(None)


def test_adapter_merges_extra_fields(caplog: pytest.LogCaptureFixture) -> None:
    """Test that adapter merges bound fields with runtime extra."""
    logger = get_logger("test.merge")
    caplog.set_level(logging.INFO)

    logger.info("Test", extra={"service": "api", "endpoint": "/search"})

    record = cast(LogContextExtra, caplog.records[0].__dict__)
    assert record["service"] == "api"
    assert record["endpoint"] == "/search"
