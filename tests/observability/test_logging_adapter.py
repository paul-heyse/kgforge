"""Unit tests for structured logging adapter and correlation ID propagation.

Tests verify that LoggerAdapter injects required fields (correlation_id,
operation, status, duration_ms) and that correlation IDs propagate via
context variables for async-safe operation.
"""

from __future__ import annotations

import json
import logging
from io import StringIO

from kgfoundry_common.logging import (
    JsonFormatter,
    LoggerAdapter,
    get_correlation_id,
    get_logger,
    set_correlation_id,
    setup_logging,
)


class TestJsonFormatter:
    """Test JSON formatter for structured logging."""

    def test_format_includes_structured_fields(self) -> None:
        """JSON formatter includes correlation_id, operation, status fields."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.correlation_id = "req-123"
        record.operation = "search"
        record.status = "success"
        record.duration_ms = 42

        output = formatter.format(record)
        data = json.loads(output)

        assert data["correlation_id"] == "req-123"
        assert data["operation"] == "search"
        assert data["status"] == "success"
        assert data["duration_ms"] == 42
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"

    def test_format_includes_context_correlation_id(self) -> None:
        """JSON formatter injects correlation_id from context if not in record."""
        formatter = JsonFormatter()
        set_correlation_id("ctx-456")
        try:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Test",
                args=(),
                exc_info=None,
            )
            output = formatter.format(record)
            data = json.loads(output)
            assert data["correlation_id"] == "ctx-456"
        finally:
            set_correlation_id(None)


class TestLoggerAdapter:
    """Test LoggerAdapter structured field injection."""

    def test_process_injects_correlation_id_from_context(self) -> None:
        """LoggerAdapter injects correlation_id from context variables."""
        adapter = LoggerAdapter(logging.getLogger("test"), {})
        set_correlation_id("req-789")
        try:
            msg, kwargs = adapter.process("Test message", {})
            assert kwargs["extra"]["correlation_id"] == "req-789"
        finally:
            set_correlation_id(None)

    def test_process_injects_operation_and_status_defaults(self) -> None:
        """LoggerAdapter injects default operation and status if missing."""
        adapter = LoggerAdapter(logging.getLogger("test"), {})
        msg, kwargs = adapter.process("Test", {})
        extra = kwargs["extra"]
        assert extra["operation"] == "unknown"
        assert extra["status"] == "success"

    def test_process_infers_status_from_level(self) -> None:
        """LoggerAdapter infers status from log level when not provided."""
        adapter = LoggerAdapter(logging.getLogger("test"), {})
        msg, kwargs = adapter.process("Error", {"level": logging.ERROR})
        assert kwargs["extra"]["status"] == "error"

        msg, kwargs = adapter.process("Warning", {"level": logging.WARNING})
        assert kwargs["extra"]["status"] == "warning"

        msg, kwargs = adapter.process("Info", {"level": logging.INFO})
        assert kwargs["extra"]["status"] == "success"


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_adds_null_handler(self) -> None:
        """get_logger adds NullHandler to prevent duplicate handlers."""
        logger_adapter = get_logger("test.module")
        logger = logger_adapter.logger  # Access underlying logger
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.NullHandler)

    def test_get_logger_returns_adapter(self) -> None:
        """get_logger returns LoggerAdapter instance."""
        adapter = get_logger("test.module")
        assert isinstance(adapter, LoggerAdapter)


class TestCorrelationIdContext:
    """Test correlation ID context variable propagation."""

    def test_set_get_correlation_id(self) -> None:
        """set_correlation_id and get_correlation_id work correctly."""
        set_correlation_id("test-id-123")
        try:
            assert get_correlation_id() == "test-id-123"
        finally:
            set_correlation_id(None)
            assert get_correlation_id() is None

    def test_correlation_id_in_logs(self) -> None:
        """Correlation ID from context appears in log output."""
        set_correlation_id("ctx-correlation")
        try:
            logger = get_logger("test")
            handler = logging.StreamHandler(StringIO())
            handler.setFormatter(JsonFormatter())
            logger.logger.addHandler(handler)
            logger.logger.setLevel(logging.INFO)

            logger.info("Test message")

            output = handler.stream.getvalue()  # type: ignore[attr-defined]
            data = json.loads(output)
            assert data["correlation_id"] == "ctx-correlation"
        finally:
            set_correlation_id(None)


class TestSetupLogging:
    """Test setup_logging function."""

    def test_setup_logging_configures_json_formatter(self) -> None:
        """setup_logging configures root logger with JSON formatter."""
        setup_logging(level=logging.DEBUG)
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0
        handler = root_logger.handlers[0]
        assert isinstance(handler.formatter, JsonFormatter)
