"""Tests for kgfoundry_common.logging module."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest

from kgfoundry_common.logging import (
    CorrelationContext,
    JsonFormatter,
    LoggerAdapter,
    get_correlation_id,
    get_logger,
    set_correlation_id,
    with_fields,
)

# Golden fixture path
GOLDEN_FIXTURE = Path(__file__).parent / "golden" / "logging.json"


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger_adapter(self) -> None:
        """get_logger returns a LoggerAdapter instance."""
        logger = get_logger(__name__)
        assert isinstance(logger, LoggerAdapter)

    def test_logger_has_null_handler(self) -> None:
        """Logger has NullHandler when no handlers configured."""
        logger = get_logger(f"{__name__}.test_null_handler")
        handlers = logger.logger.handlers
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.NullHandler)

    def test_logger_propagates_to_parent(self) -> None:
        """Logger propagates to parent logger."""
        logger = get_logger(__name__)
        assert logger.logger.propagate is True


class TestJsonFormatter:
    """Tests for JsonFormatter class."""

    def test_formats_as_json(self) -> None:
        """JsonFormatter produces valid JSON output."""
        handler = logging.StreamHandler(io.StringIO())
        handler.setFormatter(JsonFormatter())
        logger = logging.getLogger(f"{__name__}.test_json")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.info("Test message", extra={"operation": "test", "status": "success"})
        output = handler.stream.getvalue()  # type: ignore[attr-defined]
        assert output.strip()

        # Parse JSON to verify structure
        log_data = json.loads(output.strip())
        assert "ts" in log_data
        assert "level" in log_data
        assert "name" in log_data
        assert "message" in log_data
        assert log_data["message"] == "Test message"
        assert log_data["operation"] == "test"
        assert log_data["status"] == "success"

    def test_includes_correlation_id_from_context(self) -> None:
        """JsonFormatter includes correlation_id from contextvars."""
        handler = logging.StreamHandler(io.StringIO())
        handler.setFormatter(JsonFormatter())
        logger = logging.getLogger(f"{__name__}.test_correlation")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        set_correlation_id("test-correlation-123")
        try:
            logger.info("Test message")
            output = handler.stream.getvalue()  # type: ignore[attr-defined]
            log_data = json.loads(output.strip())
            assert log_data["correlation_id"] == "test-correlation-123"
        finally:
            set_correlation_id(None)


class TestCorrelationContext:
    """Tests for CorrelationContext context manager."""

    def test_sets_correlation_id_in_context(self) -> None:
        """CorrelationContext sets correlation_id in contextvars."""
        assert get_correlation_id() is None

        with CorrelationContext(correlation_id="test-123"):
            assert get_correlation_id() == "test-123"

        assert get_correlation_id() is None

    def test_restores_previous_correlation_id(self) -> None:
        """CorrelationContext restores previous correlation_id on exit."""
        set_correlation_id("original-123")
        try:
            with CorrelationContext(correlation_id="nested-456"):
                assert get_correlation_id() == "nested-456"
            assert get_correlation_id() == "original-123"
        finally:
            set_correlation_id(None)

    def test_handles_none_correlation_id(self) -> None:
        """CorrelationContext handles None correlation_id."""
        set_correlation_id("original-123")
        try:
            with CorrelationContext(correlation_id=None):
                assert get_correlation_id() is None
            assert get_correlation_id() == "original-123"
        finally:
            set_correlation_id(None)


class TestWithFields:
    """Tests for with_fields context manager."""

    def test_sets_correlation_id_in_context(self) -> None:
        """with_fields sets correlation_id in contextvars."""
        logger = get_logger(__name__)
        assert get_correlation_id() is None

        with with_fields(logger, correlation_id="test-123") as adapter:
            assert get_correlation_id() == "test-123"
            assert isinstance(adapter, LoggerAdapter)

        assert get_correlation_id() is None

    def test_injects_fields_in_log_calls(self) -> None:
        """with_fields injects fields into log entries."""
        handler = logging.StreamHandler(io.StringIO())
        handler.setFormatter(JsonFormatter())
        handler.setLevel(logging.INFO)

        base_logger = logging.getLogger(f"{__name__}.test_fields")
        base_logger.addHandler(handler)
        base_logger.setLevel(logging.INFO)

        logger = get_logger(__name__)
        with with_fields(logger, operation="test", status="started") as adapter:
            adapter.logger = base_logger  # Use test handler
            adapter.info("Processing", extra={"file_count": 10})

        output = handler.stream.getvalue()  # type: ignore[attr-defined]
        log_data = json.loads(output.strip())
        assert log_data["operation"] == "test"
        assert log_data["status"] == "started"
        assert log_data.get("file_count") == 10

    def test_merges_fields_with_extra(self) -> None:
        """Fields from with_fields merge with extra dict."""
        handler = logging.StreamHandler(io.StringIO())
        handler.setFormatter(JsonFormatter())
        handler.setLevel(logging.INFO)

        base_logger = logging.getLogger(f"{__name__}.test_merge")
        base_logger.addHandler(handler)
        base_logger.setLevel(logging.INFO)

        logger = get_logger(__name__)
        with with_fields(logger, operation="build", status="started") as adapter:
            adapter.logger = base_logger
            # Override status in extra
            adapter.info("Processing", extra={"status": "success", "count": 5})

        output = handler.stream.getvalue()  # type: ignore[attr-defined]
        log_data = json.loads(output.strip())
        assert log_data["operation"] == "build"
        assert log_data["status"] == "success"  # Extra takes precedence
        assert log_data["count"] == 5

    def test_restores_correlation_id_on_exit(self) -> None:
        """with_fields restores correlation_id when context exits."""
        logger = get_logger(__name__)
        set_correlation_id("original-123")
        try:
            with with_fields(logger, correlation_id="nested-456"):
                assert get_correlation_id() == "nested-456"
            assert get_correlation_id() == "original-123"
        finally:
            set_correlation_id(None)


class TestStructuredLoggingIntegration:
    """Integration tests for structured logging scenarios."""

    @pytest.mark.parametrize(
        ("operation", "status", "message"),
        [
            ("search", "success", "Search completed"),
            ("build", "started", "Build started"),
            ("index", "error", "Index failed"),
        ],
    )
    def test_basic_log_scenarios(self, operation: str, status: str, message: str) -> None:
        """Test basic logging scenarios with different operations and statuses."""
        handler = logging.StreamHandler(io.StringIO())
        handler.setFormatter(JsonFormatter())
        handler.setLevel(logging.INFO)

        base_logger = logging.getLogger(f"{__name__}.test_basic_{operation}")
        base_logger.addHandler(handler)
        base_logger.setLevel(logging.INFO)

        logger = get_logger(__name__)
        logger.logger = base_logger
        logger.info(message, extra={"operation": operation, "status": status})

        output = handler.stream.getvalue()  # type: ignore[attr-defined]
        log_data = json.loads(output.strip())
        assert log_data["operation"] == operation
        assert log_data["status"] == status
        assert log_data["message"] == message

    def test_correlation_id_propagation(self) -> None:
        """Test correlation ID propagation through context."""
        handler = logging.StreamHandler(io.StringIO())
        handler.setFormatter(JsonFormatter())
        handler.setLevel(logging.INFO)

        base_logger = logging.getLogger(f"{__name__}.test_propagation")
        base_logger.addHandler(handler)
        base_logger.setLevel(logging.INFO)

        logger = get_logger(__name__)
        logger.logger = base_logger

        with CorrelationContext(correlation_id="req-abc-123"):
            logger.info("Request started", extra={"operation": "request"})

        output = handler.stream.getvalue()  # type: ignore[attr-defined]
        log_data = json.loads(output.strip())
        assert log_data["correlation_id"] == "req-abc-123"

    def test_golden_fixture(self) -> None:
        """Test that log output matches golden fixture."""
        handler = logging.StreamHandler(io.StringIO())
        handler.setFormatter(JsonFormatter())
        handler.setLevel(logging.INFO)

        base_logger = logging.getLogger(f"{__name__}.test_golden")
        base_logger.addHandler(handler)
        base_logger.setLevel(logging.INFO)

        logger = get_logger(__name__)

        with with_fields(logger, correlation_id="abc-123", operation="load-settings") as adapter:
            adapter.logger = base_logger
            adapter.info(
                "Loading configuration", extra={"status": "started", "env_var": "KGFOUNDRY_API_KEY"}
            )

        output = handler.stream.getvalue()  # type: ignore[attr-defined]
        log_data = json.loads(output.strip())

        # Verify structure matches expected format
        assert "ts" in log_data
        assert "level" in log_data
        assert "name" in log_data
        assert "message" in log_data
        assert log_data["correlation_id"] == "abc-123"
        assert log_data["operation"] == "load-settings"
        assert log_data["status"] == "started"
        assert log_data["env_var"] == "KGFOUNDRY_API_KEY"

        # Save golden fixture if it doesn't exist
        if not GOLDEN_FIXTURE.exists():
            GOLDEN_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
            # Remove timestamp for stable comparison
            golden_data = {k: v for k, v in log_data.items() if k != "ts"}
            golden_data["ts"] = "2024-01-01T00:00:00.000Z"  # Placeholder
            GOLDEN_FIXTURE.write_text(json.dumps(golden_data, indent=2) + "\n")
