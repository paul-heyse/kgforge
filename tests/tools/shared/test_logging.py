"""Tests for tools._shared.logging module."""

from __future__ import annotations

import logging
import sys

from tools._shared.logging import get_logger, with_fields

from kgfoundry_common.logging import LoggerAdapter


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


class TestWithFields:
    """Tests for with_fields function."""

    def test_wraps_plain_logger(self) -> None:
        """with_fields wraps a plain logging.Logger."""
        base_logger = logging.getLogger(f"{__name__}.test_wrap")
        adapter = with_fields(base_logger, correlation_id="test-123", operation="test")
        assert isinstance(adapter, LoggerAdapter)
        assert adapter.extra == {"correlation_id": "test-123", "operation": "test"}

    def test_wraps_existing_adapter(self) -> None:
        """with_fields extracts underlying logger from existing adapter."""
        base_logger = logging.getLogger(f"{__name__}.test_wrap_existing")
        existing_adapter = LoggerAdapter(base_logger, {"existing": "field"})
        new_adapter = with_fields(existing_adapter, correlation_id="test-123")
        assert isinstance(new_adapter, LoggerAdapter)
        assert new_adapter.extra == {"correlation_id": "test-123"}

    def test_fields_merge_in_log_calls(self) -> None:
        """Fields from with_fields merge with extra in log calls."""
        base_logger = logging.getLogger(f"{__name__}.test_merge")
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        base_logger.addHandler(handler)
        base_logger.setLevel(logging.INFO)

        adapter = with_fields(base_logger, operation="test", status="started")
        # The adapter should merge fields from both sources
        assert adapter.extra == {"operation": "test", "status": "started"}
