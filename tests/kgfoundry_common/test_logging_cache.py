"""Tests for LoggingCache protocol and logging cache accessor function.

This module verifies that the LoggingCache protocol is properly defined,
the default implementation works correctly, and the accessor function returns
a usable cache instance.
"""

from __future__ import annotations

import logging

from kgfoundry_common.logging import (
    JsonFormatter,
    LoggingCache,
    __all__,
    get_logging_cache,
)


class TestLoggingCacheProtocol:
    """Tests for LoggingCache Protocol definition."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Test that LoggingCache protocol is @runtime_checkable."""
        cache = get_logging_cache()
        assert isinstance(cache, LoggingCache)

    def test_cache_has_get_formatter_method(self) -> None:
        """Test that cache has get_formatter method."""
        cache = get_logging_cache()
        assert hasattr(cache, "get_formatter")
        assert callable(cache.get_formatter)

    def test_cache_has_clear_method(self) -> None:
        """Test that cache has clear method."""
        cache = get_logging_cache()
        assert hasattr(cache, "clear")
        assert callable(cache.clear)


class TestGetLoggingCache:
    """Tests for get_logging_cache accessor function."""

    def test_returns_logging_cache_instance(self) -> None:
        """Test that get_logging_cache returns a LoggingCache instance."""
        cache = get_logging_cache()
        assert isinstance(cache, LoggingCache)

    def test_returns_same_instance(self) -> None:
        """Test that get_logging_cache returns the same singleton instance."""
        cache1 = get_logging_cache()
        cache2 = get_logging_cache()
        assert cache1 is cache2

    def test_function_is_exported(self) -> None:
        """Test that get_logging_cache is in __all__."""
        assert "get_logging_cache" in __all__

    def test_protocol_is_exported(self) -> None:
        """Test that LoggingCache is in __all__."""
        assert "LoggingCache" in __all__


class TestLoggingCacheGetFormatter:
    """Tests for LoggingCache.get_formatter method."""

    def test_returns_logging_formatter(self) -> None:
        """Test that get_formatter returns a logging.Formatter."""
        cache = get_logging_cache()
        formatter = cache.get_formatter()
        assert isinstance(formatter, logging.Formatter)

    def test_returns_json_formatter(self) -> None:
        """Test that get_formatter returns a JsonFormatter."""
        cache = get_logging_cache()
        formatter = cache.get_formatter()
        assert isinstance(formatter, JsonFormatter)

    def test_returns_cached_instance(self) -> None:
        """Test that get_formatter returns same instance on repeated calls."""
        cache = get_logging_cache()
        formatter1 = cache.get_formatter()
        formatter2 = cache.get_formatter()
        assert formatter1 is formatter2

    def test_formatter_can_format_records(self) -> None:
        """Test that returned formatter can format log records."""
        cache = get_logging_cache()
        formatter = cache.get_formatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert isinstance(formatted, str)
        assert "test message" in formatted.lower()


class TestLoggingCacheClear:
    """Tests for LoggingCache.clear method."""

    def test_clear_method_exists(self) -> None:
        """Test that clear method exists and is callable."""
        cache = get_logging_cache()
        assert hasattr(cache, "clear")
        assert callable(cache.clear)

    def test_clear_creates_new_formatter_on_next_call(self) -> None:
        """Test that clear resets the formatter cache."""
        cache = get_logging_cache()
        formatter1 = cache.get_formatter()
        cache.clear()
        formatter2 = cache.get_formatter()
        # After clear, a new formatter should be created
        assert formatter1 is not formatter2

    def test_clear_can_be_called_multiple_times(self) -> None:
        """Test that clear can be called safely multiple times."""
        cache = get_logging_cache()
        cache.clear()
        cache.clear()
        cache.clear()
        # Should not raise any exceptions
        formatter = cache.get_formatter()
        assert isinstance(formatter, logging.Formatter)


class TestLoggingCacheIntegration:
    """Integration tests for logging cache with actual logging."""

    def test_cache_formatter_integrates_with_logging(self) -> None:
        """Test that cached formatter works with standard logging."""
        cache = get_logging_cache()
        formatter = cache.get_formatter()

        # Create a logger and handler
        logger = logging.getLogger("test_cache")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Log a message - should not raise
        logger.info("Test message", extra={"operation": "test"})

        # Clean up
        logger.removeHandler(handler)

    def test_multiple_caches_independent(self) -> None:
        """Test that get_logging_cache always returns same instance."""
        # Get cache twice and verify they're the same
        cache_a = get_logging_cache()
        cache_b = get_logging_cache()
        assert cache_a is cache_b


class TestLoggingCacheDocstring:
    """Tests for LoggingCache protocol documentation."""

    def test_protocol_has_docstring(self) -> None:
        """Test that LoggingCache has a docstring."""
        assert LoggingCache.__doc__ is not None
        assert len(LoggingCache.__doc__) > 50

    def test_get_formatter_has_docstring(self) -> None:
        """Test that get_formatter method has documentation."""
        assert hasattr(LoggingCache, "get_formatter")
        assert LoggingCache.get_formatter.__doc__ is not None

    def test_clear_has_docstring(self) -> None:
        """Test that clear method has documentation."""
        assert hasattr(LoggingCache, "clear")
        assert LoggingCache.clear.__doc__ is not None

    def test_get_logging_cache_has_docstring(self) -> None:
        """Test that get_logging_cache has a docstring."""
        assert get_logging_cache.__doc__ is not None
        assert len(get_logging_cache.__doc__) > 50
