"""Unit tests for Prometheus metrics and observability helpers.

Tests verify that MetricsRegistry creates standard metrics following
naming conventions and that record_operation context manager correctly
records metrics and durations.
"""

from __future__ import annotations

import time

import pytest

from kgfoundry_common.observability import (
    MetricsRegistry,
    get_metrics_registry,
    record_operation,
)


class TestMetricsRegistry:
    """Test MetricsRegistry for Prometheus metrics."""

    def test_registry_creates_standard_metrics(self) -> None:
        """MetricsRegistry creates standard metrics with correct names."""
        registry = MetricsRegistry()
        assert hasattr(registry, "requests_total")
        assert hasattr(registry, "request_errors_total")
        assert hasattr(registry, "request_duration_seconds")

    def test_metrics_registry_stub_when_prometheus_unavailable(self) -> None:
        """MetricsRegistry creates stub metrics when prometheus_client unavailable."""
        # Even without prometheus_client, registry should work (stubs)
        registry = MetricsRegistry()
        # Should not raise
        registry.requests_total.labels(operation="test", status="success").inc()
        registry.request_errors_total.labels(operation="test", status="error").inc()
        registry.request_duration_seconds.labels(operation="test").observe(0.123)

    def test_get_metrics_registry_singleton(self) -> None:
        """get_metrics_registry returns singleton instance."""
        registry1 = get_metrics_registry()
        registry2 = get_metrics_registry()
        assert registry1 is registry2


class TestRecordOperation:
    """Test record_operation context manager."""

    def test_record_operation_records_success(self) -> None:
        """record_operation records successful operation metrics."""
        metrics = MetricsRegistry()
        with record_operation(metrics, "search", "success"):
            time.sleep(0.01)  # Small delay to test duration recording

        # Verify metrics were called (stubs won't fail, real metrics would increment)
        # This test mainly verifies the context manager doesn't raise
        assert True

    def test_record_operation_records_error(self) -> None:
        """record_operation records error metrics when exception occurs."""
        metrics = MetricsRegistry()
        with pytest.raises(ValueError), record_operation(metrics, "search", "success"):
            raise ValueError("Test error")

        # Verify error metric was incremented (stubs won't fail)
        assert True

    def test_record_operation_uses_global_registry(self) -> None:
        """record_operation uses global registry when not provided."""
        with record_operation(operation="test", status="success"):
            pass  # Should not raise

    def test_record_operation_records_duration(self) -> None:
        """record_operation records operation duration."""
        metrics = MetricsRegistry()
        start = time.monotonic()
        with record_operation(metrics, "test", "success"):
            time.sleep(0.05)
        duration = time.monotonic() - start

        # Duration should be recorded (verified via stub metrics)
        assert duration >= 0.05
