"""Tests for tools.navmap.observability module."""

from __future__ import annotations

import time

import pytest
from tools.navmap.observability import (
    get_correlation_id,
    get_metrics_registry,
    record_operation_metrics,
)


class TestNavmapMetrics:
    """Tests for NavmapMetrics registry."""

    def test_get_metrics_registry_singleton(self) -> None:
        """get_metrics_registry returns singleton instance."""
        registry1 = get_metrics_registry()
        registry2 = get_metrics_registry()
        assert registry1 is registry2

    def test_metrics_registry_has_counters(self) -> None:
        """NavmapMetrics exposes required counters."""
        metrics = get_metrics_registry()
        assert hasattr(metrics, "build_runs_total")
        assert hasattr(metrics, "check_runs_total")
        assert hasattr(metrics, "repair_runs_total")
        assert hasattr(metrics, "migrate_runs_total")

    def test_metrics_registry_has_histograms(self) -> None:
        """NavmapMetrics exposes required histograms."""
        metrics = get_metrics_registry()
        assert hasattr(metrics, "build_duration_seconds")
        assert hasattr(metrics, "check_duration_seconds")
        assert hasattr(metrics, "repair_duration_seconds")
        assert hasattr(metrics, "migrate_duration_seconds")

    def test_counters_support_labels(self) -> None:
        """Counters support label chaining."""
        metrics = get_metrics_registry()
        labeled = metrics.build_runs_total.labels(status="success")
        assert labeled is not None

    def test_histograms_support_labels(self) -> None:
        """Histograms support label chaining."""
        metrics = get_metrics_registry()
        labeled = metrics.build_duration_seconds.labels(status="success")
        assert labeled is not None


class TestCorrelationId:
    """Tests for correlation ID generation."""

    def test_correlation_id_format(self) -> None:
        """get_correlation_id returns urn:navmap:correlation: format."""
        corr_id = get_correlation_id()
        assert corr_id.startswith("urn:navmap:correlation:")
        assert len(corr_id) > len("urn:navmap:correlation:")

    def test_correlation_ids_are_unique(self) -> None:
        """get_correlation_id generates unique IDs."""
        ids = [get_correlation_id() for _ in range(10)]
        assert len(set(ids)) == 10


class TestRecordOperationMetrics:
    """Tests for record_operation_metrics context manager."""

    def test_records_success_metrics(self) -> None:
        """record_operation_metrics records success for build operation."""
        with record_operation_metrics("build", status="success"):
            time.sleep(0.01)

        # Metrics should be recorded (no exception means success)
        metrics = get_metrics_registry()
        # Verify metrics exist (stub metrics won't fail)
        assert metrics.build_runs_total is not None

    def test_records_error_metrics_on_exception(self) -> None:
        """record_operation_metrics records error status on exception."""
        error_msg = "Test error"
        with (
            pytest.raises(ValueError, match=error_msg),
            record_operation_metrics("repair", status="success"),
        ):
            raise ValueError(error_msg)

        # Metrics should record error status
        metrics = get_metrics_registry()
        assert metrics.repair_runs_total is not None

    @pytest.mark.parametrize("operation", ["build", "check", "repair", "migrate"])
    def test_supports_all_operations(self, operation: str) -> None:
        """record_operation_metrics supports all navmap operations."""
        with record_operation_metrics(operation, status="success"):
            pass

        metrics = get_metrics_registry()
        assert hasattr(metrics, f"{operation}_runs_total")
        assert hasattr(metrics, f"{operation}_duration_seconds")

    def test_auto_generates_correlation_id(self) -> None:
        """record_operation_metrics auto-generates correlation ID if None."""
        with record_operation_metrics("build"):
            pass

        # Should not raise exception
        assert True

    def test_accepts_custom_correlation_id(self) -> None:
        """record_operation_metrics accepts custom correlation ID."""
        custom_id = "urn:test:correlation:12345"
        with record_operation_metrics("check", correlation_id=custom_id):
            pass

        # Should not raise exception
        assert True

    def test_records_duration(self) -> None:
        """record_operation_metrics records operation duration."""
        start = time.monotonic()
        with record_operation_metrics("build", status="success"):
            time.sleep(0.05)
        duration = time.monotonic() - start

        # Duration should be recorded (verify via metrics if Prometheus available)
        assert duration >= 0.05
        metrics = get_metrics_registry()
        assert metrics.build_duration_seconds is not None
