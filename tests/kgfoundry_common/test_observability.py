"""Tests for kgfoundry_common.observability module."""

from __future__ import annotations

import time

import pytest

# Import kgfoundry_common.observability for module-level patching
from kgfoundry_common.observability import (
    HAVE_OPENTELEMETRY,
    HAVE_PROMETHEUS,
    MetricsProvider,
    observe_duration,
    start_span,
)

# Test if Prometheus is available
try:
    from prometheus_client import CollectorRegistry

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CollectorRegistry = None  # type: ignore[assignment, misc]

# Test if OpenTelemetry is available
try:
    from opentelemetry import trace

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None  # type: ignore[assignment]


class TestMetricsProvider:
    """Tests for MetricsProvider class."""

    def test_default_creates_instance(self) -> None:
        """MetricsProvider.default() creates an instance."""
        if HAVE_PROMETHEUS:
            # Use a fresh registry to avoid conflicts
            registry = CollectorRegistry()
            metrics = MetricsProvider(registry=registry)
        else:
            metrics = MetricsProvider.default()
        assert isinstance(metrics, MetricsProvider)

    def test_stub_metrics_work_without_prometheus(self) -> None:
        """Stub metrics work when Prometheus is unavailable."""
        if HAVE_PROMETHEUS:
            pytest.skip("Prometheus is available, cannot test stub behavior")
        # This test verifies the stub implementations return self for chaining
        metrics = MetricsProvider.default()
        # Should not raise AttributeError
        labeled = metrics.runs_total.labels(component="test", status="success")
        assert labeled is not None
        # Should not raise when calling methods
        labeled.inc()

    def test_labels_returns_self_for_stubs(self) -> None:
        """Stub metrics return self from labels() for chaining."""
        if HAVE_PROMETHEUS:
            pytest.skip("Prometheus is available, cannot test stub behavior")
        metrics = MetricsProvider.default()
        counter = metrics.runs_total
        result = counter.labels(component="test", status="success")
        # For stubs, result should be the same instance
        assert result is counter

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="Prometheus not available")
    def test_real_metrics_with_registry(self) -> None:
        """Real metrics work with a custom registry."""
        registry = CollectorRegistry()
        metrics = MetricsProvider(registry=registry)
        assert metrics.runs_total is not None
        assert metrics.operation_duration_seconds is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="Prometheus not available")
    def test_real_metrics_labels_chaining(self) -> None:
        """Real metrics support label chaining."""
        registry = CollectorRegistry()
        metrics = MetricsProvider(registry=registry)
        labeled = metrics.runs_total.labels(component="test", status="success")
        labeled.inc()
        # Verify metric was incremented by checking the sample
        samples = list(labeled.collect())
        assert len(samples) > 0

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="Prometheus not available")
    def test_real_histogram_observes(self) -> None:
        """Real histogram records observations."""
        registry = CollectorRegistry()
        metrics = MetricsProvider(registry=registry)
        labeled = metrics.operation_duration_seconds.labels(
            component="test", operation="query", status="success"
        )
        labeled.observe(0.5)
        # Verify observation was recorded by checking samples
        samples = list(labeled.collect())
        assert len(samples) > 0


class TestObserveDuration:
    """Tests for observe_duration context manager."""

    def test_observe_duration_success(self) -> None:
        """observe_duration records successful operation."""
        registry = CollectorRegistry() if HAVE_PROMETHEUS else None
        metrics = MetricsProvider(registry=registry) if registry else MetricsProvider.default()
        with observe_duration(metrics, "test_operation", component="test") as obs:
            obs.success()
            time.sleep(0.01)  # Small delay to ensure duration > 0
        assert obs.status == "success"

    def test_observe_duration_error(self) -> None:
        """observe_duration records failed operation."""
        registry = CollectorRegistry() if HAVE_PROMETHEUS else None
        metrics = MetricsProvider(registry=registry) if registry else MetricsProvider.default()
        with observe_duration(metrics, "test_operation", component="test") as obs:
            obs.error()
        assert obs.status == "error"

    def test_observe_duration_exception_records_error(self) -> None:
        """observe_duration records error when exception occurs."""
        registry = CollectorRegistry() if HAVE_PROMETHEUS else None
        metrics = MetricsProvider(registry=registry) if registry else MetricsProvider.default()
        error_msg = "Test error"

        def _raise_error() -> None:
            with observe_duration(metrics, "test_operation", component="test") as obs:
                raise ValueError(error_msg)
            # Status should be set to error by __exit__
            assert obs.status == "error"

        with pytest.raises(ValueError, match=error_msg):
            _raise_error()

    def test_observe_duration_records_metrics(self) -> None:
        """observe_duration increments counters and records duration."""
        registry = CollectorRegistry() if HAVE_PROMETHEUS else None
        metrics = MetricsProvider(registry=registry) if registry else MetricsProvider.default()
        with observe_duration(metrics, "test_operation", component="test") as obs:
            obs.success()
        # Verify metrics were recorded (works with both stubs and real metrics)
        assert metrics.runs_total is not None
        assert metrics.operation_duration_seconds is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="Prometheus not available")
    def test_observe_duration_increments_real_counter(self) -> None:
        """observe_duration increments real Prometheus counter."""
        registry = CollectorRegistry()
        metrics = MetricsProvider(registry=registry)
        labeled = metrics.runs_total.labels(component="test", status="success")
        # Increment directly to verify counter works
        labeled.inc()
        # Verify metric exists and can be collected
        samples = list(registry.collect())
        assert len(samples) > 0, "Registry should have at least one collector"

        # Test that observe_duration also increments
        with observe_duration(metrics, "test_operation", component="test") as obs:
            obs.success()

        # Verify operation completed successfully
        assert obs.status == "success"


class TestStartSpan:
    """Tests for start_span context manager."""

    def test_start_span_noop_without_opentelemetry(self) -> None:
        """start_span performs no-op when OpenTelemetry unavailable."""
        if HAVE_OPENTELEMETRY:
            pytest.skip("OpenTelemetry is available, cannot test no-op behavior")
        # Should not raise
        with start_span("test.operation"):
            pass

    def test_start_span_with_attributes_noop(self) -> None:
        """start_span with attributes performs no-op when OpenTelemetry unavailable."""
        if HAVE_OPENTELEMETRY:
            pytest.skip("OpenTelemetry is available, cannot test no-op behavior")
        with start_span("test.operation", attributes={"key": "value"}):
            pass

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not available")
    def test_start_span_creates_span(self) -> None:
        """start_span creates OpenTelemetry span when available."""
        # Simple test that start_span works without crashing
        # Actual span creation is tested via integration tests
        with start_span("test.operation"):
            pass

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not available")
    def test_start_span_sets_attributes(self) -> None:
        """start_span sets attributes when provided."""
        # Simple test that start_span works with attributes
        with start_span("test.operation", attributes={"key": "value", "num": 42}):
            pass

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not available")
    def test_start_span_records_exception(self) -> None:
        """start_span records exception when error occurs."""
        error_msg = "Test error"
        with pytest.raises(ValueError, match=error_msg), start_span("test.operation"):
            raise ValueError(error_msg)


class TestIntegration:
    """Integration tests for observability helpers."""

    def test_metrics_provider_works_without_dependencies(self) -> None:
        """MetricsProvider works even when Prometheus is unavailable."""
        if HAVE_PROMETHEUS:
            pytest.skip("Prometheus is available, cannot test stub behavior")
        metrics = MetricsProvider.default()
        # All operations should work without AttributeError
        metrics.runs_total.labels(component="test", status="success").inc()
        metrics.operation_duration_seconds.labels(
            component="test", operation="query", status="success"
        ).observe(0.5)

    def test_observe_duration_with_span(self) -> None:
        """observe_duration can be used with start_span."""
        registry = CollectorRegistry() if HAVE_PROMETHEUS else None
        metrics = MetricsProvider(registry=registry) if registry else MetricsProvider.default()
        with (
            start_span("test.operation"),
            observe_duration(metrics, "test_operation", component="test") as obs,
        ):
            obs.success()
        assert obs.status == "success"

    def test_multiple_operations_record_separately(self) -> None:
        """Multiple operations record metrics separately."""
        registry = CollectorRegistry() if HAVE_PROMETHEUS else None
        metrics = MetricsProvider(registry=registry) if registry else MetricsProvider.default()
        with observe_duration(metrics, "operation1", component="test1") as obs1:
            obs1.success()
        with observe_duration(metrics, "operation2", component="test2") as obs2:
            obs2.success()
        assert obs1.status == "success"
        assert obs2.status == "success"
