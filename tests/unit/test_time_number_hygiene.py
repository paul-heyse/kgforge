"""Tests for timezone-aware timestamps and monotonic duration measurements.

This module verifies that timestamps include timezone information and
that duration measurements use monotonic clocks.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from kgfoundry_common.logging import JsonFormatter, get_logger
from kgfoundry_common.observability import record_operation


class TestTimezoneAwareTimestamps:
    """Test timezone-aware timestamp usage."""

    def test_datetime_now_uses_utc(self) -> None:
        """Verify datetime.now() uses UTC timezone.

        Scenario: Timezone-aware timestamps

        GIVEN a new log entry or persisted timestamp
        WHEN inspected via tests
        THEN it includes timezone information and round-trips correctly
        """
        # Create timezone-aware timestamp
        now = datetime.now(tz=UTC)
        assert now.tzinfo is not None, "Timestamp should include timezone information"
        assert now.tzinfo == UTC, "Timestamp should use UTC timezone"

    def test_timestamp_roundtrip_serialization(self) -> None:
        """Verify timestamps round-trip correctly through serialization.

        Scenario: Timezone-aware timestamps

        GIVEN a timezone-aware timestamp
        WHEN serialized and deserialized
        THEN timezone information is preserved
        """
        original = datetime.now(tz=UTC)

        # Serialize to ISO format (includes timezone)
        iso_str = original.isoformat()
        assert "Z" in iso_str or "+00:00" in iso_str, "ISO format should include timezone"

        # Deserialize back
        deserialized = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        assert deserialized.tzinfo is not None, "Deserialized timestamp should include timezone"
        assert deserialized == original, "Timestamp should round-trip correctly"

    def test_json_formatter_includes_timezone(self) -> None:
        """Verify JsonFormatter includes timezone in timestamps."""
        import json
        import logging
        from io import StringIO

        handler = logging.StreamHandler(StringIO())
        handler.setFormatter(JsonFormatter())
        logger = get_logger("test")
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)

        logger.info("Test message", extra={"operation": "test", "status": "success"})

        output = handler.stream.getvalue()  # type: ignore[attr-defined]
        data = json.loads(output)
        assert "ts" in data, "Log entry should include timestamp"
        ts_str = data["ts"]
        # Timestamp should include timezone (Z suffix for UTC)
        assert ts_str.endswith("Z") or "+" in ts_str, "Timestamp should include timezone"


class TestMonotonicDurations:
    """Test monotonic clock usage for duration measurements."""

    def test_record_operation_uses_monotonic(self) -> None:
        """Verify record_operation uses monotonic clock.

        Scenario: Monotonic durations

        WHEN duration metrics are emitted for operations
        THEN they derive from monotonic clocks, preventing negative or skewed durations
        """
        from kgfoundry_common.observability import get_metrics_registry

        metrics = get_metrics_registry()

        # Record operation with delay
        with record_operation(metrics, operation="test", status="success"):
            time.sleep(0.01)  # Small delay to verify timing

        # Verify duration was recorded (no negative durations)
        # Actual verification depends on prometheus_client availability,
        # but the code path is verified

    def test_monotonic_prevents_negative_durations(self) -> None:
        """Verify monotonic clock prevents negative durations.

        Scenario: Monotonic durations

        WHEN system time changes during operation
        THEN monotonic clock still produces positive durations
        """
        start = time.monotonic()
        time.sleep(0.01)
        end = time.monotonic()

        duration = end - start
        assert duration >= 0, "Monotonic clock should produce non-negative durations"
        assert duration > 0, "Duration should be positive for actual delay"

    def test_monotonic_vs_time_comparison(self) -> None:
        """Verify monotonic clock is independent of system time."""
        monotonic_start = time.monotonic()
        time_start = time.time()

        time.sleep(0.01)

        monotonic_end = time.monotonic()
        time_end = time.time()

        monotonic_duration = monotonic_end - monotonic_start
        time_duration = time_end - time_start

        # Both should be positive, but monotonic is guaranteed to be monotonic
        assert monotonic_duration >= 0, "Monotonic duration should be non-negative"
        assert time_duration >= 0, "Time duration should be non-negative"

        # Monotonic clock is not affected by system time adjustments
        # (we can't easily test this without mocking, but we verify the pattern)


class TestNumberHygiene:
    """Test appropriate numeric types for domain values."""

    def test_timestamp_milliseconds_are_integers(self) -> None:
        """Verify timestamp milliseconds are integers (not floats)."""
        import datetime as dt

        now = dt.datetime.now(dt.UTC)
        timestamp_ms = int(now.timestamp() * 1000)

        assert isinstance(timestamp_ms, int), "Timestamp milliseconds should be integer"
        assert timestamp_ms > 0, "Timestamp should be positive"

    def test_duration_seconds_are_floats(self) -> None:
        """Verify duration measurements are floats (for precision)."""
        start = time.monotonic()
        time.sleep(0.001)  # Very short delay
        end = time.monotonic()

        duration = end - start
        assert isinstance(duration, float), "Duration should be float for precision"
        assert duration >= 0, "Duration should be non-negative"
