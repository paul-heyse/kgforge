"""Micro-benchmark tests for docstring builder performance.

These tests verify that refactored code maintains acceptable runtime performance.
Run with pytest -m benchmark (non-gating).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from tools.docstring_builder.config import load_config_with_selection
from tools.docstring_builder.harvest import harvest_file
from tools.docstring_builder.observability import record_operation_metrics

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.benchmark
def test_harvest_performance_benchmark() -> None:
    """Verify harvest operation completes within acceptable time budget.

    This test ensures that refactored harvest code maintains performance
    characteristics. The budget is intentionally generous to avoid flakiness
    while still catching significant regressions.
    """
    config, _ = load_config_with_selection()
    test_file = REPO_ROOT / "tools" / "docstring_builder" / "models.py"

    # Warmup run
    harvest_file(test_file, config, REPO_ROOT)

    # Timed run
    start = time.perf_counter()
    with record_operation_metrics("harvest", status="success"):
        harvest_file(test_file, config, REPO_ROOT)
    duration = time.perf_counter() - start

    # Budget: 5 seconds for a single file harvest (generous)
    assert duration < 5.0, f"Harvest took {duration:.3f}s, exceeds 5s budget"


@pytest.mark.benchmark
def test_observability_overhead() -> None:
    """Verify observability instrumentation adds minimal overhead.

    This test ensures that metrics recording and correlation ID generation
    do not significantly impact performance.
    """
    from tools.docstring_builder.observability import (  # noqa: PLC0415
        get_correlation_id,
        record_operation_metrics,
    )

    # Baseline: correlation ID generation
    start = time.perf_counter()
    for _ in range(1000):
        get_correlation_id()
    baseline_duration = time.perf_counter() - start

    # With metrics: context manager overhead
    start = time.perf_counter()
    for _ in range(1000):
        with record_operation_metrics("harvest", status="success"):
            pass
    overhead_duration = time.perf_counter() - start

    # Overhead should be < 10ms per operation
    per_operation_overhead = (overhead_duration - baseline_duration) / 1000
    assert per_operation_overhead < 0.01, (
        f"Observability overhead {per_operation_overhead * 1000:.3f}ms per operation exceeds 10ms"
    )


@pytest.mark.benchmark
def test_metrics_registry_initialization() -> None:
    """Verify metrics registry initialization is fast.

    This test ensures that Prometheus metric initialization does not add
    significant startup overhead.
    """
    from tools.docstring_builder.observability import get_metrics_registry  # noqa: PLC0415

    start = time.perf_counter()
    for _ in range(100):
        get_metrics_registry()
    duration = time.perf_counter() - start

    # Should be < 1ms per initialization (registry is cached)
    per_init = duration / 100
    assert per_init < 0.001, f"Metrics registry init {per_init * 1000:.3f}ms exceeds 1ms"
