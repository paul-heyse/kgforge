"""Tests for async context propagation and correlation ID isolation.

This module verifies that correlation IDs propagate correctly through async
tasks and that concurrent tasks maintain separate correlation IDs.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from kgfoundry_common.logging import get_correlation_id, get_logger, set_correlation_id


async def test_contextvar_propagation_through_async_tasks():
    """Verify correlation ID propagates through async tasks.

    Scenario: ContextVar propagation

    GIVEN an async search request with a seeded correlation ID
    WHEN the request awaits downstream adapters
    THEN logs, metrics, and Problem Details emitted from inside the adapter
    include the same correlation ID
    """
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)

    async def downstream_adapter() -> str:
        """Simulate a downstream adapter that should preserve correlation ID."""
        # Correlation ID should propagate automatically via ContextVar
        await asyncio.sleep(0.01)  # Simulate async work
        logger = get_logger(__name__)
        logger.info(
            "Downstream adapter called", extra={"operation": "adapter", "status": "success"}
        )
        return get_correlation_id() or ""

    result_id = await downstream_adapter()
    assert result_id == correlation_id, "Correlation ID should propagate through async tasks"


@pytest.mark.asyncio
async def test_async_context_invariant_parallel_tasks():
    """Verify correlation IDs don't cross between parallel async tasks.

    Scenario: ContextVar propagation

    GIVEN two parallel async tasks with different correlation IDs
    WHEN they execute concurrently
    THEN each task maintains its own correlation ID without cross-contamination
    """
    correlation_id_1 = str(uuid.uuid4())
    correlation_id_2 = str(uuid.uuid4())

    async def task_with_id(task_id: str, correlation_id: str) -> tuple[str, str]:
        """Execute a task with a specific correlation ID."""
        set_correlation_id(correlation_id)
        await asyncio.sleep(0.01)  # Simulate async work
        logger = get_logger(__name__)
        logger.info(f"Task {task_id} executed", extra={"operation": "task", "status": "success"})
        # Verify correlation ID is still correct after async work
        retrieved_id = get_correlation_id()
        return task_id, retrieved_id or ""

    # Run both tasks concurrently
    results = await asyncio.gather(
        task_with_id("task-1", correlation_id_1),
        task_with_id("task-2", correlation_id_2),
    )

    # Verify each task maintained its own correlation ID
    task_1_result_id, task_1_correlation_id = results[0]
    task_2_result_id, task_2_correlation_id = results[1]

    assert task_1_correlation_id == correlation_id_1, (
        f"Task 1 should maintain correlation ID {correlation_id_1}"
    )
    assert task_2_correlation_id == correlation_id_2, (
        f"Task 2 should maintain correlation ID {correlation_id_2}"
    )
    assert task_1_correlation_id != task_2_correlation_id, (
        "Correlation IDs should not cross between tasks"
    )


@pytest.mark.asyncio
async def test_nested_async_context_propagation():
    """Verify correlation ID propagates through nested async calls."""
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)

    async def nested_function() -> str:
        """Nested async function that should inherit correlation ID."""
        await asyncio.sleep(0.01)
        return get_correlation_id() or ""

    async def outer_function() -> str:
        """Outer async function that calls nested function."""
        await asyncio.sleep(0.01)
        return await nested_function()

    result_id = await outer_function()
    assert result_id == correlation_id, "Correlation ID should propagate through nested async calls"


@pytest.mark.asyncio
async def test_context_isolation_with_task_creation():
    """Verify correlation IDs are isolated when creating new tasks."""
    correlation_id_1 = str(uuid.uuid4())
    correlation_id_2 = str(uuid.uuid4())

    async def task_with_id(correlation_id: str) -> str:
        """Task that sets and retrieves correlation ID."""
        set_correlation_id(correlation_id)
        await asyncio.sleep(0.01)
        return get_correlation_id() or ""

    # Set initial correlation ID
    set_correlation_id(correlation_id_1)

    # Create a task with a different correlation ID
    task = asyncio.create_task(task_with_id(correlation_id_2))

    # Verify original correlation ID is still set
    assert get_correlation_id() == correlation_id_1, (
        "Original correlation ID should remain unchanged"
    )

    # Wait for task to complete and verify it used its own correlation ID
    task_result = await task
    assert task_result == correlation_id_2, "Task should use its own correlation ID"


@pytest.mark.asyncio
async def test_multiple_concurrent_requests():
    """Verify multiple concurrent requests maintain separate correlation IDs."""
    correlation_ids = [str(uuid.uuid4()) for _ in range(5)]

    async def simulate_request(correlation_id: str, request_num: int) -> tuple[int, str]:
        """Simulate a request with a correlation ID."""
        set_correlation_id(correlation_id)
        await asyncio.sleep(0.01)
        logger = get_logger(__name__)
        logger.info(
            f"Request {request_num} processed",
            extra={"operation": "request", "status": "success"},
        )
        return request_num, get_correlation_id() or ""

    # Run all requests concurrently
    results = await asyncio.gather(
        *[simulate_request(cid, i) for i, cid in enumerate(correlation_ids)]
    )

    # Verify each request maintained its own correlation ID
    for request_num, retrieved_id in results:
        expected_id = correlation_ids[request_num]
        assert retrieved_id == expected_id, (
            f"Request {request_num} should maintain correlation ID {expected_id}"
        )

    # Verify all correlation IDs are unique
    retrieved_ids = [rid for _, rid in results]
    assert len(set(retrieved_ids)) == len(retrieved_ids), "All correlation IDs should be unique"
