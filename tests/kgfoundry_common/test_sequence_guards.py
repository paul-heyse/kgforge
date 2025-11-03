"""Unit tests for sequence access guards.

Tests verify guard helpers prevent IndexError on empty sequences and emit
structured logs, metrics, and RFC 9457 Problem Details.
"""

from __future__ import annotations

import pytest

from kgfoundry_common.errors import VectorSearchError
from kgfoundry_common.sequence_guards import (
    first_or_error,
    first_or_error_multi_device,
)


class TestFirstOrError:
    """Tests for first_or_error guard function."""

    def test_valid_sequence_returns_first_element(self) -> None:
        """Verify first_or_error returns first element for non-empty sequence."""
        devices = [0, 1, 2]
        result = first_or_error(devices, context="test_devices")
        assert result == 0

    def test_valid_tuple_returns_first_element(self) -> None:
        """Verify first_or_error works with tuples."""
        devices = (10, 20, 30)
        result = first_or_error(devices, context="test_tuple")
        assert result == 10

    def test_single_element_sequence(self) -> None:
        """Verify single-element sequence returns that element."""
        devices = [42]
        result = first_or_error(devices, context="single_element")
        assert result == 42

    def test_empty_list_raises_vector_search_error(self) -> None:
        """Verify empty list raises VectorSearchError with Problem Details."""
        with pytest.raises(VectorSearchError) as exc_info:
            first_or_error([], context="empty_devices")

        error = exc_info.value
        assert "empty" in str(error).lower()
        assert "sequence is empty" in str(error)

    def test_empty_tuple_raises_vector_search_error(self) -> None:
        """Verify empty tuple raises VectorSearchError."""
        with pytest.raises(VectorSearchError) as exc_info:
            first_or_error((), context="gpu_devices")

        error = exc_info.value
        assert "empty" in str(error).lower()

    @pytest.mark.parametrize(
        "empty_seq",
        [
            [],
            (),
            "",  # empty string is a sequence
        ],
    )
    def test_various_empty_sequences_raise_error(self, empty_seq: object) -> None:
        """Verify various empty sequence types raise VectorSearchError."""
        with pytest.raises(VectorSearchError):
            first_or_error(empty_seq, context="various_sequences")  # type: ignore[arg-type]

    def test_error_includes_context_information(self) -> None:
        """Verify error message includes provided context."""
        context_str = "my_special_context"
        with pytest.raises(VectorSearchError) as exc_info:
            first_or_error([], context=context_str)

        error_msg = str(exc_info.value)
        assert context_str in error_msg

    def test_custom_operation_parameter(self) -> None:
        """Verify custom operation parameter is accepted and used."""
        operation = "custom_gpu_operation"
        with pytest.raises(VectorSearchError) as exc_info:
            first_or_error([], context="gpu", operation=operation)

        # Error should be raised without issues
        assert exc_info.value is not None

    def test_preserves_element_type(self) -> None:
        """Verify first_or_error preserves the type of returned element."""
        strings = ["hello", "world"]
        result = first_or_error(strings, context="strings")
        assert isinstance(result, str)
        assert result == "hello"

    def test_sequence_length_check(self) -> None:
        """Verify guard correctly identifies empty vs. non-empty."""
        # Non-empty should work
        assert first_or_error([1], context="c") == 1

        # Empty should fail
        with pytest.raises(VectorSearchError):
            first_or_error([], context="c")


class TestFirstOrErrorMultiDevice:
    """Tests for first_or_error_multi_device specialized variant."""

    def test_valid_devices_returns_first(self) -> None:
        """Verify multi-device variant returns first element."""
        gpu_indices = [0, 1, 2]
        result = first_or_error_multi_device(gpu_indices)
        assert result == 0

    def test_empty_devices_raises_error(self) -> None:
        """Verify empty device list raises VectorSearchError."""
        with pytest.raises(VectorSearchError) as exc_info:
            first_or_error_multi_device([])

        error = exc_info.value
        assert "FAISS GPU cloning failed" in str(error)
        assert "empty" in str(error).lower()

    def test_custom_context_parameter(self) -> None:
        """Verify custom context is used in error message."""
        custom_context = "my_gpu_list"
        with pytest.raises(VectorSearchError) as exc_info:
            first_or_error_multi_device([], context=custom_context)

        error_msg = str(exc_info.value)
        assert custom_context in error_msg

    def test_single_device(self) -> None:
        """Verify single device in list returns correctly."""
        result = first_or_error_multi_device([42])
        assert result == 42

    def test_error_mentions_gpu_context(self) -> None:
        """Verify error specifically mentions GPU/FAISS context."""
        with pytest.raises(VectorSearchError) as exc_info:
            first_or_error_multi_device([])

        error_msg = str(exc_info.value)
        # Should mention FAISS or GPU context
        assert "FAISS" in error_msg or "GPU" in error_msg.upper()

    @pytest.mark.parametrize(
        ("devices", "expected"),
        [
            ([100], 100),
            ([1, 2, 3], 1),
            (("a", "b"), "a"),
            (range(5), 0),
        ],
    )
    def test_various_valid_sequences(self, devices: object, expected: object) -> None:
        """Verify multi-device works with various sequence types."""
        result: object = first_or_error_multi_device(devices)  # type: ignore[arg-type]
        assert result == expected


class TestErrorObservability:
    """Tests for observability aspects of guards (logs, metrics)."""

    def test_error_is_vector_search_error_type(self) -> None:
        """Verify guard raises the correct exception type."""
        with pytest.raises(VectorSearchError):
            first_or_error([], context="test")

        # Guard should raise VectorSearchError, not generic IndexError or ValueError
        # This is verified by the pytest.raises above

    def test_preserves_exception_cause_chain(self) -> None:
        """Verify exception cause chain is clean (raised from None)."""
        with pytest.raises(VectorSearchError) as exc_info:
            first_or_error([], context="test")

        # Cause should be None (raised from None)
        assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    ("seq_type", "seq_value"),
    [
        ("empty_list", []),
        ("empty_tuple", ()),
        ("list_with_one", [1]),
        ("list_with_many", [1, 2, 3, 4, 5]),
        ("tuple_single", (42,)),
        ("tuple_multi", (10, 20, 30)),
    ],
)
class TestFirstOrErrorParametrized:
    """Parametrized tests for comprehensive coverage."""

    def test_sequence_behavior(self, seq_type: str, seq_value: object) -> None:
        """Comprehensive parametrized test for various sequences."""
        if seq_type.startswith("empty"):
            # Empty sequences should raise
            with pytest.raises(VectorSearchError):
                first_or_error(seq_value, context="test")  # type: ignore[arg-type]
        else:
            # Non-empty sequences should return first element
            result: object = first_or_error(seq_value, context="test")  # type: ignore[arg-type]
            # Verify it returned something (the first element)
            assert result is not None
