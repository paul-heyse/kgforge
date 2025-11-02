"""Schema round-trip tests for vector ingestion payloads."""

from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator, ValidationError


def test_vector_schema_accepts_canonical_payload(
    vector_validator: Draft202012Validator,
    canonical_vector_payload: list[dict[str, object]],
) -> None:
    """Canonical payloads should pass JSON Schema validation."""
    vector_validator.validate(canonical_vector_payload)


def test_vector_schema_rejects_missing_key(vector_validator: Draft202012Validator) -> None:
    """Missing `key` entries must produce a validation error."""
    payload = [{"vector": [0.1, 0.2]}]

    with pytest.raises(ValidationError, match="'key' is a required property"):
        vector_validator.validate(payload)


def test_vector_schema_rejects_empty_vector(vector_validator: Draft202012Validator) -> None:
    """Vectors must contain at least one element."""
    payload = [{"key": "vec-1", "vector": []}]

    with pytest.raises(ValidationError, match="too short"):
        vector_validator.validate(payload)


def test_vector_schema_rejects_non_numeric_entries(vector_validator: Draft202012Validator) -> None:
    """Vector elements must be numeric values."""
    payload = [{"key": "vec-1", "vector": ["oops", 0.2]}]

    with pytest.raises(ValidationError, match="is not of type 'number'"):
        vector_validator.validate(payload)


def test_vector_schema_rejects_additional_properties(
    vector_validator: Draft202012Validator,
) -> None:
    """Additional properties should be disallowed to keep payloads strict."""
    payload = [{"key": "vec-1", "vector": [0.1, 0.2], "metadata": {"foo": "bar"}}]

    with pytest.raises(ValidationError, match="Additional properties are not allowed"):
        vector_validator.validate(payload)
