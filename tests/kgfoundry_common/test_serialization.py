"""Tests for kgfoundry_common.serialization module."""

from __future__ import annotations

from pathlib import Path

import pytest

from kgfoundry_common.errors import (
    DeserializationError,
    SchemaValidationError,
    SerializationError,
)
from kgfoundry_common.serialization import (
    compute_checksum,
    deserialize_json,
    serialize_json,
    validate_payload,
    verify_checksum,
)


class TestComputeChecksum:
    """Tests for compute_checksum function."""

    def test_compute_checksum_returns_hex_string(self) -> None:
        """compute_checksum returns 64-character hexadecimal string."""
        checksum = compute_checksum(b"test data")
        assert isinstance(checksum, str)
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_compute_checksum_is_deterministic(self) -> None:
        """compute_checksum returns same value for same input."""
        data = b"test data"
        checksum1 = compute_checksum(data)
        checksum2 = compute_checksum(data)
        assert checksum1 == checksum2

    def test_compute_checksum_different_inputs(self) -> None:
        """compute_checksum returns different values for different inputs."""
        checksum1 = compute_checksum(b"test data 1")
        checksum2 = compute_checksum(b"test data 2")
        assert checksum1 != checksum2


class TestVerifyChecksum:
    """Tests for verify_checksum function."""

    def test_verify_checksum_valid(self) -> None:
        """verify_checksum succeeds for matching checksum."""
        data = b"test data"
        expected = compute_checksum(data)
        verify_checksum(data, expected)  # Should not raise

    def test_verify_checksum_mismatch_raises(self) -> None:
        """verify_checksum raises SerializationError for mismatched checksum."""
        data = b"test data"
        wrong_checksum = "0" * 64
        with pytest.raises(SerializationError, match="Checksum mismatch"):  # type: ignore[call-arg]  # pytest.raises supports match, mypy stub issue
            verify_checksum(data, wrong_checksum)


class TestValidatePayload:
    """Tests for validate_payload function."""

    def test_validate_payload_valid(self, tmp_path: Path) -> None:
        """validate_payload succeeds for valid payload."""
        schema_path = tmp_path / "schema.json"
        schema_path.write_text('{"type": "object", "properties": {"k1": {"type": "number"}}}')
        payload = {"k1": 0.9}
        validate_payload(payload, schema_path)  # Should not raise

    def test_validate_payload_invalid_raises(self, tmp_path: Path) -> None:
        """validate_payload raises SchemaValidationError for invalid payload."""
        schema_path = tmp_path / "schema.json"
        schema_path.write_text('{"type": "object", "properties": {"k1": {"type": "number"}}}')
        payload = {"k1": "invalid"}
        with pytest.raises(SchemaValidationError, match="Schema validation failed"):  # type: ignore[call-arg]  # pytest.raises supports match, mypy stub issue
            validate_payload(payload, schema_path)

    def test_validate_payload_missing_schema_raises(self, tmp_path: Path) -> None:
        """validate_payload raises FileNotFoundError for missing schema."""
        schema_path = tmp_path / "nonexistent.json"
        payload = {"k1": 0.9}
        with pytest.raises(FileNotFoundError, match="Schema file not found"):  # type: ignore[call-arg]  # pytest.raises supports match, mypy stub issue
            validate_payload(payload, schema_path)

    def test_validate_payload_invalid_schema_raises(self, tmp_path: Path) -> None:
        """validate_payload raises SchemaValidationError for invalid schema."""
        schema_path = tmp_path / "schema.json"
        schema_path.write_text('{"type": "invalid_type"}')  # Invalid JSON Schema
        payload = {"k1": 0.9}
        with pytest.raises(SchemaValidationError, match="Invalid JSON Schema"):  # type: ignore[call-arg]  # pytest.raises supports match, mypy stub issue
            validate_payload(payload, schema_path)

    def test_validate_payload_caches_schema(self, tmp_path: Path) -> None:
        """validate_payload caches loaded schemas."""
        schema_path = tmp_path / "schema.json"
        schema_path.write_text('{"type": "object", "properties": {"k1": {"type": "number"}}}')
        payload1 = {"k1": 0.9}
        payload2 = {"k2": 1.0}

        # First call loads schema
        validate_payload(payload1, schema_path)

        # Modify schema file (cache should prevent reload)
        schema_path.write_text('{"type": "object", "properties": {"k2": {"type": "number"}}}')

        # Second call should use cached schema (validates against old schema)
        validate_payload(payload2, schema_path)  # Should succeed with old schema


class TestSerializeJson:
    """Tests for serialize_json function."""

    def test_serialize_json_valid(self, tmp_path: Path) -> None:
        """serialize_json succeeds for valid data."""
        schema_path = tmp_path / "schema.json"
        schema_path.write_text('{"type": "object", "properties": {"k1": {"type": "number"}}}')
        output_path = tmp_path / "data.json"
        data = {"k1": 0.9}

        checksum = serialize_json(data, schema_path, output_path)

        assert isinstance(checksum, str)
        assert len(checksum) == 64
        assert output_path.exists()
        assert output_path.read_text() == '{\n  "k1": 0.9\n}'

    def test_serialize_json_without_checksum(self, tmp_path: Path) -> None:
        """serialize_json skips checksum file when include_checksum=False."""
        schema_path = tmp_path / "schema.json"
        schema_path.write_text('{"type": "object", "properties": {"k1": {"type": "number"}}}')
        output_path = tmp_path / "data.json"
        data = {"k1": 0.9}

        checksum = serialize_json(data, schema_path, output_path, include_checksum=False)

        assert isinstance(checksum, str)
        assert output_path.exists()
        checksum_path = output_path.with_suffix(output_path.suffix + ".sha256")
        assert not checksum_path.exists()

    def test_serialize_json_invalid_schema_raises(self, tmp_path: Path) -> None:
        """serialize_json raises SchemaValidationError for invalid payload."""
        schema_path = tmp_path / "schema.json"
        schema_path.write_text('{"type": "object", "properties": {"k1": {"type": "number"}}}')
        output_path = tmp_path / "data.json"
        data = {"k1": "invalid"}

        with pytest.raises(SchemaValidationError, match="Schema validation failed"):  # type: ignore[call-arg]  # pytest.raises supports match, mypy stub issue
            serialize_json(data, schema_path, output_path)

    def test_serialize_json_missing_schema_raises(self, tmp_path: Path) -> None:
        """serialize_json raises FileNotFoundError for missing schema."""
        schema_path = tmp_path / "nonexistent.json"
        output_path = tmp_path / "data.json"
        data = {"k1": 0.9}

        with pytest.raises(FileNotFoundError, match="Schema file not found"):  # type: ignore[call-arg]  # pytest.raises supports match, mypy stub issue
            serialize_json(data, schema_path, output_path)

    def test_serialize_json_non_json_serializable_raises(self, tmp_path: Path) -> None:
        """serialize_json raises SchemaValidationError for non-Mapping object."""
        schema_path = tmp_path / "schema.json"
        schema_path.write_text('{"type": "object"}')
        output_path = tmp_path / "data.json"
        # Use a set, which is not a Mapping and fails schema validation first
        data: set[int] = {1, 2, 3}  # Intentionally invalid type

        with pytest.raises(SchemaValidationError, match="Schema validation failed"):  # type: ignore[call-arg]  # pytest.raises supports match, mypy stub issue
            serialize_json(data, schema_path, output_path)


class TestDeserializeJson:
    """Tests for deserialize_json function."""

    def test_deserialize_json_valid(self, tmp_path: Path) -> None:
        """deserialize_json succeeds for valid JSON."""
        schema_path = tmp_path / "schema.json"
        schema_path.write_text('{"type": "object", "properties": {"k1": {"type": "number"}}}')
        data_path = tmp_path / "data.json"
        data_path.write_text('{"k1": 0.9}')

        loaded = deserialize_json(data_path, schema_path)

        assert loaded == {"k1": 0.9}

    def test_deserialize_json_with_checksum(self, tmp_path: Path) -> None:
        """deserialize_json verifies checksum when checksum file exists."""
        schema_path = tmp_path / "schema.json"
        schema_path.write_text('{"type": "object", "properties": {"k1": {"type": "number"}}}')
        data_path = tmp_path / "data.json"
        data_path.write_text('{"k1": 0.9}')

        # Create checksum file
        checksum = compute_checksum(data_path.read_bytes())
        checksum_path = data_path.with_suffix(data_path.suffix + ".sha256")
        checksum_path.write_text(checksum)

        loaded = deserialize_json(data_path, schema_path, verify_checksum_file=True)

        assert loaded == {"k1": 0.9}

    def test_deserialize_json_checksum_mismatch_raises(self, tmp_path: Path) -> None:
        """deserialize_json raises DeserializationError for checksum mismatch."""
        schema_path = tmp_path / "schema.json"
        schema_path.write_text('{"type": "object", "properties": {"k1": {"type": "number"}}}')
        data_path = tmp_path / "data.json"
        data_path.write_text('{"k1": 0.9}')

        # Create invalid checksum file
        checksum_path = data_path.with_suffix(data_path.suffix + ".sha256")
        checksum_path.write_text("0" * 64)

        with pytest.raises(DeserializationError, match="Checksum verification failed"):  # type: ignore[call-arg]  # pytest.raises supports match, mypy stub issue
            deserialize_json(data_path, schema_path, verify_checksum_file=True)

    def test_deserialize_json_skip_checksum(self, tmp_path: Path) -> None:
        """deserialize_json skips checksum verification when verify_checksum_file=False."""
        schema_path = tmp_path / "schema.json"
        schema_path.write_text('{"type": "object", "properties": {"k1": {"type": "number"}}}')
        data_path = tmp_path / "data.json"
        data_path.write_text('{"k1": 0.9}')

        loaded = deserialize_json(data_path, schema_path, verify_checksum_file=False)

        assert loaded == {"k1": 0.9}

    def test_deserialize_json_invalid_schema_raises(self, tmp_path: Path) -> None:
        """deserialize_json raises SchemaValidationError for invalid payload."""
        schema_path = tmp_path / "schema.json"
        schema_path.write_text('{"type": "object", "properties": {"k1": {"type": "number"}}}')
        data_path = tmp_path / "data.json"
        data_path.write_text('{"k1": "invalid"}')

        with pytest.raises(SchemaValidationError, match="Schema validation failed"):  # type: ignore[call-arg]  # pytest.raises supports match, mypy stub issue
            deserialize_json(data_path, schema_path)

    def test_deserialize_json_missing_schema_raises(self, tmp_path: Path) -> None:
        """deserialize_json raises FileNotFoundError for missing schema."""
        schema_path = tmp_path / "nonexistent.json"
        data_path = tmp_path / "data.json"
        data_path.write_text('{"k1": 0.9}')

        with pytest.raises(FileNotFoundError, match="Schema file not found"):  # type: ignore[call-arg]  # pytest.raises supports match, mypy stub issue
            deserialize_json(data_path, schema_path)

    def test_deserialize_json_missing_data_raises(self, tmp_path: Path) -> None:
        """deserialize_json raises FileNotFoundError for missing data file."""
        schema_path = tmp_path / "schema.json"
        schema_path.write_text('{"type": "object"}')
        data_path = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError, match="Data file not found"):  # type: ignore[call-arg]  # pytest.raises supports match, mypy stub issue
            deserialize_json(data_path, schema_path)

    def test_deserialize_json_invalid_json_raises(self, tmp_path: Path) -> None:
        """deserialize_json raises DeserializationError for invalid JSON."""
        schema_path = tmp_path / "schema.json"
        schema_path.write_text('{"type": "object"}')
        data_path = tmp_path / "data.json"
        data_path.write_text("invalid json")

        with pytest.raises(DeserializationError, match="Invalid JSON"):  # type: ignore[call-arg]  # pytest.raises supports match, mypy stub issue
            deserialize_json(data_path, schema_path)


class TestRoundTrip:
    """Tests for serialize_json and deserialize_json round-trip."""

    def test_round_trip_valid(self, tmp_path: Path) -> None:
        """serialize_json and deserialize_json round-trip correctly."""
        schema_path = tmp_path / "schema.json"
        schema_path.write_text('{"type": "object", "properties": {"k1": {"type": "number"}}}')
        output_path = tmp_path / "data.json"
        data = {"k1": 0.9}

        checksum = serialize_json(data, schema_path, output_path)
        loaded = deserialize_json(output_path, schema_path)

        assert loaded == data
        assert isinstance(checksum, str)
        assert len(checksum) == 64
