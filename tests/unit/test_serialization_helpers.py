"""Unit tests for secure serialization helpers.

Tests cover schema validation, checksum computation/verification, and
error handling for JSON serialization/deserialization.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kgfoundry_common.errors import DeserializationError, SerializationError
from kgfoundry_common.serialization import (
    compute_checksum,
    deserialize_json,
    serialize_json,
    verify_checksum,
)


class TestComputeChecksum:
    """Test checksum computation."""

    def test_compute_checksum_returns_hex_string(self) -> None:
        """Checksum is a 64-character hexadecimal string."""
        checksum = compute_checksum(b"test data")
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_compute_checksum_deterministic(self) -> None:
        """Same input produces same checksum."""
        data = b"consistent input"
        assert compute_checksum(data) == compute_checksum(data)

    def test_compute_checksum_different_inputs(self) -> None:
        """Different inputs produce different checksums."""
        c1 = compute_checksum(b"input1")
        c2 = compute_checksum(b"input2")
        assert c1 != c2


class TestVerifyChecksum:
    """Test checksum verification."""

    def test_verify_checksum_valid(self) -> None:
        """Valid checksum passes verification."""
        data = b"test data"
        checksum = compute_checksum(data)
        verify_checksum(data, checksum)  # Should not raise

    def test_verify_checksum_invalid(self) -> None:
        """Invalid checksum raises SerializationError."""
        data = b"test data"
        wrong_checksum = "0" * 64
        with pytest.raises(SerializationError, match="Checksum mismatch"):  # type: ignore[call-arg]
            verify_checksum(data, wrong_checksum)


class TestSerializeJson:
    """Test JSON serialization with schema validation."""

    @pytest.fixture
    def valid_schema(self, tmp_path: Path) -> Path:
        """Create a valid JSON Schema file."""
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "k1": {"type": "number"},
                "b": {"type": "number"},
                "N": {"type": "integer"},
            },
            "required": ["k1", "b", "N"],
        }
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps(schema))
        return schema_path

    @pytest.fixture
    def invalid_schema(self, tmp_path: Path) -> Path:
        """Create an invalid JSON Schema file."""
        schema_path = tmp_path / "invalid_schema.json"
        schema_path.write_text('{"type": "invalid_type"}')
        return schema_path

    def test_serialize_json_success(self, valid_schema: Path, tmp_path: Path) -> None:
        """Successfully serialize valid data."""
        data = {"k1": 0.9, "b": 0.4, "N": 100}
        output_path = tmp_path / "output.json"
        checksum = serialize_json(data, valid_schema, output_path)
        assert output_path.exists()
        assert len(checksum) == 64
        assert (tmp_path / "output.json.sha256").exists()

    def test_serialize_json_validates_schema(self, valid_schema: Path, tmp_path: Path) -> None:
        """Schema validation rejects invalid data."""
        invalid_data = {"k1": "not a number", "b": 0.4, "N": 100}
        output_path = tmp_path / "output.json"
        with pytest.raises(SerializationError, match="Schema validation failed"):  # type: ignore[call-arg]
            serialize_json(invalid_data, valid_schema, output_path)

    def test_serialize_json_missing_required_field(
        self, valid_schema: Path, tmp_path: Path
    ) -> None:
        """Missing required fields fail validation."""
        incomplete_data = {"k1": 0.9, "b": 0.4}  # Missing N
        output_path = tmp_path / "output.json"
        with pytest.raises(SerializationError, match="Schema validation failed"):  # type: ignore[call-arg]
            serialize_json(incomplete_data, valid_schema, output_path)

    def test_serialize_json_non_serializable(self, valid_schema: Path, tmp_path: Path) -> None:
        """Non-JSON-serializable objects raise SerializationError."""
        non_serializable = {"func": lambda x: x}  # Functions not JSON-serializable
        output_path = tmp_path / "output.json"
        with pytest.raises(SerializationError, match="Failed to serialize"):  # type: ignore[call-arg]
            serialize_json(non_serializable, valid_schema, output_path)

    def test_serialize_json_missing_schema(self, tmp_path: Path) -> None:
        """Missing schema file raises FileNotFoundError."""
        data = {"k1": 0.9}
        schema_path = tmp_path / "nonexistent.json"
        output_path = tmp_path / "output.json"
        with pytest.raises(FileNotFoundError):
            serialize_json(data, schema_path, output_path)

    def test_serialize_json_no_checksum(self, valid_schema: Path, tmp_path: Path) -> None:
        """Serialization without checksum file works."""
        data = {"k1": 0.9, "b": 0.4, "N": 100}
        output_path = tmp_path / "output.json"
        checksum = serialize_json(data, valid_schema, output_path, include_checksum=False)
        assert output_path.exists()
        assert not (tmp_path / "output.json.sha256").exists()
        assert len(checksum) == 64  # Checksum still computed, just not written

    def test_serialize_json_compact_format(self, valid_schema: Path, tmp_path: Path) -> None:
        """Compact JSON format (no indent) works."""
        data = {"k1": 0.9, "b": 0.4, "N": 100}
        output_path = tmp_path / "output.json"
        serialize_json(data, valid_schema, output_path, indent=None)
        content = output_path.read_text()
        assert "\n" not in content.strip()  # Compact format


class TestDeserializeJson:
    """Test JSON deserialization with schema validation and checksum verification."""

    @pytest.fixture
    def valid_schema(self, tmp_path: Path) -> Path:
        """Create a valid JSON Schema file."""
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "k1": {"type": "number"},
                "b": {"type": "number"},
                "N": {"type": "integer"},
            },
            "required": ["k1", "b", "N"],
        }
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps(schema))
        return schema_path

    @pytest.fixture
    def valid_data_file(self, tmp_path: Path) -> Path:
        """Create a valid JSON data file."""
        data = {"k1": 0.9, "b": 0.4, "N": 100}
        data_path = tmp_path / "data.json"
        data_path.write_text(json.dumps(data))
        return data_path

    def test_deserialize_json_success(self, valid_schema: Path, valid_data_file: Path) -> None:
        """Successfully deserialize valid data."""
        result = deserialize_json(valid_data_file, valid_schema)
        assert result == {"k1": 0.9, "b": 0.4, "N": 100}

    def test_deserialize_json_validates_schema(self, valid_schema: Path, tmp_path: Path) -> None:
        """Schema validation rejects invalid data."""
        invalid_data = {"k1": "not a number", "b": 0.4, "N": 100}
        data_path = tmp_path / "invalid.json"
        data_path.write_text(json.dumps(invalid_data))
        with pytest.raises(DeserializationError, match="Schema validation failed"):
            deserialize_json(data_path, valid_schema)

    def test_deserialize_json_invalid_json(self, valid_schema: Path, tmp_path: Path) -> None:
        """Invalid JSON raises DeserializationError."""
        invalid_json_path = tmp_path / "invalid.json"
        invalid_json_path.write_text("{invalid json}")
        with pytest.raises(DeserializationError, match="Invalid JSON"):  # type: ignore[call-arg]
            deserialize_json(invalid_json_path, valid_schema)

    def test_deserialize_json_missing_file(self, valid_schema: Path, tmp_path: Path) -> None:
        """Missing data file raises FileNotFoundError."""
        missing_path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            deserialize_json(missing_path, valid_schema)

    def test_deserialize_json_missing_schema(self, valid_data_file: Path, tmp_path: Path) -> None:
        """Missing schema file raises FileNotFoundError."""
        missing_schema = tmp_path / "nonexistent_schema.json"
        with pytest.raises(FileNotFoundError):
            deserialize_json(valid_data_file, missing_schema)

    def test_deserialize_json_checksum_verification(
        self, valid_schema: Path, valid_data_file: Path
    ) -> None:
        """Checksum verification works when checksum file exists."""
        # Write correct checksum
        data_bytes = valid_data_file.read_bytes()
        checksum = compute_checksum(data_bytes)
        checksum_path = valid_data_file.with_suffix(".json.sha256")
        checksum_path.write_text(checksum)
        # Should succeed
        result = deserialize_json(valid_data_file, valid_schema, verify_checksum_file=True)
        assert result == {"k1": 0.9, "b": 0.4, "N": 100}

    def test_deserialize_json_checksum_mismatch(
        self, valid_schema: Path, valid_data_file: Path
    ) -> None:
        """Checksum mismatch raises DeserializationError."""
        # Write wrong checksum
        checksum_path = valid_data_file.with_suffix(".json.sha256")
        checksum_path.write_text("0" * 64)
        with pytest.raises(DeserializationError, match="Checksum verification failed"):  # type: ignore[call-arg]
            deserialize_json(valid_data_file, valid_schema, verify_checksum_file=True)

    def test_deserialize_json_no_checksum_file_warning(
        self, valid_schema: Path, valid_data_file: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Missing checksum file logs warning but succeeds."""
        result = deserialize_json(valid_data_file, valid_schema, verify_checksum_file=True)
        assert result == {"k1": 0.9, "b": 0.4, "N": 100}
        assert "Checksum file not found" in caplog.text

    def test_deserialize_json_skip_checksum(
        self, valid_schema: Path, valid_data_file: Path
    ) -> None:
        """Deserialization can skip checksum verification."""
        result = deserialize_json(valid_data_file, valid_schema, verify_checksum_file=False)
        assert result == {"k1": 0.9, "b": 0.4, "N": 100}


class TestRoundTrip:
    """Test round-trip serialization/deserialization."""

    @pytest.fixture
    def schema(self, tmp_path: Path) -> Path:
        """Create a valid JSON Schema file."""
        schema_obj = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "k1": {"type": "number"},
                "b": {"type": "number"},
                "N": {"type": "integer"},
                "df": {"type": "object"},
            },
            "required": ["k1", "b", "N"],
        }
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps(schema_obj))
        return schema_path

    def test_round_trip_preserves_data(self, schema: Path, tmp_path: Path) -> None:
        """Round-trip serialization preserves data."""
        original = {"k1": 0.9, "b": 0.4, "N": 100, "df": {"term1": 5, "term2": 10}}
        output_path = tmp_path / "data.json"
        serialize_json(original, schema, output_path)
        loaded = deserialize_json(output_path, schema)
        assert loaded == original

    def test_round_trip_with_checksum(self, schema: Path, tmp_path: Path) -> None:
        """Round-trip with checksum verification works."""
        original = {"k1": 0.9, "b": 0.4, "N": 100}
        output_path = tmp_path / "data.json"
        serialize_json(original, schema, output_path, include_checksum=True)
        loaded = deserialize_json(output_path, schema, verify_checksum_file=True)
        assert loaded == original
