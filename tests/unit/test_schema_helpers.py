"""Unit tests for schema and model round-trip validation helpers.

Tests cover schema loading, model validation against schemas, and
round-trip validation with example JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kgfoundry_common.errors import DeserializationError, SerializationError
from kgfoundry_common.models import Doc
from kgfoundry_common.schema_helpers import (
    assert_model_roundtrip,
    load_schema,
    validate_model_against_schema,
)
from search_api.schemas import SearchRequest, SearchResult


class TestLoadSchema:
    """Test schema loading and validation."""

    @pytest.fixture
    def valid_schema(self, tmp_path: Path) -> Path:
        """Create a valid JSON Schema 2020-12 file."""
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://example.com/test.v1.json",
            "type": "object",
            "properties": {
                "id": {"type": "string"},
            },
            "required": ["id"],
        }
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps(schema))
        return schema_path

    def test_load_schema_success(self, valid_schema: Path) -> None:
        """Successfully load a valid schema."""
        schema_obj = load_schema(valid_schema)
        assert schema_obj["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema_obj["type"] == "object"

    def test_load_schema_missing_file(self, tmp_path: Path) -> None:
        """Missing schema file raises FileNotFoundError."""
        missing_path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            load_schema(missing_path)

    def test_load_schema_invalid_json(self, tmp_path: Path) -> None:
        """Invalid JSON raises DeserializationError."""
        invalid_path = tmp_path / "invalid.json"
        invalid_path.write_text("{invalid json}")
        with pytest.raises(DeserializationError, match="Invalid JSON"):
            load_schema(invalid_path)

    def test_load_schema_invalid_schema(self, tmp_path: Path) -> None:
        """Invalid JSON Schema raises DeserializationError."""
        invalid_schema_path = tmp_path / "invalid_schema.json"
        invalid_schema_path.write_text('{"type": "invalid_type"}')
        with pytest.raises(DeserializationError, match="Invalid JSON Schema"):
            load_schema(invalid_schema_path)


class TestValidateModelAgainstSchema:
    """Test model validation against schemas."""

    @pytest.fixture
    def doc_schema(self) -> dict[str, object]:
        """Create a schema matching Doc model."""
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
            },
            "required": ["id"],
        }

    def test_validate_model_success(self, doc_schema: dict[str, object]) -> None:
        """Valid model instance passes validation."""
        doc = Doc(id="urn:doc:test", title="Test")
        validate_model_against_schema(doc, doc_schema)  # Should not raise

    def test_validate_model_invalid(self, doc_schema: dict[str, object]) -> None:  # noqa: ARG002
        """Model instance that doesn't match schema raises SerializationError."""
        # Schema requires id, but we'll use an invalid schema
        invalid_schema = {"type": "object", "properties": {"id": {"type": "integer"}}}
        doc = Doc(id="urn:doc:test")
        with pytest.raises(SerializationError, match="does not match schema"):
            validate_model_against_schema(doc, invalid_schema)


class TestAssertModelRoundtrip:
    """Test round-trip validation with example JSON files."""

    @pytest.fixture
    def doc_example(self, tmp_path: Path) -> Path:
        """Create example JSON for Doc model."""
        example = {
            "id": "urn:doc:test",
            "title": "Test Document",
            "authors": [],
        }
        example_path = tmp_path / "doc_example.json"
        example_path.write_text(json.dumps(example))
        return example_path

    @pytest.fixture
    def doc_schema(self, tmp_path: Path) -> Path:
        """Create schema for Doc model."""
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://example.com/doc.v1.json",
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "authors": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["id"],
        }
        schema_path = tmp_path / "doc_schema.json"
        schema_path.write_text(json.dumps(schema))
        return schema_path

    def test_assert_model_roundtrip_success(self, doc_example: Path, doc_schema: Path) -> None:
        """Round-trip validation succeeds with valid example and schema."""
        assert_model_roundtrip(Doc, doc_example, schema_path=doc_schema)  # Should not raise

    def test_assert_model_roundtrip_no_schema(self, doc_example: Path) -> None:
        """Round-trip validation works without schema."""
        assert_model_roundtrip(Doc, doc_example, schema_path=None)  # Should not raise

    def test_assert_model_roundtrip_missing_example(self, tmp_path: Path) -> None:
        """Missing example file raises FileNotFoundError."""
        missing_path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            assert_model_roundtrip(Doc, missing_path)

    def test_assert_model_roundtrip_invalid_example(self, tmp_path: Path, doc_schema: Path) -> None:
        """Invalid example JSON raises DeserializationError."""
        invalid_path = tmp_path / "invalid.json"
        invalid_path.write_text("{invalid json}")
        with pytest.raises(DeserializationError, match="Invalid JSON"):
            assert_model_roundtrip(Doc, invalid_path, schema_path=doc_schema)

    def test_assert_model_roundtrip_example_mismatch_schema(
        self, tmp_path: Path, doc_schema: Path
    ) -> None:
        """Example that doesn't match schema raises DeserializationError."""
        invalid_example = {"wrong_field": "value"}
        invalid_path = tmp_path / "invalid_example.json"
        invalid_path.write_text(json.dumps(invalid_example))
        with pytest.raises(DeserializationError, match="does not match schema"):
            assert_model_roundtrip(Doc, invalid_path, schema_path=doc_schema)

    def test_assert_model_roundtrip_model_validation_fails(self, tmp_path: Path) -> None:
        """Example that fails model validation raises DeserializationError."""
        # Missing required field 'id' - test without schema so model validation runs
        invalid_example = {"title": "Test"}
        invalid_path = tmp_path / "invalid_model.json"
        invalid_path.write_text(json.dumps(invalid_example))
        with pytest.raises(DeserializationError, match="Failed to deserialize"):  # type: ignore[call-arg]
            assert_model_roundtrip(Doc, invalid_path, schema_path=None)


class TestRoundTripWithRealSchemas:
    """Test round-trip validation using real schema files from the repo."""

    def test_doc_roundtrip(self) -> None:
        """Doc model round-trips with example from schema/examples."""
        schema_path = Path("schema/models/doc.v1.json")
        example_path = Path("schema/examples/models/doc.v1.json")
        if schema_path.exists() and example_path.exists():
            assert_model_roundtrip(Doc, example_path, schema_path=schema_path)

    def test_search_request_roundtrip(self) -> None:
        """SearchRequest model round-trips with example."""
        schema_path = Path("schema/models/search_request.v1.json")
        example_path = Path("schema/examples/search_api/search_request.v1.json")
        if schema_path.exists() and example_path.exists():
            assert_model_roundtrip(SearchRequest, example_path, schema_path=schema_path)

    def test_search_result_roundtrip(self) -> None:
        """SearchResult model round-trips with example."""
        schema_path = Path("schema/models/search_result.v1.json")
        example_path = Path("schema/examples/search_api/search_result.v1.json")
        if schema_path.exists() and example_path.exists():
            assert_model_roundtrip(SearchResult, example_path, schema_path=schema_path)
