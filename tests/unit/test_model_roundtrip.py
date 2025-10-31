"""Unit tests for model round-trip validation.

Tests validate that Pydantic models can serialize from example JSON,
deserialize into the model, and re-serialize, ensuring schema parity.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kgfoundry_common.models import Doc
from kgfoundry_common.schema_helpers import assert_model_roundtrip
from search_api.schemas import SearchRequest, SearchResult


@pytest.mark.parametrize(
    ("model_cls", "example_path", "schema_filename"),
    [
        (
            Doc,
            Path(__file__).parent.parent.parent / "schema" / "examples" / "models" / "doc.v1.json",
            "doc.v1.json",
        ),
        (
            SearchRequest,
            Path(__file__).parent.parent.parent
            / "schema"
            / "examples"
            / "search_api"
            / "search_request.v1.json",
            "search_request.v1.json",
        ),
        (
            SearchResult,
            Path(__file__).parent.parent.parent
            / "schema"
            / "examples"
            / "search_api"
            / "search_result.v1.json",
            "search_result.v1.json",
        ),
    ],
)
def test_model_roundtrip_with_schema(
    model_cls: type[Doc | SearchRequest | SearchResult],
    example_path: Path,
    schema_filename: str,
) -> None:
    """Test that models can round-trip through serialization with schema validation.

    Scenario: Schema validation for agent session state
    - Maps to Requirement: Typed JSON Contracts (R4)
    """
    schema_path = Path(__file__).parent.parent.parent / "schema" / "models" / schema_filename
    assert_model_roundtrip(model_cls, example_path, schema_path=schema_path)


@pytest.mark.parametrize(
    ("model_cls", "example_path"),
    [
        (
            Doc,
            Path(__file__).parent.parent.parent / "schema" / "examples" / "models" / "doc.v1.json",
        ),
        (
            SearchRequest,
            Path(__file__).parent.parent.parent
            / "schema"
            / "examples"
            / "search_api"
            / "search_request.v1.json",
        ),
        (
            SearchResult,
            Path(__file__).parent.parent.parent
            / "schema"
            / "examples"
            / "search_api"
            / "search_result.v1.json",
        ),
    ],
)
def test_model_roundtrip_without_schema(
    model_cls: type[Doc | SearchRequest | SearchResult], example_path: Path
) -> None:
    """Test that models can round-trip through serialization without schema validation.

    Scenario: Schema validation for agent session state
    - Maps to Requirement: Typed JSON Contracts (R4)
    """
    assert_model_roundtrip(model_cls, example_path)


@pytest.mark.parametrize(
    ("model_cls", "field_name", "value"),
    [
        (SearchRequest, "extra_field", "value"),
        (SearchResult, "extra_field", "value"),
        (Doc, "extra_field", "value"),
    ],
)
def test_model_extra_fields_forbidden(
    model_cls: type[Doc | SearchRequest | SearchResult], field_name: str, value: str
) -> None:
    """Test that models reject extra fields (extra="forbid").

    Scenario: Schema validation for agent session state
    - Maps to Requirement: Typed JSON Contracts (R4)
    """
    with pytest.raises(Exception, match="Extra inputs are not permitted"):  # type: ignore[misc]
        model_cls(**{field_name: value})  # type: ignore[call-arg]
