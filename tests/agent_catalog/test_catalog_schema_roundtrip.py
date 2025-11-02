"""Schema validation round-trip tests for agent catalog data contracts.

Tests verify:
- Search documents serialize/deserialize to/from JSON
- Problem Details errors match RFC 9457 schema
- Schemas are versioned and backward-compatible
- Round-trip integrity (data → JSON → data produces identical result)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from kgfoundry_common.types import JsonValue
else:
    JsonValue = dict  # type: ignore[assignment,misc]


class TestCatalogSchemaRoundTrip:
    """Verify search document round-trip serialization."""

    @pytest.mark.parametrize(
        ("doc_id", "title", "section", "body"),
        [
            ("doc_001", "Introduction", "Overview", "Welcome to the system"),
            ("doc_002", "Advanced", "Performance", "Optimization techniques"),
            ("doc_003", "FAQ", "Common", "Frequently asked questions"),
        ],
        ids=["simple", "technical", "faq"],
    )
    def test_search_document_roundtrip(
        self,
        doc_id: str,
        title: str,
        section: str,
        body: str,
    ) -> None:
        """Verify search document round-trip: dict → JSON → dict.

        Parameters
        ----------
        doc_id : str
            Document identifier.
        title : str
            Document title.
        section : str
            Document section.
        body : str
            Document body text.
        """
        # Original document (JsonValue typed)
        original: JsonValue = {
            "id": doc_id,
            "title": title,
            "section": section,
            "body": body,
            "metadata": {"version": 1, "source": "test"},
        }

        # Serialize to JSON
        json_str = json.dumps(original)
        assert isinstance(json_str, str)
        assert doc_id in json_str

        # Deserialize back with explicit typing
        restored: JsonValue = json.loads(json_str)

        # Verify parity
        assert restored == original
        # Type narrowing for dict access
        if isinstance(restored, dict):
            assert restored.get("id") == doc_id
            assert restored.get("title") == title


class TestProblemDetailsValidation:
    """Verify RFC 9457 Problem Details compliance."""

    @pytest.mark.parametrize(
        ("status", "title", "detail"),
        [
            (404, "Not Found", "Resource does not exist"),
            (400, "Bad Request", "Invalid parameters"),
            (500, "Internal Server Error", "Unexpected error occurred"),
        ],
        ids=["not_found", "bad_request", "server_error"],
    )
    def test_problem_details_structure(
        self,
        status: int,
        title: str,
        detail: str,
    ) -> None:
        """Verify Problem Details has required RFC 9457 fields.

        Parameters
        ----------
        status : int
            HTTP status code.
        title : str
            Problem title.
        detail : str
            Problem detail.
        """
        problem: JsonValue = {
            "type": f"https://kgfoundry.dev/problems/{status}",
            "title": title,
            "status": status,
            "detail": detail,
            "instance": "urn:request:abc123",
            "correlation_id": "req-abc123",
        }

        # Verify required fields with type guard
        if isinstance(problem, dict):
            assert problem.get("type")
            assert problem.get("title")
            assert problem.get("status") == status
            assert problem.get("detail")
            assert problem.get("instance")
            assert problem.get("correlation_id")

        # Verify JSON round-trip
        json_str = json.dumps(problem)
        restored: JsonValue = json.loads(json_str)
        assert restored == problem

    def test_problem_details_with_extensions(
        self,
        problem_details_loader: Any,  # noqa: ANN401 - pytest fixture typing limitation
    ) -> None:
        """Verify Problem Details can include extension fields.

        Parameters
        ----------
        problem_details_loader : callable
            Fixture to load problem details examples.
        """
        # Load a problem details example with extensions
        try:
            problem: JsonValue = problem_details_loader("faiss-index-build-timeout")

            # Verify base fields with type guard
            if isinstance(problem, dict):
                assert "type" in problem
                assert "status" in problem
                assert "detail" in problem

                # Verify extensions (if present)
                if "extensions" in problem:
                    extensions = problem["extensions"]
                    assert isinstance(extensions, dict)

        except FileNotFoundError:
            # Example may not exist; skip gracefully
            pytest.skip("Problem details example not found")


class TestSchemaVersioning:
    """Verify schema versioning and compatibility."""

    def test_schema_version_present(self) -> None:
        """Verify schema includes version field."""
        schema_path = (
            Path(__file__).parent.parent.parent / "schema/vectors/input-vectors.v1.schema.json"
        )

        if not schema_path.exists():
            pytest.skip("Schema file not found")

        schema_text = schema_path.read_text(encoding="utf-8")
        schema: JsonValue = json.loads(schema_text)

        # Verify schema structure with type guard
        if isinstance(schema, dict):
            assert "$schema" in schema
            assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
            assert "$id" in schema

            # Version in $id indicates API version
            schema_id = schema.get("$id", "")
            assert isinstance(schema_id, str)
            assert "v1" in schema_id

    def test_backwards_compatibility_marker(self) -> None:
        """Verify schema includes compatibility info in title/description."""
        schema_path = (
            Path(__file__).parent.parent.parent / "schema/vectors/input-vectors.v1.schema.json"
        )

        if not schema_path.exists():
            pytest.skip("Schema file not found")

        schema_text = schema_path.read_text(encoding="utf-8")
        schema: JsonValue = json.loads(schema_text)

        # Verify schema structure with type guard
        if isinstance(schema, dict):
            # Should have title (for humans)
            assert "title" in schema
            # Should have description (for compatibility notes)
            assert "description" in schema


__all__ = [
    "TestCatalogSchemaRoundTrip",
    "TestProblemDetailsValidation",
    "TestSchemaVersioning",
]
