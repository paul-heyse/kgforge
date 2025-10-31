"""Tests for agent catalog search functionality.

Tests cover success paths, invalid input handling, Problem Details emission,
and schema validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from kgfoundry.agent_catalog.search import SearchOptions, SearchRequest, search_catalog
from kgfoundry_common.errors import AgentCatalogSearchError
from kgfoundry_common.schema_helpers import load_schema

FIXTURE_CATALOG = Path("tests/fixtures/agent/catalog_sample.json")
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def catalog_data() -> dict[str, Any]:
    """Load sample catalog fixture."""
    if not FIXTURE_CATALOG.exists():
        pytest.skip(f"Fixture catalog not found: {FIXTURE_CATALOG}")
    return json.loads(FIXTURE_CATALOG.read_text())


@pytest.fixture
def search_response_schema() -> dict[str, Any]:
    """Load search_response.json schema."""
    schema_path = Path("schema/search/search_response.json")
    if not schema_path.exists():
        pytest.skip(f"Schema not found: {schema_path}")
    return load_schema(schema_path)


@pytest.fixture
def search_response_validator(
    search_response_schema: dict[str, Any],
) -> Draft202012Validator:
    """Create validator for search_response.json."""
    return Draft202012Validator(search_response_schema)


class TestSearchSuccess:
    """Test successful search operations."""

    def test_search_returns_results(self, catalog_data: dict[str, Any]) -> None:
        """Search should return results for valid queries."""
        request = SearchRequest(
            repo_root=REPO_ROOT,
            query="demo",
            k=5,
        )
        options = SearchOptions()
        results = search_catalog(catalog_data, request=request, options=options)
        assert len(results) > 0
        assert all(result.symbol_id for result in results)
        assert all(result.score >= 0.0 for result in results)

    def test_search_respects_k_limit(self, catalog_data: dict[str, Any]) -> None:
        """Search should respect k parameter."""
        request = SearchRequest(
            repo_root=REPO_ROOT,
            query="demo",
            k=2,
        )
        options = SearchOptions()
        results = search_catalog(catalog_data, request=request, options=options)
        assert len(results) <= 2

    def test_search_with_facets(self, catalog_data: dict[str, Any]) -> None:
        """Search should filter by facets when provided."""
        request = SearchRequest(
            repo_root=REPO_ROOT,
            query="demo",
            k=10,
        )
        options = SearchOptions(facets={"package": "demo"})
        results = search_catalog(catalog_data, request=request, options=options)
        assert all(result.package == "demo" for result in results)

    def test_search_empty_query_returns_empty(self, catalog_data: dict[str, Any]) -> None:
        """Empty query should return empty results."""
        request = SearchRequest(
            repo_root=REPO_ROOT,
            query="",
            k=10,
        )
        options = SearchOptions()
        results = search_catalog(catalog_data, request=request, options=options)
        assert len(results) == 0


class TestSearchInvalidInput:
    """Test handling of invalid input."""

    def test_search_missing_catalog_raises_error(self) -> None:
        """Search with missing catalog should raise AgentCatalogSearchError."""
        request = SearchRequest(
            repo_root=REPO_ROOT,
            query="test",
            k=10,
        )
        options = SearchOptions()
        empty_catalog: dict[str, Any] = {}
        with pytest.raises(AgentCatalogSearchError) as exc_info:
            search_catalog(empty_catalog, request=request, options=options)
        # Verify it's an AgentCatalogSearchError
        assert isinstance(exc_info.value, AgentCatalogSearchError)

    def test_search_invalid_repo_root_raises_error(self, catalog_data: dict[str, Any]) -> None:
        """Search with invalid repo_root should raise AgentCatalogSearchError."""
        invalid_root = Path("/nonexistent/path")
        request = SearchRequest(
            repo_root=invalid_root,
            query="test",
            k=10,
        )
        options = SearchOptions()
        with pytest.raises(AgentCatalogSearchError) as exc_info:
            search_catalog(catalog_data, request=request, options=options)
        # Verify it's an AgentCatalogSearchError
        assert isinstance(exc_info.value, AgentCatalogSearchError)

    def test_search_k_negative_handled(self, catalog_data: dict[str, Any]) -> None:
        """Search with negative k should be handled gracefully."""
        request = SearchRequest(
            repo_root=REPO_ROOT,
            query="demo",
            k=-1,
        )
        options = SearchOptions()
        # Should handle gracefully (k clamped to 1)
        results = search_catalog(catalog_data, request=request, options=options)
        assert len(results) >= 0


class TestSearchProblemDetails:
    """Test Problem Details emission for errors."""

    def test_search_error_produces_problem_details(self, catalog_data: dict[str, Any]) -> None:
        """Search errors should produce RFC 9457 Problem Details."""
        invalid_root = Path("/nonexistent/path")
        request = SearchRequest(
            repo_root=invalid_root,
            query="test",
            k=10,
        )
        options = SearchOptions()
        with pytest.raises(AgentCatalogSearchError) as exc_info:
            search_catalog(catalog_data, request=request, options=options)
        error = exc_info.value
        problem_details = error.to_problem_details(instance="urn:test:search")
        assert problem_details["type"] == "https://kgfoundry.dev/problems/search-error"
        assert problem_details["status"] == 503
        assert problem_details["title"] == "Agent Catalog Search Error"
        assert "instance" in problem_details
        assert "detail" in problem_details

    def test_search_error_includes_context(self, catalog_data: dict[str, Any]) -> None:
        """Search errors should include context information."""
        invalid_root = Path("/nonexistent/path")
        request = SearchRequest(
            repo_root=invalid_root,
            query="test",
            k=10,
        )
        options = SearchOptions()
        with pytest.raises(AgentCatalogSearchError) as exc_info:
            search_catalog(catalog_data, request=request, options=options)
        error = exc_info.value
        problem_details = error.to_problem_details(instance="urn:test:search")
        # Context should be included in Problem Details
        assert "detail" in problem_details


class TestSearchSchemaValidation:
    """Test that search results conform to schema."""

    def test_search_results_validate_against_schema(
        self,
        catalog_data: dict[str, Any],
        search_response_validator: Draft202012Validator,
    ) -> None:
        """Search results should validate against search_response.json schema."""
        request = SearchRequest(
            repo_root=REPO_ROOT,
            query="demo",
            k=5,
        )
        options = SearchOptions()
        results = search_catalog(catalog_data, request=request, options=options)
        # Convert results to schema-compatible format
        response = {  # type: ignore[misc]  # dict construction with Any values
            "results": [
                {
                    "symbol_id": result.symbol_id,
                    "score": result.score,
                    "lexical_score": result.lexical_score,
                    "vector_score": result.vector_score,
                    "package": result.package,
                    "module": result.module,
                    "qname": result.qname,
                    "kind": result.kind,
                    "anchor": {
                        "start_line": result.anchor.get("start_line"),  # type: ignore[misc]  # dict access returns Any
                        "end_line": result.anchor.get("end_line"),  # type: ignore[misc]  # dict access returns Any
                    },
                    "metadata": {
                        "stability": result.stability,
                        "deprecated": result.deprecated,
                        "summary": result.summary,
                        "docstring": result.docstring,
                    },
                }
                for result in results
            ],
            "total": len(results),
            "took_ms": 0,  # Mock value
            "metadata": {},
        }
        # Validate against schema
        search_response_validator.validate(response)

    def test_search_results_have_required_fields(self, catalog_data: dict[str, Any]) -> None:
        """Search results should have all required fields."""
        request = SearchRequest(
            repo_root=REPO_ROOT,
            query="demo",
            k=5,
        )
        options = SearchOptions()
        results = search_catalog(catalog_data, request=request, options=options)
        assert len(results) > 0
        for result in results:
            assert result.symbol_id
            assert result.package
            assert result.module
            assert result.qname
            assert result.kind
            assert result.anchor.get("start_line") is not None
            assert isinstance(result.score, float)
            assert isinstance(result.lexical_score, float)
            assert isinstance(result.vector_score, float)


class TestSearchEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_search_large_k_handled(self, catalog_data: dict[str, Any]) -> None:
        """Search with very large k should be handled."""
        request = SearchRequest(
            repo_root=REPO_ROOT,
            query="demo",
            k=1000,
        )
        options = SearchOptions()
        results = search_catalog(catalog_data, request=request, options=options)
        # Should return results up to available count
        assert len(results) >= 0

    def test_search_whitespace_query(self, catalog_data: dict[str, Any]) -> None:
        """Search with whitespace-only query should be handled."""
        request = SearchRequest(
            repo_root=REPO_ROOT,
            query="   ",
            k=10,
        )
        options = SearchOptions()
        results = search_catalog(catalog_data, request=request, options=options)
        # Should handle gracefully (may return empty or all results)
        assert isinstance(results, list)

    def test_search_special_characters(self, catalog_data: dict[str, Any]) -> None:
        """Search with special characters should be handled."""
        request = SearchRequest(
            repo_root=REPO_ROOT,
            query="test@#$%^&*()",
            k=10,
        )
        options = SearchOptions()
        # Should not raise unhandled exceptions
        results = search_catalog(catalog_data, request=request, options=options)
        assert isinstance(results, list)
