"""Tests for kgfoundry_common.problem_details module.

Tests cover schema validation, error handling, and Problem Details
construction from exceptions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kgfoundry_common.errors import ConfigurationError, DownloadError
from kgfoundry_common.problem_details import (
    ProblemDetailsValidationError,
    build_problem_details,
    problem_from_exception,
    render_problem,
    validate_problem_details,
)


class TestBuildProblemDetails:
    """Tests for build_problem_details function."""

    def test_builds_minimal_problem_details(self) -> None:
        """build_problem_details creates valid RFC 9457 payload."""
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/runtime-error",
            title="Runtime Error",
            status=500,
            detail="Operation failed",
            instance="urn:kgfoundry:error",
            code="runtime-error",
        )

        assert problem["type"] == "https://kgfoundry.dev/problems/runtime-error"
        assert problem["title"] == "Runtime Error"
        assert problem["status"] == 500
        assert problem["detail"] == "Operation failed"
        assert problem["instance"] == "urn:kgfoundry:error"
        assert problem["code"] == "runtime-error"

    def test_includes_extensions(self) -> None:
        """build_problem_details includes extension fields."""
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/download-failed",
            title="Download Failed",
            status=503,
            detail="Failed to download resource",
            instance="/api/v1/download",
            code="download-failed",
            extensions={"url": "https://example.com/file.pdf", "retry_after_seconds": 60},
        )

        assert problem["errors"]["url"] == "https://example.com/file.pdf"
        assert problem["errors"]["retry_after_seconds"] == 60

    def test_no_extensions_when_none_provided(self) -> None:
        """build_problem_details doesn't include extensions when None."""
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/runtime-error",
            title="Runtime Error",
            status=500,
            detail="Operation failed",
            instance="urn:kgfoundry:error",
            code="runtime-error",
            extensions=None,
        )

        assert "errors" not in problem

    def test_validates_against_schema(self) -> None:
        """build_problem_details validates payload against schema."""
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/configuration-error",
            title="Configuration Missing",
            status=500,
            detail="Missing required env var",
            instance="urn:kgfoundry:settings",
            code="configuration-error",
        )

        # Should not raise
        validate_problem_details(problem)

    def test_invalid_payload_raises_validation_error(self) -> None:
        """build_problem_details raises validation error for invalid payload."""
        with pytest.raises(ProblemDetailsValidationError, match="validation failed"):
            validate_problem_details(
                {
                    "type": "not-a-uri",
                    "title": "Missing required fields",
                    "status": 500,
                },
            )


class TestProblemFromException:
    """Tests for problem_from_exception function."""

    def test_converts_exception_to_problem_details(self) -> None:
        """problem_from_exception creates Problem Details from exception."""
        exc = ValueError("Invalid input provided")
        problem = problem_from_exception(
            exc,
            type="https://kgfoundry.dev/problems/invalid-input",
            title="Invalid input",
            status=400,
            instance="urn:validation:input",
            code="invalid-input",
        )

        assert problem["type"] == "https://kgfoundry.dev/problems/invalid-input"
        assert problem["title"] == "Invalid input"
        assert problem["status"] == 400
        assert "Invalid input provided" in problem["detail"]
        assert problem["errors"]["exception_type"] == "ValueError"

    def test_includes_custom_extensions(self) -> None:
        """problem_from_exception merges custom extensions."""
        exc = RuntimeError("Operation failed")
        problem = problem_from_exception(
            exc,
            type="https://kgfoundry.dev/problems/runtime-error",
            title="Runtime error",
            status=500,
            instance="urn:runtime:error",
            code="runtime-error",
            extensions={"error_code": "E123", "context": "test"},
        )

        assert problem["errors"]["exception_type"] == "RuntimeError"
        assert problem["errors"]["error_code"] == "E123"
        assert problem["errors"]["context"] == "test"

    def test_preserves_cause_chain(self) -> None:
        """problem_from_exception preserves exception cause chain."""
        inner_msg = "Inner error"
        outer_msg = "Outer error"

        # Create exception with cause chain manually
        inner_exc = ValueError(inner_msg)
        outer_exc = RuntimeError(outer_msg)
        outer_exc.__cause__ = inner_exc

        problem = problem_from_exception(
            outer_exc,
            type="https://kgfoundry.dev/problems/runtime-error",
            title="Runtime error",
            status=500,
            instance="urn:runtime:error",
            code="runtime-error",
        )

        assert problem["errors"]["caused_by"] == "ValueError"


class TestRenderProblem:
    """Tests for render_problem function."""

    def test_renders_as_json_string(self) -> None:
        """render_problem returns valid JSON string."""
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/runtime-error",
            title="Runtime Error",
            status=500,
            detail="Operation failed",
            instance="urn:kgfoundry:error",
            code="runtime-error",
        )

        json_str = render_problem(problem)
        assert isinstance(json_str, str)
        assert json_str.startswith("{")
        assert json_str.endswith("}")

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["type"] == problem["type"]

    def test_handles_extensions(self) -> None:
        """render_problem includes extension fields in JSON."""
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/download-failed",
            title="Download Failed",
            status=503,
            detail="Failed to download",
            instance="/api/v1/download",
            code="download-failed",
            extensions={"url": "https://example.com/file.pdf", "timeout": 10.0},
        )

        json_str = render_problem(problem)
        parsed = json.loads(json_str)
        assert parsed["errors"]["url"] == "https://example.com/file.pdf"
        assert parsed["errors"]["timeout"] == 10.0


class TestValidateProblemDetails:
    """Tests for validate_problem_details function."""

    def test_valid_payload_passes(self) -> None:
        """validate_problem_details accepts valid payload."""
        problem = {
            "type": "https://kgfoundry.dev/problems/runtime-error",
            "title": "Runtime Error",
            "status": 500,
            "detail": "Operation failed",
            "instance": "/api/v1/operation",
            "code": "runtime-error",
        }

        # Should not raise
        validate_problem_details(problem)

    def test_invalid_payload_raises(self) -> None:
        """validate_problem_details raises for invalid payload."""
        invalid = {
            "type": "not-a-uri",
            "title": "Missing fields",
            "status": 500,
        }

        with pytest.raises(ProblemDetailsValidationError, match="validation failed"):
            validate_problem_details(invalid)

    def test_missing_required_fields_raises(self) -> None:
        """validate_problem_details raises for missing required fields."""
        incomplete = {
            "type": "https://kgfoundry.dev/problems/runtime-error",
            "title": "Runtime Error",
            "status": 500,
            # Missing detail and instance
        }

        with pytest.raises(ProblemDetailsValidationError, match="required property"):
            validate_problem_details(incomplete)


class TestKgFoundryErrorIntegration:
    """Tests for integration with KgFoundryError exceptions."""

    def test_download_error_to_problem_details(self) -> None:
        """DownloadError converts to Problem Details."""
        error = DownloadError("Failed to download PDF", cause=OSError("Connection refused"))
        details = error.to_problem_details(instance="/api/v1/download")

        assert details["type"] == "https://kgfoundry.dev/problems/download-failed"
        assert details["status"] == 503
        assert details["code"] == "download-failed"
        assert details["detail"] == "Failed to download PDF"
        assert details["instance"] == "/api/v1/download"

    def test_configuration_error_to_problem_details(self) -> None:
        """ConfigurationError converts to Problem Details."""
        error = ConfigurationError("Missing required env var: KGFOUNDRY_API_KEY")
        details = error.to_problem_details(instance="urn:kgfoundry:settings")

        assert details["type"] == "https://kgfoundry.dev/problems/configuration-error"
        assert details["status"] == 500
        assert details["code"] == "configuration-error"

    def test_problem_details_validates_from_exception(self) -> None:
        """Problem Details from exceptions validate against schema."""
        error = DownloadError("Download failed")
        details = error.to_problem_details(instance="/api/v1/download")

        # Should not raise
        validate_problem_details(details)

    def test_error_with_context_includes_extensions(self) -> None:
        """Errors with context include extensions in Problem Details."""
        error = DownloadError(
            "Download failed",
            context={"url": "https://example.com/file.pdf", "retry_after_seconds": 60},
        )
        details = error.to_problem_details(instance="/api/v1/download")

        assert details["errors"]["url"] == "https://example.com/file.pdf"
        assert details["errors"]["retry_after_seconds"] == 60


class TestSchemaExamples:
    """Tests that example Problem Details JSON files validate."""

    def test_example_config_missing_validates(self) -> None:
        """Example config-missing.json validates against schema."""
        example_path = (
            Path(__file__).parent.parent.parent
            / "docs"
            / "examples"
            / "problem_details"
            / "config-missing.json"
        )
        if example_path.exists():
            example_data = json.loads(example_path.read_text())
            validate_problem_details(example_data)

    def test_example_download_failed_validates(self) -> None:
        """Example download-failed.json validates against schema."""
        example_path = (
            Path(__file__).parent.parent.parent
            / "docs"
            / "examples"
            / "problem_details"
            / "download-failed.json"
        )
        if example_path.exists():
            example_data = json.loads(example_path.read_text())
            validate_problem_details(example_data)
