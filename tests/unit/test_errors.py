"""Unit tests for kgfoundry_common.errors exception hierarchy.

These tests verify exception taxonomy, Problem Details mapping, and
cause chain preservation as required by R2.

Requirement: R2 — Exception Taxonomy & Problem Details
Scenario: Blind excepts eliminated
Scenario: HTTP error surfaces Problem Details
"""

from __future__ import annotations

import json
import logging

import pytest

from kgfoundry_common.errors import (
    ChunkingError,
    ConfigurationError,
    DoclingError,
    DownloadError,
    EmbeddingError,
    ErrorCode,
    IndexBuildError,
    KgFoundryError,
    LinkerCalibrationError,
    Neo4jError,
    OCRTimeoutError,
    OntologyParseError,
    SpladeOOMError,
    UnsupportedMIMEError,
    get_type_uri,
)


class TestErrorCode:
    """Test suite for ErrorCode enum and type URI mapping."""

    def test_error_code_values(self) -> None:
        """Verify error codes use kebab-case."""
        assert ErrorCode.DOWNLOAD_FAILED.value == "download-failed"
        assert ErrorCode.SEARCH_INDEX_MISSING.value == "search-index-missing"

    def test_get_type_uri(self) -> None:
        """Verify type URI generation."""
        uri = get_type_uri(ErrorCode.DOWNLOAD_FAILED)
        assert uri == "https://kgfoundry.dev/problems/download-failed"
        assert uri.startswith("https://kgfoundry.dev/problems/")

    def test_error_code_str(self) -> None:
        """Verify string representation."""
        assert str(ErrorCode.DOWNLOAD_FAILED) == "download-failed"


class TestKgFoundryError:
    """Test suite for base KgFoundryError class.

    Requirement: R2 — Exception Taxonomy & Problem Details
    """

    def test_basic_initialization(self) -> None:
        """Verify basic error creation."""
        error = KgFoundryError("Test message", ErrorCode.RUNTIME_ERROR)
        assert error.message == "Test message"
        assert error.code == ErrorCode.RUNTIME_ERROR
        assert error.http_status == 500
        assert error.log_level == logging.ERROR

    def test_custom_http_status(self) -> None:
        """Verify custom HTTP status code."""
        error = KgFoundryError(
            "Not found",
            ErrorCode.RESOURCE_UNAVAILABLE,
            http_status=404,
        )
        assert error.http_status == 404

    def test_cause_preservation(self) -> None:
        """Verify cause chain preservation."""
        original = ValueError("Original error")
        error = KgFoundryError("Wrapper", cause=original)
        assert error.__cause__ is original
        assert "caused by: ValueError" in str(error)

    def test_context_field(self) -> None:
        """Verify context dictionary."""
        error = KgFoundryError(
            "Error",
            context={"field": "value", "count": 42},
        )
        assert error.context == {"field": "value", "count": 42}

    def test_to_problem_details_minimal(self) -> None:
        """Verify minimal Problem Details structure."""
        error = KgFoundryError("Test", ErrorCode.RUNTIME_ERROR)
        details = error.to_problem_details()
        assert details["type"] == "https://kgfoundry.dev/problems/runtime-error"
        assert details["title"] == "KgFoundryError"
        assert details["status"] == 500
        assert details["detail"] == "Test"
        assert details["code"] == "runtime-error"

    def test_to_problem_details_with_instance(self) -> None:
        """Verify Problem Details with instance URI."""
        error = KgFoundryError("Test", ErrorCode.DOWNLOAD_FAILED)
        details = error.to_problem_details(instance="/api/download/123")
        assert details["instance"] == "/api/download/123"

    def test_to_problem_details_with_custom_title(self) -> None:
        """Verify custom title override."""
        error = KgFoundryError("Test", ErrorCode.DOWNLOAD_FAILED)
        details = error.to_problem_details(title="Custom Title")
        assert details["title"] == "Custom Title"

    def test_to_problem_details_with_context(self) -> None:
        """Verify errors field includes context."""
        error = KgFoundryError(
            "Test",
            ErrorCode.DOWNLOAD_FAILED,
            context={"url": "https://example.com", "retries": 3},
        )
        details = error.to_problem_details()
        assert "errors" in details
        assert details["errors"] == {"url": "https://example.com", "retries": 3}

    def test_problem_details_json_serializable(self) -> None:
        """Verify Problem Details can be JSON serialized."""
        error = KgFoundryError(
            "Test",
            ErrorCode.DOWNLOAD_FAILED,
            context={"nested": {"key": "value"}},
        )
        details = error.to_problem_details(instance="/test")
        # Should not raise
        json_str = json.dumps(details)
        assert "download-failed" in json_str


class TestSpecificExceptions:
    """Test suite for specific exception types.

    Requirement: R2 — Exception Taxonomy & Problem Details
    """

    @pytest.mark.parametrize(
        ("exception_cls", "expected_code", "expected_status"),
        [
            (DownloadError, ErrorCode.DOWNLOAD_FAILED, 503),
            (UnsupportedMIMEError, ErrorCode.UNSUPPORTED_MIME, 415),
            (DoclingError, ErrorCode.DOCLING_ERROR, 422),
            (OCRTimeoutError, ErrorCode.OCR_TIMEOUT, 504),
            (ChunkingError, ErrorCode.CHUNKING_ERROR, 422),
            (EmbeddingError, ErrorCode.EMBEDDING_ERROR, 503),
            (SpladeOOMError, ErrorCode.SPLADE_OOM, 507),
            (IndexBuildError, ErrorCode.INDEX_BUILD_ERROR, 500),
            (OntologyParseError, ErrorCode.ONTOLOGY_PARSE_ERROR, 422),
            (LinkerCalibrationError, ErrorCode.LINKER_CALIBRATION_ERROR, 500),
            (Neo4jError, ErrorCode.NEO4J_ERROR, 503),
            (ConfigurationError, ErrorCode.CONFIGURATION_ERROR, 500),
        ],
    )
    def test_exception_properties(
        self,
        exception_cls: type[KgFoundryError],
        expected_code: ErrorCode,
        expected_status: int,
    ) -> None:
        """Verify each exception type has correct code and status."""
        error = exception_cls("Test message")
        assert error.code == expected_code
        assert error.http_status == expected_status
        assert isinstance(error, KgFoundryError)

    def test_cause_propagation(self) -> None:
        """Verify cause is preserved in specific exceptions."""
        original = OSError("Connection refused")
        error = DownloadError("Download failed", cause=original)
        assert error.__cause__ is original
        assert error.code == ErrorCode.DOWNLOAD_FAILED

    def test_context_propagation(self) -> None:
        """Verify context is preserved."""
        error = DownloadError(
            "Failed",
            context={"url": "https://example.com"},
        )
        assert error.context == {"url": "https://example.com"}
        details = error.to_problem_details()
        assert details["errors"] == {"url": "https://example.com"}


class TestRaiseFromPattern:
    """Test suite for raise ... from exc pattern.

    Requirement: R2 — Exception Taxonomy & Problem Details
    Scenario: Blind excepts eliminated
    """

    def test_raise_from_preserves_chain(self) -> None:
        """Verify raise ... from exc preserves exception chain."""
        inner_msg = "Inner error"
        outer_msg = "Download failed"
        try:
            try:
                raise ValueError(inner_msg)
            except ValueError as exc:
                raise DownloadError(outer_msg, cause=exc) from exc
        except DownloadError as outer:
            assert outer.__cause__ is not None
            assert isinstance(outer.__cause__, ValueError)
            assert str(outer.__cause__) == inner_msg

    def test_problem_details_includes_cause_info(self) -> None:
        """Verify Problem Details can reference cause."""
        original = OSError("Network error")
        error = DownloadError("Download failed", cause=original)
        error.to_problem_details()  # Verify it doesn't raise
        # Cause should be preserved
        assert error.__cause__ is original
        # String representation should mention cause type
        assert "OSError" in str(error)
        assert "caused by" in str(error)
