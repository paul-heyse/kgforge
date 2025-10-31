"""HTTP Problem Details integration tests.

These tests verify FastAPI exception handlers convert KgFoundryError
to RFC 9457 Problem Details responses.

Requirement: R2 — Exception Taxonomy & Problem Details
Scenario: HTTP error surfaces Problem Details
"""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kgfoundry_common.errors import DownloadError, ErrorCode, KgFoundryError
from kgfoundry_common.errors.http import problem_details_response, register_problem_details_handler


class TestProblemDetailsResponse:
    """Test suite for Problem Details HTTP responses."""

    def test_basic_response(self) -> None:
        """Verify basic Problem Details response structure."""
        error = DownloadError("Download failed")
        response = problem_details_response(error)

        assert response.status_code == 503
        assert response.headers["Content-Type"] == "application/problem+json"

        content = json.loads(response.body.decode())
        assert content["type"] == "https://kgfoundry.dev/problems/download-failed"
        assert content["title"] == "DownloadError"
        assert content["status"] == 503
        assert content["detail"] == "Download failed"
        assert content["code"] == "download-failed"

    def test_response_with_instance(self) -> None:
        """Verify instance URI is included when request provided."""
        error = DownloadError("Download failed")
        # Just verify the function accepts request parameter
        # Full instance test is covered in test_handler_includes_instance
        response = problem_details_response(error, request=None)
        content = json.loads(response.body.decode())
        # Without request, instance should be None (not in content)
        assert "instance" not in content or content.get("instance") is None

    def test_response_with_context(self) -> None:
        """Verify errors field includes context."""
        error = DownloadError(
            "Download failed",
            context={"url": "https://example.com", "retries": 3},
        )
        response = problem_details_response(error)

        content = json.loads(response.body.decode())
        assert "errors" in content
        assert content["errors"] == {"url": "https://example.com", "retries": 3}


class TestFastAPIHandler:
    """Test suite for FastAPI exception handler integration."""

    def test_handler_registration(self) -> None:
        """Verify handler converts exceptions to Problem Details."""
        app = FastAPI()
        register_problem_details_handler(app)

        @app.get("/test-error")
        def test_endpoint() -> None:
            msg = "Test error"
            raise DownloadError(msg)

        client = TestClient(app)
        response = client.get("/test-error")

        assert response.status_code == 503
        assert response.headers["content-type"] == "application/problem+json"

        content = response.json()
        assert content["type"] == "https://kgfoundry.dev/problems/download-failed"
        assert content["status"] == 503
        assert content["detail"] == "Test error"
        assert content["instance"] == "/test-error"

    def test_handler_preserves_status_code(self) -> None:
        """Verify HTTP status code matches error.http_status."""
        app = FastAPI()
        register_problem_details_handler(app)

        @app.get("/not-found")
        def not_found() -> None:
            msg = "Resource not found"
            raise KgFoundryError(
                msg,
                code=ErrorCode.RESOURCE_UNAVAILABLE,
                http_status=404,
            )

        client = TestClient(app)
        response = client.get("/not-found")

        assert response.status_code == 404
        content = response.json()
        assert content["status"] == 404

    def test_handler_includes_instance(self) -> None:
        """Verify instance URI is correctly set from request."""
        app = FastAPI()
        register_problem_details_handler(app)

        @app.get("/api/search")
        def search() -> None:
            msg = "Search failed"
            raise DownloadError(msg)

        client = TestClient(app)
        response = client.get("/api/search?q=test")

        content = response.json()
        assert content["instance"] == "/api/search?q=test"
