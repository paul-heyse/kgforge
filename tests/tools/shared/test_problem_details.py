"""Tests for tools._shared.problem_details module."""

from __future__ import annotations

import json

from tools._shared.problem_details import (
    build_problem_details,
    problem_from_exception,
    render_problem,
)


class TestBuildProblemDetails:
    """Tests for build_problem_details function."""

    def test_builds_minimal_problem_details(self) -> None:
        """build_problem_details creates valid RFC 9457 payload."""
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/tool-failure",
            title="Tool failed",
            status=500,
            detail="Command failed",
            instance="urn:tool:git:exit-1",
        )

        assert problem["type"] == "https://kgfoundry.dev/problems/tool-failure"
        assert problem["title"] == "Tool failed"
        assert problem["status"] == 500
        assert problem["detail"] == "Command failed"
        assert problem["instance"] == "urn:tool:git:exit-1"

    def test_includes_extensions(self) -> None:
        """build_problem_details includes extension fields."""
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/tool-timeout",
            title="Timeout",
            status=504,
            detail="Command timed out",
            instance="urn:tool:git:timeout",
            extensions={"command": ["git", "status"], "timeout": 10.0},
        )

        assert problem["command"] == ["git", "status"]
        assert problem["timeout"] == 10.0

    def test_no_extensions_when_none_provided(self) -> None:
        """build_problem_details doesn't include extensions when None."""
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/tool-failure",
            title="Tool failed",
            status=500,
            detail="Command failed",
            instance="urn:tool:git:exit-1",
            extensions=None,
        )

        assert "extensions" not in problem or problem.get("extensions") is None


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
        )

        assert problem["type"] == "https://kgfoundry.dev/problems/invalid-input"
        assert problem["title"] == "Invalid input"
        assert problem["status"] == 400
        assert "Invalid input provided" in problem["detail"]
        assert problem["exception_type"] == "ValueError"

    def test_includes_custom_extensions(self) -> None:
        """problem_from_exception merges custom extensions."""
        exc = RuntimeError("Operation failed")
        problem = problem_from_exception(
            exc,
            type="https://kgfoundry.dev/problems/runtime-error",
            title="Runtime error",
            status=500,
            instance="urn:runtime:error",
            extensions={"error_code": "E123", "context": "test"},
        )

        assert problem["exception_type"] == "RuntimeError"
        assert problem["error_code"] == "E123"
        assert problem["context"] == "test"


class TestRenderProblem:
    """Tests for render_problem function."""

    def test_renders_as_json_string(self) -> None:
        """render_problem returns valid JSON string."""
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/tool-failure",
            title="Tool failed",
            status=500,
            detail="Command failed",
            instance="urn:tool:git:exit-1",
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
            type="https://kgfoundry.dev/problems/tool-timeout",
            title="Timeout",
            status=504,
            detail="Command timed out",
            instance="urn:tool:git:timeout",
            extensions={"command": ["git", "status"], "timeout": 10.0},
        )

        json_str = render_problem(problem)
        parsed = json.loads(json_str)
        assert parsed["command"] == ["git", "status"]
        assert parsed["timeout"] == 10.0
