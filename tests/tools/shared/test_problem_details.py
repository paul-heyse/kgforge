"""Tests for tools._shared.problem_details module."""

from __future__ import annotations

import json

from tools import (
    build_problem_details,
    build_schema_problem_details,
    problem_from_exception,
    render_problem,
)


def test_build_problem_details_minimal() -> None:
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


def test_build_problem_details_includes_extensions() -> None:
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


def test_build_problem_details_omits_none_extensions() -> None:
    problem = build_problem_details(
        type="https://kgfoundry.dev/problems/tool-failure",
        title="Tool failed",
        status=500,
        detail="Command failed",
        instance="urn:tool:git:exit-1",
        extensions=None,
    )

    assert "extensions" not in problem or problem.get("extensions") is None


def test_build_schema_problem_details_includes_pointer_and_validator() -> None:
    class _FakeValidationError(Exception):
        def __init__(self) -> None:
            super().__init__("Invalid payload")
            self.message = "Invalid payload"
            self.absolute_path = ("entries", 0, "name")
            self.validator = "required"

    error = _FakeValidationError()

    problem = build_schema_problem_details(
        error=error,
        type="https://kgfoundry.dev/problems/example-schema-mismatch",
        title="Schema validation failed",
        status=422,
        instance="urn:example:schema:validation",
    )

    assert problem["type"] == "https://kgfoundry.dev/problems/example-schema-mismatch"
    assert problem["status"] == 422
    assert problem["detail"] == "Invalid payload"
    extensions = problem.get("extensions")
    assert isinstance(extensions, dict)
    assert extensions["jsonPointer"] == "/entries/0/name"
    assert extensions["validator"] == "required"


def test_problem_from_exception_converts_exception() -> None:
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


def test_problem_from_exception_includes_custom_extensions() -> None:
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


def test_render_problem_returns_json() -> None:
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
    parsed = json.loads(json_str)
    assert parsed["type"] == problem["type"]


def test_render_problem_handles_extensions() -> None:
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
