"""Public wrapper for :mod:`tools._shared.problem_details`."""

from __future__ import annotations

from tools._shared.problem_details import (
    ExceptionProblemDetailsParams,
    JsonPrimitive,
    JsonValue,
    ProblemDetailsDict,
    ProblemDetailsParams,
    SchemaProblemDetailsParams,
    ToolProblemDetailsParams,
    build_problem_details,
    build_schema_problem_details,
    build_tool_problem_details,
    problem_from_exception,
    render_problem,
    tool_digest_mismatch_problem_details,
    tool_disallowed_problem_details,
    tool_failure_problem_details,
    tool_missing_problem_details,
    tool_timeout_problem_details,
)

__all__: tuple[str, ...] = (
    "ExceptionProblemDetailsParams",
    "JsonPrimitive",
    "JsonValue",
    "ProblemDetailsDict",
    "ProblemDetailsParams",
    "SchemaProblemDetailsParams",
    "ToolProblemDetailsParams",
    "build_problem_details",
    "build_schema_problem_details",
    "build_tool_problem_details",
    "problem_from_exception",
    "render_problem",
    "tool_digest_mismatch_problem_details",
    "tool_disallowed_problem_details",
    "tool_failure_problem_details",
    "tool_missing_problem_details",
    "tool_timeout_problem_details",
)
