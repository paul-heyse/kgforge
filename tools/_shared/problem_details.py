"""Problem Details helpers for RFC 9457 compliance.

This module provides builders and helpers for creating RFC 9457 Problem Details
payloads used across tooling CLIs and error responses.

Examples
--------
>>> from tools._shared.problem_details import build_problem_details, render_problem
>>> problem = build_problem_details(
...     type="https://kgfoundry.dev/problems/tool-failure",
...     title="Tool failed",
...     status=500,
...     detail="Command exited with code 1",
...     instance="urn:tool:git:exit-1",
...     extensions={"command": ["git", "status"], "returncode": 1},
... )
>>> json_str = render_problem(problem)
>>> assert "tool-failure" in json_str
"""

from __future__ import annotations

import json
from collections.abc import Mapping

__all__ = [
    "JsonPrimitive",
    "JsonValue",
    "ProblemDetailsDict",
    "build_problem_details",
    "problem_from_exception",
    "render_problem",
]

# Type aliases for JSON values (RFC 7159 compatible)
JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]

ProblemDetailsDict = dict[str, JsonValue]


def build_problem_details(  # noqa: PLR0913
    *,
    type: str,  # noqa: A002
    title: str,
    status: int,
    detail: str,
    instance: str,
    extensions: Mapping[str, JsonValue] | None = None,
) -> ProblemDetailsDict:
    """Build an RFC 9457 Problem Details payload.

    This function constructs a Problem Details dictionary conforming to
    RFC 9457. All fields except extensions are required.

    Parameters
    ----------
    type : str
        Type URI identifying the problem (e.g., "https://kgfoundry.dev/problems/tool-failure").
    title : str
        Short summary of the problem.
    status : int
        HTTP status code (or equivalent for non-HTTP contexts).
    detail : str
        Human-readable explanation of the problem.
    instance : str
        URI reference identifying the specific occurrence (e.g., "urn:tool:git:exit-1").
    extensions : Mapping[str, JsonValue] | None, optional
        Additional problem-specific fields.

    Returns
    -------
    dict[str, JsonValue]
        Problem Details payload conforming to RFC 9457.

    Examples
    --------
    >>> problem = build_problem_details(
    ...     type="https://kgfoundry.dev/problems/tool-timeout",
    ...     title="Tool execution timed out",
    ...     status=504,
    ...     detail="Command 'git' timed out after 10.0 seconds",
    ...     instance="urn:tool:git:timeout",
    ...     extensions={"command": ["git", "status"], "timeout": 10.0},
    ... )
    >>> assert problem["type"] == "https://kgfoundry.dev/problems/tool-timeout"
    >>> assert problem["status"] == 504
    """
    payload: ProblemDetailsDict = {
        "type": type,
        "title": title,
        "status": status,
        "detail": detail,
        "instance": instance,
    }
    if extensions:
        for key, value in extensions.items():
            payload[key] = value
    return payload


def problem_from_exception(  # noqa: PLR0913
    exc: Exception,
    *,
    type: str,  # noqa: A002
    title: str,
    status: int,
    instance: str,
    extensions: Mapping[str, JsonValue] | None = None,
) -> ProblemDetailsDict:
    """Build Problem Details from an exception.

    This function extracts detail from the exception message and optionally
    includes exception type and traceback information in extensions.

    Parameters
    ----------
    exc : Exception
        Exception to convert.
    type : str
        Type URI identifying the problem.
    title : str
        Short summary of the problem.
    status : int
        HTTP status code.
    instance : str
        URI reference identifying the specific occurrence.
    extensions : Mapping[str, JsonValue] | None, optional
        Additional problem-specific fields.

    Returns
    -------
    dict[str, JsonValue]
        Problem Details payload.

    Examples
    --------
    >>> try:
    ...     raise ValueError("Invalid input")
    ... except ValueError as e:
    ...     problem = problem_from_exception(
    ...         e,
    ...         type="https://kgfoundry.dev/problems/invalid-input",
    ...         title="Invalid input",
    ...         status=400,
    ...         instance="urn:validation:input",
    ...     )
    >>> assert "Invalid input" in problem["detail"]
    """
    detail = str(exc)
    exc_type_name = exc.__class__.__name__
    merged_extensions: dict[str, JsonValue] = {"exception_type": exc_type_name}
    if extensions:
        for key, value in extensions.items():
            merged_extensions[key] = value
    return build_problem_details(
        type=type,
        title=title,
        status=status,
        detail=detail,
        instance=instance,
        extensions=merged_extensions if merged_extensions else None,
    )


def render_problem(problem: ProblemDetailsDict) -> str:
    """Render Problem Details as JSON string.

    This function serializes a Problem Details payload to JSON for stdout
    or HTTP response bodies.

    Parameters
    ----------
    problem : ProblemDetailsDict
        Problem Details payload.

    Returns
    -------
    str
        JSON-encoded Problem Details (minified, no trailing newline).

    Examples
    --------
    >>> problem = build_problem_details(
    ...     type="https://kgfoundry.dev/problems/tool-failure",
    ...     title="Tool failed",
    ...     status=500,
    ...     detail="Command failed",
    ...     instance="urn:tool:git:exit-1",
    ... )
    >>> json_str = render_problem(problem)
    >>> assert json_str.startswith("{")
    >>> assert "tool-failure" in json_str
    """
    return json.dumps(problem, default=str)
