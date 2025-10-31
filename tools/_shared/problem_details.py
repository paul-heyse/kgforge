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
from collections.abc import Mapping, Sequence
from pathlib import Path

__all__ = [
    "JsonPrimitive",
    "JsonValue",
    "ProblemDetailsDict",
    "build_problem_details",
    "build_tool_problem_details",
    "problem_from_exception",
    "render_problem",
    "tool_disallowed_problem_details",
    "tool_failure_problem_details",
    "tool_missing_problem_details",
    "tool_timeout_problem_details",
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


def build_tool_problem_details(  # noqa: PLR0913
    *,
    category: str,
    command: Sequence[str],
    status: int,
    title: str,
    detail: str,
    instance_suffix: str,
    extensions: Mapping[str, JsonValue] | None = None,
) -> ProblemDetailsDict:
    """Return a Problem Details payload describing a tooling subprocess failure."""
    command_list = list(command)
    tool_name = Path(command_list[0]).name if command_list else "<unknown>"
    command_extension: JsonValue = [str(part) for part in command_list]
    merged_extensions: dict[str, JsonValue] = {"command": command_extension}
    if extensions:
        merged_extensions.update(dict(extensions))
    return build_problem_details(
        type=f"https://kgfoundry.dev/problems/{category}",
        title=title,
        status=status,
        detail=detail,
        instance=f"urn:tool:{tool_name}:{instance_suffix}",
        extensions=merged_extensions,
    )


def tool_timeout_problem_details(
    command: Sequence[str],
    *,
    timeout: float | None,
) -> ProblemDetailsDict:
    """Return Problem Details describing a tooling timeout."""
    detail = (
        f"Command '{command[0]}' timed out after {timeout} seconds"
        if command and timeout is not None
        else (f"Command '{command[0]}' timed out" if command else "Command timed out")
    )
    extensions: dict[str, JsonValue] = {}
    if timeout is not None:
        extensions["timeout"] = timeout
    return build_tool_problem_details(
        category="tool-timeout",
        command=command,
        status=504,
        title="Tool execution timed out",
        detail=detail,
        instance_suffix="timeout",
        extensions=extensions,
    )


def tool_missing_problem_details(
    command: Sequence[str],
    *,
    executable: str,
    detail: str,
) -> ProblemDetailsDict:
    """Return Problem Details describing a missing executable."""
    return build_tool_problem_details(
        category="tool-missing",
        command=command or [executable],
        status=500,
        title="Executable not found",
        detail=detail,
        instance_suffix="missing",
    )


def tool_disallowed_problem_details(
    command: Sequence[str],
    *,
    executable: Path,
    allowlist: Sequence[str],
) -> ProblemDetailsDict:
    """Return Problem Details describing an allowlisted executable violation."""
    return build_tool_problem_details(
        category="tool-exec-disallowed",
        command=command,
        status=403,
        title="Executable not allowed",
        detail=(
            f"Executable '{executable.name}' is not permitted by the TOOLS_EXEC_ALLOWLIST setting"
        ),
        instance_suffix="disallowed",
        extensions={
            "executable": str(executable),
            "allowlist": list(allowlist),
        },
    )


def tool_failure_problem_details(
    command: Sequence[str],
    *,
    returncode: int,
    detail: str,
) -> ProblemDetailsDict:
    """Return Problem Details describing a non-zero tooling exit code."""
    return build_tool_problem_details(
        category="tool-failure",
        command=command,
        status=500,
        title="Tool returned a non-zero exit code",
        detail=detail,
        instance_suffix=f"exit-{returncode}",
        extensions={"returncode": returncode},
    )


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
