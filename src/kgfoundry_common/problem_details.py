"""RFC 9457 Problem Details helpers with schema validation.

This module provides typed helpers for building RFC 9457 Problem Details payloads
with JSON Schema 2020-12 validation. All payloads validate against the canonical
schema at `schema/common/problem_details.json`.

Examples
--------
>>> from kgfoundry_common.problem_details import build_problem_details, render_problem
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
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import jsonschema
from jsonschema.exceptions import SchemaError, ValidationError

from kgfoundry_common.fs import read_text
from kgfoundry_common.logging import get_logger

# Type aliases for JSON values (RFC 7159 compatible)
# JsonPrimitive represents the basic JSON types
JsonPrimitive = str | int | float | bool | None

# Recursive JSON type alias (used only for type checking)
if TYPE_CHECKING:
    JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
else:
    JsonValue = object  # type: ignore[assignment, misc]

JsonObject = dict[str, JsonValue]

# JSON Schema type (from jsonschema stubs: Mapping[str, object])
JsonSchema = Mapping[str, object]

__all__ = [
    "JsonObject",
    "JsonPrimitive",
    "JsonValue",
    "ProblemDetails",
    "ProblemDetailsValidationError",
    "build_problem_details",
    "problem_from_exception",
    "render_problem",
    "validate_problem_details",
]

logger = get_logger(__name__)

# Path to canonical Problem Details schema
_SCHEMA_PATH = Path(__file__).parent.parent.parent / "schema" / "common" / "problem_details.json"


class ProblemDetails(TypedDict, total=False):
    """Describe ProblemDetails.

    &lt;!-- auto:docstring-builder v1 --&gt;

    Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.
    """

    type: str
    title: str
    status: int
    detail: str
    instance: str
    code: str  # NotRequired via total=False
    extensions: dict[str, JsonValue]  # NotRequired via total=False


class ProblemDetailsValidationError(Exception):
    """Raised when Problem Details payload fails schema validation.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    message : str
        Describe ``message``.
    validation_errors : list[str] | None, optional
        Describe ``validation_errors``.
        Defaults to ``None``.

    Examples
    --------
    >>> raise ProblemDetailsValidationError("Missing required field: type")
    """

    def __init__(self, message: str, validation_errors: list[str] | None = None) -> None:
        """Initialize validation error.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        message : str
            Error message describing the validation failure.
        validation_errors : list[str] | NoneType, optional
            List of specific validation error messages. Defaults to None.
            Defaults to ``None``.
        """
        super().__init__(message)
        self.validation_errors = validation_errors or []


@lru_cache(maxsize=1)  # type: ignore[misc]  # lru_cache is a descriptor; mypy handles its type correctly
def _load_schema() -> JsonSchema:
    """Load and cache the Problem Details schema.

    <!-- auto:docstring-builder v1 -->

    Returns
    -------
    str | object
        Parsed schema dictionary.

    Raises
    ------
    ProblemDetailsValidationError
        If schema file is missing or invalid.
    """
    if not _SCHEMA_PATH.exists():
        msg = f"Problem Details schema not found: {_SCHEMA_PATH}"
        raise ProblemDetailsValidationError(msg)

    try:
        schema_text = read_text(_SCHEMA_PATH)
        schema_obj: dict[str, object] = json.loads(schema_text)
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Failed to load Problem Details schema: {exc}"
        raise ProblemDetailsValidationError(msg) from exc

    # Validate against JSON Schema 2020-12 meta-schema
    try:
        jsonschema.Draft202012Validator.check_schema(schema_obj)
    except SchemaError as exc:
        msg = f"Invalid Problem Details schema: {exc.message}"
        raise ProblemDetailsValidationError(msg) from exc

    return schema_obj


def validate_problem_details(payload: dict[str, object]) -> None:
    """Validate Problem Details payload against canonical schema.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    payload : dict[str, object]
        Problem Details payload to validate.

    Raises
    ------
    ProblemDetailsValidationError
        If payload fails schema validation.

    Examples
    --------
    >>> problem = {
    ...     "type": "https://kgfoundry.dev/problems/runtime-error",
    ...     "title": "Runtime Error",
    ...     "status": 500,
    ...     "detail": "Operation failed",
    ...     "instance": "/api/v1/operation",
    ... }
    >>> validate_problem_details(problem)
    """
    schema = _load_schema()
    try:
        jsonschema.validate(instance=payload, schema=schema)
    except ValidationError as exc:
        errors = [exc.message]
        if exc.absolute_path:
            path_str = ".".join(str(p) for p in exc.absolute_path)
            errors.append(f"at path: {path_str}")
        msg = f"Problem Details validation failed: {'; '.join(errors)}"
        raise ProblemDetailsValidationError(msg, validation_errors=errors) from exc
    except SchemaError as exc:
        msg = f"Invalid schema: {exc.message}"
        raise ProblemDetailsValidationError(msg) from exc


def build_problem_details(  # noqa: PLR0913
    *,
    type: str,  # noqa: A002
    title: str,
    status: int,
    detail: str,
    instance: str,
    code: str | None = None,
    extensions: Mapping[str, JsonValue] | None = None,
) -> ProblemDetails:
    """Build an RFC 9457 Problem Details payload.

    <!-- auto:docstring-builder v1 -->

    This function constructs a Problem Details dictionary conforming to
    RFC 9457 and validates it against the canonical schema.

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
    code : str | NoneType, optional
        Machine-readable error code (kgfoundry extension). Defaults to None.
        Defaults to ``None``.
    extensions : str | object | NoneType, optional
        Additional problem-specific fields. Defaults to None.
        Defaults to ``None``.

    Returns
    -------
    ProblemDetails
        Problem Details payload conforming to RFC 9457 and validated against schema.

    Raises
    ------
    ProblemDetailsValidationError
        If the constructed payload fails schema validation.

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
    payload: dict[str, object] = {
        "type": type,
        "title": title,
        "status": status,
        "detail": detail,
        "instance": instance,
    }
    if code is not None:
        payload["code"] = code
    if extensions:
        payload["errors"] = dict(extensions)

    # Validate against schema
    validate_problem_details(payload)

    return payload  # type: ignore[return-value]


def problem_from_exception(  # noqa: PLR0913
    exc: Exception,
    *,
    type: str,  # noqa: A002
    title: str,
    status: int,
    instance: str,
    code: str | None = None,
    extensions: Mapping[str, JsonValue] | None = None,
) -> ProblemDetails:
    """Build Problem Details from an exception.

    <!-- auto:docstring-builder v1 -->

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
    code : str | NoneType, optional
        Machine-readable error code. Defaults to None.
        Defaults to ``None``.
    extensions : str | object | NoneType, optional
        Additional problem-specific fields. Defaults to None.
        Defaults to ``None``.

    Returns
    -------
    ProblemDetails
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
    merged_extensions: dict[str, JsonValue] = {
        "exception_type": exc_type_name,
    }
    if extensions:
        merged_extensions.update(extensions)

    # Preserve cause chain if present
    if exc.__cause__ is not None:
        merged_extensions["caused_by"] = exc.__cause__.__class__.__name__

    return build_problem_details(
        type=type,
        title=title,
        status=status,
        detail=detail,
        instance=instance,
        code=code,
        extensions=merged_extensions,
    )


def render_problem(problem: ProblemDetails | dict[str, object]) -> str:
    """Render Problem Details as JSON string.

    <!-- auto:docstring-builder v1 -->

    This function serializes a Problem Details payload to JSON for stdout
    or HTTP response bodies.

    Parameters
    ----------
    problem : ProblemDetails | dict[str, object]
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
    return json.dumps(problem, default=str, ensure_ascii=False)
