from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list[JsonValue] | dict[str, JsonValue]
ProblemDetailsDict = dict[str, JsonValue]

def build_problem_details(
    *,
    type: str,
    title: str,
    status: int,
    detail: str,
    instance: str,
    extensions: Mapping[str, JsonValue] | None = ...,
) -> ProblemDetailsDict: ...
def build_schema_problem_details(
    *,
    error: Exception,
    type: str,
    title: str,
    status: int,
    instance: str,
    extensions: Mapping[str, JsonValue] | None = ...,
) -> ProblemDetailsDict: ...
def build_tool_problem_details(
    *,
    category: str,
    command: Sequence[str],
    status: int,
    title: str,
    detail: str,
    instance_suffix: str,
    extensions: Mapping[str, JsonValue] | None = ...,
) -> ProblemDetailsDict: ...
def tool_timeout_problem_details(
    command: Sequence[str],
    *,
    timeout: float | None,
) -> ProblemDetailsDict: ...
def tool_missing_problem_details(
    command: Sequence[str],
    *,
    executable: str,
    detail: str,
) -> ProblemDetailsDict: ...
def tool_disallowed_problem_details(
    command: Sequence[str],
    *,
    executable: Path,
    allowlist: Sequence[str],
) -> ProblemDetailsDict: ...
def tool_failure_problem_details(
    command: Sequence[str],
    *,
    returncode: int,
    detail: str,
) -> ProblemDetailsDict: ...
def problem_from_exception(
    exc: Exception,
    *,
    type: str,
    title: str,
    status: int,
    instance: str,
    extensions: Mapping[str, JsonValue] | None = ...,
) -> ProblemDetailsDict: ...
def render_problem(problem: ProblemDetailsDict) -> str: ...
