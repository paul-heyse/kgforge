from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list[JsonValue] | dict[str, JsonValue]
ProblemDetailsDict = dict[str, JsonValue]

class ProblemDetailsParams:
    def __init__(
        self,
        *,
        type: str,
        title: str,
        status: int,
        detail: str,
        instance: str,
        extensions: Mapping[str, JsonValue] | None = ...,
    ) -> None: ...


class SchemaProblemDetailsParams:
    def __init__(
        self,
        *,
        base: ProblemDetailsParams,
        error: Exception,
        extensions: Mapping[str, JsonValue] | None = ...,
    ) -> None: ...


class ToolProblemDetailsParams:
    def __init__(
        self,
        *,
        category: str,
        command: Sequence[str],
        status: int,
        title: str,
        detail: str,
        instance_suffix: str,
        extensions: Mapping[str, JsonValue] | None = ...,
    ) -> None: ...


class ExceptionProblemDetailsParams:
    def __init__(
        self,
        *,
        base: ProblemDetailsParams,
        exception: Exception,
        extensions: Mapping[str, JsonValue] | None = ...,
    ) -> None: ...


def build_problem_details(
    params: ProblemDetailsParams,
) -> ProblemDetailsDict: ...
def build_schema_problem_details(
    params: SchemaProblemDetailsParams,
) -> ProblemDetailsDict: ...
def build_tool_problem_details(
    params: ToolProblemDetailsParams,
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
    params: ExceptionProblemDetailsParams,
) -> ProblemDetailsDict: ...
def render_problem(problem: ProblemDetailsDict) -> str: ...
