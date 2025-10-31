from __future__ import annotations

from collections.abc import Mapping

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
