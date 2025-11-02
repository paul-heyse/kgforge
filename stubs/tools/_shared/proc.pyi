from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools._shared.problem_details import ProblemDetailsDict

@dataclass
class ToolRunResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool

class ToolExecutionError(RuntimeError):
    command: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    problem: ProblemDetailsDict | None

    def __init__(
        self,
        message: str,
        *,
        command: Sequence[str],
        returncode: int | None = ...,
        streams: tuple[str, str] | None = ...,
        problem: ProblemDetailsDict | None = ...,
    ) -> None: ...

def run_tool(
    command: Sequence[str],
    *,
    cwd: Path | None = ...,
    env: Mapping[str, str] | None = ...,
    timeout: float | None = ...,
    check: bool = ...,
) -> ToolRunResult: ...
