from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    problem: Any

def run_tool(
    command: Sequence[str],
    *,
    cwd: Path | None = ...,
    env: Mapping[str, str] | None = ...,
    timeout: float | None = ...,
    check: bool = ...,
) -> ToolRunResult: ...
