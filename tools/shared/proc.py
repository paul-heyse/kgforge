"""Public wrapper for :mod:`tools._shared.proc`."""

from __future__ import annotations

from tools._shared.proc import (
    ProcessRunner,
    ToolExecutionError,
    ToolRunResult,
    get_process_runner,
    run_tool,
    set_process_runner,
)

__all__: tuple[str, ...] = (
    "ProcessRunner",
    "ToolExecutionError",
    "ToolRunResult",
    "get_process_runner",
    "run_tool",
    "set_process_runner",
)
