"""Compatibility facade for tooling subprocess helpers.

The heavy lifting now lives in :mod:`tools._shared.process`. This module keeps
the longstanding ``run_tool`` entry-point so existing call sites continue to
function without modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tools._shared.process import ProcessRunner, ToolExecutionError, ToolRunResult

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


@dataclass(slots=True)
class _ProcessRunnerState:
    runner: ProcessRunner


_PROCESS_STATE = _ProcessRunnerState(ProcessRunner())


def get_process_runner() -> ProcessRunner:
    """Return the global process runner instance used by :func:`run_tool`."""
    return _PROCESS_STATE.runner


def set_process_runner(runner: ProcessRunner) -> None:
    """Replace the global process runner used by :func:`run_tool`.

    Intended for advanced scenarios (e.g. tests) where dependency injection is required. Callers
    should restore the previous runner when finished.
    """
    _PROCESS_STATE.runner = runner


def run_tool(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    check: bool = False,
) -> ToolRunResult:
    """Execute ``command`` using the shared :class:`ProcessRunner` policies."""
    return _PROCESS_STATE.runner.run(
        command,
        cwd=cwd,
        env=env,
        timeout=timeout,
        check=check,
    )


__all__ = [
    "ProcessRunner",
    "ToolExecutionError",
    "ToolRunResult",
    "get_process_runner",
    "run_tool",
    "set_process_runner",
]
