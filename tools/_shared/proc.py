"""Secure subprocess helpers for repository tooling."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tools._shared.problem_details import ProblemDetailsDict, build_problem_details


@dataclass(slots=True)
class ToolRunResult:
    """Structured result from invoking a subprocess."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool


class ToolExecutionError(RuntimeError):
    """Raised when a subprocess fails to execute successfully.

    This exception includes Problem Details for structured error handling
    and preserves stdout/stderr for debugging.

    Parameters
    ----------
    message : str
        Human-readable error message.
    command : Sequence[str]
        Command that failed.
    returncode : int | None, optional
        Process exit code if available.
    streams : tuple[str, str] | None, optional
        (stdout, stderr) tuple if available.
    problem : ProblemDetailsDict | None, optional
        RFC 9457 Problem Details payload.

    Examples
    --------
    >>> from tools._shared.proc import run_tool, ToolExecutionError
    >>> try:
    ...     run_tool(["nonexistent"], check=True)
    ... except ToolExecutionError as e:
    ...     assert e.problem is not None
    ...     assert e.problem["type"].startswith("https://kgfoundry.dev/problems/")
    """

    def __init__(
        self,
        message: str,
        *,
        command: Sequence[str],
        returncode: int | None = None,
        streams: tuple[str, str] | None = None,
        problem: ProblemDetailsDict | None = None,
    ) -> None:
        super().__init__(message)
        self.command: tuple[str, ...] = tuple(command)
        self.returncode = returncode
        self.stdout, self.stderr = streams if streams is not None else ("", "")
        self.problem = problem


def _resolve_executable(executable: str) -> Path:
    candidate = Path(executable)
    if candidate.is_absolute():
        return candidate
    resolved = shutil.which(executable)
    if resolved is None:
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/tool-missing",
            title="Executable not found",
            status=500,
            detail=f"Executable '{executable}' could not be resolved to an absolute path",
            instance=f"urn:tool:{executable}:missing",
        )
        message = f"Executable '{executable}' could not be resolved to an absolute path"
        raise ToolExecutionError(message, command=[executable], problem=problem)
    return Path(resolved)


def _sanitise_env(env: Mapping[str, str] | None) -> dict[str, str]:
    allowed_keys = {
        "HOME",
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "PYTHONPATH",
        "PYTHONHASHSEED",
        "TZ",
    }
    baseline: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in allowed_keys or key.startswith(("GIT_", "UV_", "CI")):
            baseline[key] = value
    if env:
        for key, value in env.items():
            baseline[key] = value
    return {key: str(value) for key, value in baseline.items()}


def run_tool(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    check: bool = False,
) -> ToolRunResult:
    """Execute ``command`` and return a :class:`ToolRunResult`."""
    if not command:
        message = "Command must contain at least one argument"
        raise ToolExecutionError(message, command=[])

    executable = _resolve_executable(command[0])
    final_command = (str(executable), *command[1:])
    sanitised_env = _sanitise_env(env)
    start = time.monotonic()
    try:
        completed = subprocess.run(  # noqa: S603
            final_command,
            cwd=str(cwd) if cwd else None,
            env=sanitised_env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/tool-timeout",
            title="Tool execution timed out",
            status=504,
            detail=(
                f"Command '{command[0]}' timed out after {timeout} seconds"
                if timeout is not None
                else f"Command '{command[0]}' timed out"
            ),
            instance=f"urn:tool:{command[0]}:timeout",
            extensions={
                "command": list(command),
                "timeout": timeout,
            },
        )
        message = "Subprocess timed out"
        stdout_text = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr_text = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        raise ToolExecutionError(
            message,
            command=command,
            returncode=None,
            streams=(stdout_text, stderr_text),
            problem=problem,
        ) from exc
    except FileNotFoundError as exc:
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/tool-missing",
            title="Executable not found",
            status=500,
            detail=str(exc),
            instance=f"urn:tool:{command[0]}:missing",
        )
        message = "Executable not found"
        raise ToolExecutionError(
            message,
            command=command,
            returncode=None,
            problem=problem,
        ) from exc

    duration = time.monotonic() - start
    result = ToolRunResult(
        command=tuple(final_command),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_seconds=duration,
        timed_out=timed_out,
    )
    if check and completed.returncode != 0:
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/tool-failure",
            title="Tool returned a non-zero exit code",
            status=500,
            detail=completed.stderr.strip() or "Unknown failure",
            instance=f"urn:tool:{command[0]}:exit-{completed.returncode}",
            extensions={
                "command": list(command),
                "returncode": completed.returncode,
            },
        )
        message = "Subprocess returned a non-zero exit status"
        raise ToolExecutionError(
            message,
            command=command,
            returncode=completed.returncode,
            streams=(completed.stdout, completed.stderr),
            problem=problem,
        )
    return result


__all__ = [
    "ToolExecutionError",
    "ToolRunResult",
    "run_tool",
]
