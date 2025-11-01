"""Secure subprocess helpers for repository tooling."""

from __future__ import annotations

import os
import shutil
import subprocess  # noqa: S404
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tools._shared.logging import get_logger
from tools._shared.metrics import observe_tool_run
from tools._shared.problem_details import (
    ProblemDetailsDict,
    tool_disallowed_problem_details,
    tool_failure_problem_details,
    tool_missing_problem_details,
    tool_timeout_problem_details,
)
from tools._shared.settings import get_runtime_settings


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


LOGGER = get_logger(__name__)


def _resolve_executable(executable: str, command: Sequence[str]) -> Path:
    candidate = Path(executable)
    if candidate.is_absolute():
        _enforce_allowlist(candidate, command)
        return candidate
    resolved = shutil.which(executable)
    if resolved is None:
        detail = f"Executable '{executable}' could not be resolved to an absolute path"
        problem = tool_missing_problem_details(
            command=command, executable=executable, detail=detail
        )
        message = detail
        raise ToolExecutionError(message, command=[executable], problem=problem)
    resolved_path = Path(resolved)
    _enforce_allowlist(resolved_path, command)
    return resolved_path


def _enforce_allowlist(executable: Path, command: Sequence[str]) -> None:
    settings = get_runtime_settings()
    if settings.is_allowed(executable):
        return
    problem = tool_disallowed_problem_details(
        command=command,
        executable=executable,
        allowlist=settings.exec_allowlist,
    )
    message = f"Executable '{executable}' is not permitted by TOOLS_EXEC_ALLOWLIST"
    raise ToolExecutionError(message, command=command, problem=problem)


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

    executable = _resolve_executable(command[0], command)
    final_command = (str(executable), *command[1:])
    sanitised_env = _sanitise_env(env)
    with observe_tool_run(final_command, cwd=cwd, timeout=timeout) as observation:
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
        except subprocess.TimeoutExpired as exc:
            observation.failure("timeout", timed_out=True)
            problem = tool_timeout_problem_details(command=command, timeout=timeout)
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
            observation.failure("missing_executable")
            problem = tool_missing_problem_details(
                command=command, executable=command[0], detail=str(exc)
            )
            message = "Executable not found"
            raise ToolExecutionError(
                message,
                command=command,
                returncode=None,
                problem=problem,
            ) from exc

        result = ToolRunResult(
            command=tuple(final_command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=observation.duration_seconds(),
            timed_out=False,
        )

        if completed.returncode == 0:
            observation.success(completed.returncode)
        else:
            observation.failure("non_zero_exit", returncode=completed.returncode)

        if check and completed.returncode != 0:
            problem = tool_failure_problem_details(
                command=command,
                returncode=completed.returncode,
                detail=completed.stderr.strip() or "Unknown failure",
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
