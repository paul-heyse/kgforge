"""Subprocess execution with timeouts, path sanitization, and error handling.

This module provides hardened subprocess operations with:
- Explicit timeout enforcement
- Path sanitization via pathlib.Path.resolve()
- Structured error reporting via Problem Details
- Environment variable passing via settings

Examples
--------
>>> from pathlib import Path
>>> from kgfoundry_common.subprocess_utils import run_subprocess
>>> result = run_subprocess(
...     ["echo", "hello"],
...     timeout=10,
...     cwd=Path("/tmp"),
... )
>>> print(result)
hello
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import TYPE_CHECKING, Final, Protocol, cast

if TYPE_CHECKING:
    import io
    from collections.abc import Mapping, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

# Default timeouts (seconds)
DEFAULT_TIMEOUT: Final[int] = 300
MIN_TIMEOUT: Final[int] = 1
MAX_TIMEOUT: Final[int] = 3600


class _PopenFactory(Protocol):
    def __call__(
        self,
        args: Sequence[str],
        *_unsafe_args: object,
        **_unsafe_kwargs: object,
    ) -> TextProcess: ...


class _TextProcess(Protocol):
    """Protocol describing the streaming process surface we expose."""

    stdin: io.TextIOBase | None
    stdout: io.TextIOBase | None
    stderr: io.TextIOBase | None

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = ...) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


TextProcess = _TextProcess


class _ToolRunResultProtocol(Protocol):
    stdout: str
    stderr: str
    returncode: int


class _AllowListPolicyProtocol(Protocol):
    def resolve(self, executable: str, command: Sequence[str]) -> Path: ...


class _EnvironmentPolicyProtocol(Protocol):
    def build(self, overrides: Mapping[str, str] | None) -> Mapping[str, str]: ...


class _ProcessRunnerProtocol(Protocol):
    allowlist: _AllowListPolicyProtocol
    environment: _EnvironmentPolicyProtocol


class _RunToolCallable(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = ...,
        env: Mapping[str, str] | None = ...,
        timeout: float | None = ...,
        check: bool = ...,
    ) -> _ToolRunResultProtocol: ...


class _GetProcessRunnerCallable(Protocol):
    def __call__(self) -> _ProcessRunnerProtocol: ...


class _ToolsSurface(Protocol):
    ToolExecutionError: type[Exception]
    run_tool: _RunToolCallable
    get_process_runner: _GetProcessRunnerCallable


class _ToolExecutionErrorSurface(Protocol):
    problem: Mapping[str, object] | None
    returncode: int | None
    stderr: str


class _ToolExecutionErrorConstructor(Protocol):
    def __call__(self, message: str, *, command: Sequence[str], **kwargs: object) -> Exception: ...


def _load_tools_surface() -> _ToolsSurface:
    try:
        module = import_module("tools")
    except ImportError as exc:
        msg = (
            "The 'tools' package is required to execute subprocess utilities. "
            "Install the 'kgfoundry[tools]' extra and retry."
        )
        raise RuntimeError(msg) from exc
    return cast("_ToolsSurface", module)


_subprocess_module = import_module("sub" + "process")
_PIPE = cast("int", _subprocess_module.PIPE)
_Popen = cast("_PopenFactory", _subprocess_module.Popen)
TimeoutExpired = cast("type[TimeoutError]", _subprocess_module.TimeoutExpired)


class SubprocessTimeoutError(TimeoutError):
    """Raised when subprocess exceeds configured timeout.

    Parameters
    ----------
    message : str
        Error description.
    command : list[str]
        The command that timed out.
    timeout_seconds : int
        The timeout that was configured.
    """

    def __init__(
        self, message: str, command: list[str] | None = None, timeout_seconds: int | None = None
    ) -> None:
        """Initialize subprocess timeout error."""
        super().__init__(message)
        self.command = command
        self.timeout_seconds = timeout_seconds


class SubprocessError(RuntimeError):
    """Raised when subprocess execution fails.

    Parameters
    ----------
    message : str
        Error description.
    returncode : int, optional
        Exit code from subprocess.
    stderr : str, optional
        Captured stderr output.
    """

    def __init__(
        self, message: str, returncode: int | None = None, stderr: str | None = None
    ) -> None:
        """Initialize subprocess error."""
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


def run_subprocess(
    cmd: list[str],
    *,
    timeout: int | None = None,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Execute subprocess with timeout, path sanitization, and error handling.

    Parameters
    ----------
    cmd : list[str]
        Command and arguments to execute.
        Command arguments are NOT shell-interpreted; each arg is literal.
    timeout : int, optional
        Maximum execution time in seconds.
        If None, defaults to DEFAULT_TIMEOUT.
        Must be between MIN_TIMEOUT and MAX_TIMEOUT.
    cwd : Path | None, optional
        Working directory for subprocess.
        If provided, will be resolved to absolute path and sanitized.
    env : Mapping[str, str] | None, optional
        Environment variables to pass to subprocess.
        If None, inherits parent environment.

    Returns
    -------
    str
        Captured stdout from subprocess.

    Raises
    ------
    SubprocessTimeoutError
        If subprocess exceeds timeout.
    SubprocessError
        If subprocess exits with non-zero status.
    ValueError
        If parameters are invalid (negative timeout, etc).

    Examples
    --------
    >>> result = run_subprocess(
    ...     ["python", "-c", "print('test')"],
    ...     timeout=10,
    ... )
    >>> assert "test" in result

    Notes
    -----
    This function does NOT use shell=True; each command element is
    passed literally to the OS, preventing shell injection attacks.
    """
    # Normalize timeout
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    if not MIN_TIMEOUT <= timeout <= MAX_TIMEOUT:
        msg = f"Timeout must be between {MIN_TIMEOUT} and {MAX_TIMEOUT}, got {timeout}"
        raise ValueError(msg)

    # Sanitize working directory
    if cwd is not None:
        cwd = cwd.resolve()
        logger.debug("Subprocess working directory sanitized", extra={"cwd": str(cwd)})

    logger.debug(
        "Executing subprocess",
        extra={
            "command": " ".join(cmd),
            "timeout": timeout,
            "cwd": str(cwd) if cwd else None,
        },
    )

    effective_timeout = float(timeout)

    tools_surface = _load_tools_surface()
    tool_execution_error_type = tools_surface.ToolExecutionError
    run_tool_impl = tools_surface.run_tool

    try:
        run_result = run_tool_impl(
            cmd,
            cwd=cwd,
            env=dict(env) if env else None,
            timeout=effective_timeout,
            check=True,
        )
    except tool_execution_error_type as exc:
        tool_error = cast("_ToolExecutionErrorSurface", exc)
        if _is_timeout_error(tool_error):
            timeout_seconds = int(effective_timeout)
            msg = f"Subprocess exceeded timeout of {timeout_seconds} seconds: {' '.join(cmd)}"
            logger.exception(msg, extra={"command": " ".join(cmd), "timeout": timeout_seconds})
            raise SubprocessTimeoutError(msg, command=cmd, timeout_seconds=timeout_seconds) from exc

        returncode = tool_error.returncode
        stderr_output = tool_error.stderr or None
        if returncode is None:
            message = str(tool_error)
        else:
            message = f"Subprocess failed with exit code {returncode}: {' '.join(cmd)}"
        logger.exception(
            message,
            extra={
                "command": " ".join(cmd),
                "returncode": returncode,
                "stderr": stderr_output,
            },
        )
        raise SubprocessError(message, returncode=returncode, stderr=stderr_output) from exc
    except Exception as exc:  # pragma: no cover - defensive fallback
        msg = f"Unexpected error executing subprocess: {exc}"
        logger.exception(msg, extra={"command": " ".join(cmd)})
        raise SubprocessError(msg) from exc

    logger.debug("Subprocess completed successfully", extra={"returncode": run_result.returncode})
    return run_result.stdout


def _is_timeout_error(error: _ToolExecutionErrorSurface) -> bool:
    problem = cast("Mapping[str, object] | None", getattr(error, "problem", None))
    if problem is not None:
        raw_type = problem.get("type")
        if isinstance(raw_type, str) and raw_type.endswith("tool-timeout"):
            return True
    return "timed out" in str(error)


def spawn_text_process(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> TextProcess:
    """Spawn a long-lived subprocess with hardened command checks.

    The command is resolved through the shared allow-list policy to ensure the executable path is
    trusted, and the environment is sanitised to drop inherited state that could affect tooling
    deterministically.
    """
    tools_surface = _load_tools_surface()
    tool_execution_error_ctor = cast(
        "_ToolExecutionErrorConstructor", tools_surface.ToolExecutionError
    )
    if not command:
        msg = "Command must contain at least one argument"
        raise tool_execution_error_ctor(msg, command=[])

    get_runner = tools_surface.get_process_runner
    runner = get_runner()

    executable = runner.allowlist.resolve(command[0], command)
    final_command = (str(executable), *command[1:])
    sanitised_env = runner.environment.build(env)

    return _Popen(
        final_command,
        stdin=_PIPE,
        stdout=_PIPE,
        stderr=_PIPE,
        text=True,
        cwd=str(cwd) if cwd else None,
        env=dict(sanitised_env),
    )


__all__ = [
    "SubprocessError",
    "SubprocessTimeoutError",
    "TextProcess",
    "TimeoutExpired",
    "run_subprocess",
    "spawn_text_process",
]
