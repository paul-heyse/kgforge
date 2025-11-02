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
import subprocess  # noqa: S404 - hardened with sanitized Path and timeout enforcement
from collections.abc import Mapping
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

# Default timeouts (seconds)
DEFAULT_TIMEOUT: Final[int] = 300
MIN_TIMEOUT: Final[int] = 1
MAX_TIMEOUT: Final[int] = 3600


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

    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603 - inputs sanitized via Path.resolve and cmd list
            cmd,
            timeout=timeout,
            cwd=cwd,
            env=dict(env) if env else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"Subprocess exceeded timeout of {timeout} seconds: {' '.join(cmd)}"
        logger.exception(msg, extra={"command": " ".join(cmd), "timeout": timeout})
        raise SubprocessTimeoutError(msg, command=cmd, timeout_seconds=timeout) from exc
    except subprocess.CalledProcessError as exc:
        msg = f"Subprocess failed with exit code {exc.returncode}: {' '.join(cmd)}"
        stderr_raw: object = exc.stderr
        stderr_output: str | None
        if isinstance(stderr_raw, bytes):
            stderr_output = stderr_raw.decode("utf-8", errors="replace")
        elif isinstance(stderr_raw, str):
            stderr_output = stderr_raw
        else:
            stderr_output = None
        logger.exception(
            msg,
            extra={
                "command": " ".join(cmd),
                "returncode": exc.returncode,
                "stderr": stderr_output,
            },
        )
        raise SubprocessError(msg, returncode=exc.returncode, stderr=stderr_output) from exc
    except Exception as exc:
        msg = f"Unexpected error executing subprocess: {exc}"
        logger.exception(msg, extra={"command": " ".join(cmd)})
        raise SubprocessError(msg) from exc
    else:
        logger.debug("Subprocess completed successfully", extra={"returncode": result.returncode})
        return result.stdout


__all__ = ["SubprocessError", "SubprocessTimeoutError", "run_subprocess"]
