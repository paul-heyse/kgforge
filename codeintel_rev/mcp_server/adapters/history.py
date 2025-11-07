"""Git history adapter for blame and log operations.

Provides git blame and commit history via subprocess.
"""

from __future__ import annotations

import string
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from codeintel_rev.config.settings import load_settings
from codeintel_rev.io.path_utils import (
    PathOutsideRepositoryError,
    resolve_within_repo,
)
from codeintel_rev.mcp_server.schemas import GitBlameEntry
from kgfoundry_common.subprocess_utils import (
    SubprocessError,
    SubprocessTimeoutError,
    run_subprocess,
)

BLAME_TIMEOUT_SECONDS = 30
LOG_TIMEOUT_SECONDS = 30
SHORT_SHA_LENGTH = 8
FULL_SHA_LENGTH = 40
LOG_LINE_PARTS = 5
BLAME_HEADER_FIELDS = 3


def blame_range(
    path: str,
    start_line: int,
    end_line: int,
) -> dict:
    """Get git blame for line range.

    Parameters
    ----------
    path : str
        File path relative to repo root.
    start_line : int
        Start line (1-indexed).
    end_line : int
        End line (1-indexed).

    Returns
    -------
    dict
        Blame entries for each line.

    Examples
    --------
    >>> result = blame_range("README.md", 1, 10)
    >>> isinstance(result["blame"], list)
    True
    """
    settings = load_settings()
    repo_root = Path(settings.paths.repo_root).expanduser().resolve()
    try:
        file_path = resolve_within_repo(repo_root, path, allow_nonexistent=False)
    except PathOutsideRepositoryError as exc:
        return {"blame": [], "error": str(exc)}
    except FileNotFoundError:
        return {"blame": [], "error": "File not found"}

    stdout, error = _invoke_git(
        [
            "git",
            "blame",
            "--line-porcelain",
            f"-L{start_line},{end_line}",
            str(file_path.relative_to(repo_root)),
        ],
        repo_root=repo_root,
        timeout=BLAME_TIMEOUT_SECONDS,
    )
    if error is not None or stdout is None:
        return {"blame": [], "error": error}

    return {"blame": _parse_blame_porcelain(stdout)}


def file_history(
    path: str,
    limit: int = 50,
) -> dict:
    """Get commit history for file.

    Parameters
    ----------
    path : str
        File path relative to repo root.
    limit : int
        Maximum number of commits.

    Returns
    -------
    dict
        Commit history entries.

    Examples
    --------
    >>> result = file_history("README.md", limit=10)
    >>> isinstance(result["commits"], list)
    True
    """
    settings = load_settings()
    repo_root = Path(settings.paths.repo_root).expanduser().resolve()
    try:
        file_path = resolve_within_repo(repo_root, path, allow_nonexistent=False)
    except PathOutsideRepositoryError as exc:
        return {"commits": [], "error": str(exc)}
    except FileNotFoundError:
        return {"commits": [], "error": "File not found"}

    stdout, error = _invoke_git(
        [
            "git",
            "log",
            f"-n{limit}",
            "--format=%H|%an|%ae|%at|%s",
            "--",
            str(file_path.relative_to(repo_root)),
        ],
        repo_root=repo_root,
        timeout=LOG_TIMEOUT_SECONDS,
    )
    if error is not None or stdout is None:
        return {"commits": [], "error": error}

    # Parse output
    commits = []
    for line in stdout.splitlines():
        if not line.strip():
            continue

        parts = line.split("|", LOG_LINE_PARTS - 1)
        if len(parts) < LOG_LINE_PARTS:
            continue

        sha, author_name, author_email, timestamp_str, message = parts

        date = _to_iso_timestamp(timestamp_str)

        commits.append(
            {
                "sha": sha[:SHORT_SHA_LENGTH],
                "full_sha": sha,
                "author": author_name,
                "email": author_email,
                "date": date,
                "message": message,
            }
        )

    return {"commits": commits}


def _invoke_git(
    command: list[str],
    *,
    repo_root: Path,
    timeout: int,
) -> tuple[str | None, str | None]:
    """Execute a git command and capture stdout or return an error.

    Parameters
    ----------
    command : list[str]
        Git command and arguments.
    repo_root : Path
        Repository root directory.
    timeout : int
        Command timeout in seconds.

    Returns
    -------
    tuple[str | None, str | None]
        Pair of stdout and error message. Exactly one element will be ``None``.
    """
    try:
        stdout = run_subprocess(command, cwd=repo_root, timeout=timeout)
    except (SubprocessError, SubprocessTimeoutError, ValueError) as exc:
        return None, str(exc)
    else:
        return stdout, None


def _parse_blame_porcelain(stdout: str) -> list[GitBlameEntry]:
    """Parse porcelain blame output into structured entries.

    Parameters
    ----------
    stdout : str
        Git blame porcelain format output.

    Returns
    -------
    list[GitBlameEntry]
        Parsed blame entries ordered by output sequence.
    """
    entries: list[GitBlameEntry] = []
    block: list[str] = []

    for line in stdout.splitlines():
        if _is_blame_header(line):
            if block:
                entry = _parse_blame_block(block)
                if entry is not None:
                    entries.append(entry)
                block = []

        if line.strip():
            block.append(line)

    if block:
        entry = _parse_blame_block(block)
        if entry is not None:
            entries.append(entry)

    return entries


def _parse_blame_block(lines: Sequence[str]) -> GitBlameEntry | None:
    """Parse a single line-porcelain blame block.

    Parameters
    ----------
    lines : Sequence[str]
        Lines comprising a single blame block.

    Returns
    -------
    GitBlameEntry | None
        Parsed blame entry when successful, otherwise ``None``.
    """
    if not lines:
        return None

    header_parts = lines[0].split()
    if len(header_parts) < BLAME_HEADER_FIELDS:
        return None

    commit = header_parts[0]
    try:
        line_number = int(header_parts[2])
    except ValueError:
        return None

    author = ""
    timestamp = ""
    summary = ""
    for line in lines[1:]:
        if line.startswith("author "):
            author = line[len("author ") :]
        elif line.startswith("author-time "):
            timestamp = line[len("author-time ") :]
        elif line.startswith("summary "):
            summary = line[len("summary ") :]

    return {
        "line": line_number,
        "commit": commit[:SHORT_SHA_LENGTH],
        "author": author,
        "date": _to_iso_timestamp(timestamp),
        "message": summary,
    }


def _is_blame_header(line: str) -> bool:
    """Return ``True`` when the given line starts a new blame entry."""

    if not line:
        return False

    sha, *_ = line.split(maxsplit=1)
    if len(sha) != FULL_SHA_LENGTH:
        return False

    return all(char in string.hexdigits for char in sha)


def _to_iso_timestamp(timestamp: str) -> str:
    """Convert a unix timestamp string to ISO 8601 format.

    Parameters
    ----------
    timestamp : str
        Unix timestamp string.

    Returns
    -------
    str
        ISO 8601 timestamp or the original string if parsing fails.
    """
    try:
        return datetime.fromtimestamp(int(timestamp), tz=UTC).isoformat()
    except ValueError:
        return timestamp


__all__ = ["blame_range", "file_history"]
