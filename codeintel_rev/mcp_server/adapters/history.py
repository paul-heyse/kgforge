"""Git history adapter for blame and log operations.

Provides git blame and commit history via subprocess.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from codeintel_rev.config.settings import load_settings
from codeintel_rev.mcp_server.schemas import GitBlameEntry


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
    repo_root = Path(settings.paths.repo_root)
    file_path = repo_root / path

    if not file_path.exists():
        return {"blame": [], "error": "File not found"}

    # Run git blame with porcelain format
    cmd = [
        "git",
        "blame",
        "--porcelain",
        f"-L{start_line},{end_line}",
        str(file_path.relative_to(repo_root)),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30.0,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        return {"blame": [], "error": str(e)}

    # Parse porcelain format
    blame_entries: list[GitBlameEntry] = []
    current_commit = ""
    current_author = ""
    current_date = ""
    current_message = ""
    current_line = start_line

    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        # Commit SHA line
        if len(line) > 40 and " " in line and not line.startswith("\t"):
            parts = line.split()
            if len(parts) >= 3:
                current_commit = parts[0]
                current_line = int(parts[2])
                continue

        # Author line
        if line.startswith("author "):
            current_author = line[7:]
            continue

        # Date line (timestamp)
        if line.startswith("author-time "):
            timestamp = int(line[12:])
            from datetime import UTC, datetime

            current_date = datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
            continue

        # Summary line
        if line.startswith("summary "):
            current_message = line[8:]
            continue

        # Content line (starts with tab)
        if line.startswith("\t") and current_commit:
            entry: GitBlameEntry = {
                "line": current_line,
                "commit": current_commit[:8],  # Short SHA
                "author": current_author,
                "date": current_date,
                "message": current_message,
            }
            blame_entries.append(entry)

            # Reset for next entry
            current_commit = ""
            current_author = ""
            current_date = ""
            current_message = ""

    return {"blame": blame_entries}


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
    repo_root = Path(settings.paths.repo_root)
    file_path = repo_root / path

    if not file_path.exists():
        return {"commits": [], "error": "File not found"}

    # Run git log
    cmd = [
        "git",
        "log",
        f"-n{limit}",
        "--format=%H|%an|%ae|%at|%s",
        "--",
        str(file_path.relative_to(repo_root)),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30.0,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        return {"commits": [], "error": str(e)}

    # Parse output
    commits = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        parts = line.split("|", 4)
        if len(parts) < 5:
            continue

        sha, author_name, author_email, timestamp, message = parts

        try:
            from datetime import UTC, datetime

            date = datetime.fromtimestamp(int(timestamp), tz=UTC).isoformat()
        except ValueError:
            date = timestamp

        commits.append(
            {
                "sha": sha[:8],
                "full_sha": sha,
                "author": author_name,
                "email": author_email,
                "date": date,
                "message": message,
            }
        )

    return {"commits": commits}


__all__ = ["blame_range", "file_history"]
