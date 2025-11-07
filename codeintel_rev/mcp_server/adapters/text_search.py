"""Text search adapter using ripgrep.

Fast text search with regex support.
"""

from __future__ import annotations

import subprocess
import json
from pathlib import Path

from codeintel_rev.config.settings import load_settings
from codeintel_rev.mcp_server.schemas import Match


def search_text(
    query: str,
    regex: bool = False,
    case_sensitive: bool = False,
    paths: list[str] | None = None,
    max_results: int = 50,
) -> dict:
    """Fast text search using ripgrep.

    Parameters
    ----------
    query : str
        Search query (literal or regex).
    regex : bool
        Treat query as regex pattern.
    case_sensitive : bool
        Case-sensitive search.
    paths : list[str] | None
        Paths to search in (relative to repo root).
    max_results : int
        Maximum number of results.

    Returns
    -------
    dict
        Search matches with locations and previews.

    Examples
    --------
    >>> result = search_text("def main", regex=False)
    >>> isinstance(result["matches"], list)
    True
    """
    settings = load_settings()
    repo_root = Path(settings.paths.repo_root)

    # Build ripgrep command
    cmd = ["rg", "--json", "--max-count", str(max_results)]

    if not case_sensitive:
        cmd.append("--ignore-case")

    if not regex:
        cmd.append("--fixed-strings")

    # Add query
    cmd.append(query)

    # Add paths
    if paths:
        for p in paths:
            cmd.append(str(repo_root / p))
    else:
        cmd.append(str(repo_root))

    # Run ripgrep
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "matches": [],
            "total": 0,
            "error": "Search timeout",
        }
    except FileNotFoundError:
        # Fallback to simple grep if ripgrep not installed
        return _fallback_grep(repo_root, query, case_sensitive, max_results)

    # Parse JSON output
    matches: list[Match] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        try:
            entry = json.loads(line)

            if entry.get("type") != "match":
                continue

            data = entry.get("data", {})
            path = data.get("path", {}).get("text", "")
            line_number = data.get("line_number", 0)
            lines = data.get("lines", {}).get("text", "")
            submatches = data.get("submatches", [])

            # Get first submatch for column
            column = 0
            if submatches:
                column = submatches[0].get("start", 0)

            match: Match = {
                "path": str(Path(path).relative_to(repo_root)),
                "line": line_number,
                "column": column,
                "preview": lines.strip()[:200],
            }
            matches.append(match)

        except (json.JSONDecodeError, KeyError, ValueError):
            continue

        if len(matches) >= max_results:
            break

    return {
        "matches": matches,
        "total": len(matches),
        "truncated": len(matches) >= max_results,
    }


def _fallback_grep(
    repo_root: Path,
    query: str,
    case_sensitive: bool,
    max_results: int,
) -> dict:
    """Fallback to basic grep if ripgrep unavailable.

    Parameters
    ----------
    repo_root : Path
        Repository root directory.
    query : str
        Search query.
    case_sensitive : bool
        Case-sensitive search.
    max_results : int
        Maximum results.

    Returns
    -------
    dict
        Search matches.
    """
    cmd = ["grep", "-r", "-n"]

    if not case_sensitive:
        cmd.append("-i")

    cmd.extend([query, str(repo_root)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {
            "matches": [],
            "total": 0,
            "error": "Search tool unavailable",
        }

    # Parse grep output
    matches: list[Match] = []
    for line in result.stdout.splitlines()[:max_results]:
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue

        try:
            path = str(Path(parts[0]).relative_to(repo_root))
            line_number = int(parts[1])
            preview = parts[2].strip()[:200]

            match: Match = {
                "path": path,
                "line": line_number,
                "column": 0,
                "preview": preview,
            }
            matches.append(match)
        except (ValueError, IndexError):
            continue

    return {
        "matches": matches,
        "total": len(matches),
        "truncated": len(matches) >= max_results,
    }


__all__ = ["search_text"]
