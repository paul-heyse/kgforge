"""Text search adapter using ripgrep.

Fast text search with regex support.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from codeintel_rev.config.settings import load_settings
from codeintel_rev.mcp_server.schemas import Match
from kgfoundry_common.subprocess_utils import (
    SubprocessError,
    SubprocessTimeoutError,
    run_subprocess,
)

SEARCH_TIMEOUT_SECONDS = 30
MAX_PREVIEW_CHARS = 200
GREP_SPLIT_PARTS = 3
COMMAND_NOT_FOUND_RETURN_CODE = 127


def search_text(
    query: str,
    *,
    regex: bool = False,
    case_sensitive: bool = False,
    paths: Sequence[str] | None = None,
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
    repo_root = Path(settings.paths.repo_root).resolve()

    cmd = _build_ripgrep_command(
        query=query,
        regex=regex,
        case_sensitive=case_sensitive,
        paths=paths,
        max_results=max_results,
    )

    try:
        stdout = run_subprocess(cmd, cwd=repo_root, timeout=SEARCH_TIMEOUT_SECONDS)
    except SubprocessTimeoutError:
        return {
            "matches": [],
            "total": 0,
            "error": "Search timeout",
        }
    except SubprocessError as exc:
        if exc.returncode == 1:
            stdout = ""
        elif exc.returncode == COMMAND_NOT_FOUND_RETURN_CODE:
            return _fallback_grep(
                repo_root=repo_root,
                query=query,
                case_sensitive=case_sensitive,
                max_results=max_results,
            )
        else:
            error_message = (exc.stderr or "").strip() or str(exc)
            return {
                "matches": [],
                "total": 0,
                "error": error_message,
            }
    except ValueError as exc:
        return {
            "matches": [],
            "total": 0,
            "error": str(exc),
        }

    matches, truncated = _parse_ripgrep_output(stdout, repo_root, max_results)
    return {
        "matches": matches,
        "total": len(matches),
        "truncated": truncated,
    }


def _fallback_grep(
    *,
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
    command = ["grep", "-r", "-n"]

    if not case_sensitive:
        command.append("-i")

    command.extend(["--", query, "."])

    try:
        stdout = run_subprocess(command, cwd=repo_root, timeout=SEARCH_TIMEOUT_SECONDS)
    except SubprocessTimeoutError:
        return {
            "matches": [],
            "total": 0,
            "error": "Search tool unavailable",
        }
    except SubprocessError as exc:
        if exc.returncode == 1:
            stdout = ""
        else:
            error_message = (exc.stderr or "").strip() or str(exc)
            return {
                "matches": [],
                "total": 0,
                "error": error_message,
            }
    except ValueError as exc:
        return {"matches": [], "total": 0, "error": str(exc)}

    matches: list[Match] = []
    for line in stdout.splitlines()[:max_results]:
        parts = line.split(":", GREP_SPLIT_PARTS - 1)
        if len(parts) < GREP_SPLIT_PARTS:
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


def _build_ripgrep_command(
    *,
    query: str,
    regex: bool,
    case_sensitive: bool,
    paths: Sequence[str] | None,
    max_results: int,
) -> list[str]:
    """Assemble the ripgrep command arguments.

    Returns
    -------
    list[str]
        Argument vector for ripgrep invocation.
    """
    command = ["rg", "--json", "--max-count", str(max_results)]

    if not case_sensitive:
        command.append("--ignore-case")

    if not regex:
        command.append("--fixed-strings")

    command.append("--")
    command.append(query)

    if paths:
        command.extend(str(Path(path)) for path in paths)
    else:
        command.append(".")

    return command


def _parse_ripgrep_output(
    stdout: str,
    repo_root: Path,
    max_results: int,
) -> tuple[list[Match], bool]:
    """Parse ripgrep JSON output into structured matches.

    Returns
    -------
    tuple[list[Match], bool]
        Parsed match list and whether results were truncated at max_results.
    """
    matches: list[Match] = []
    truncated = False

    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        try:
            entry = json.loads(stripped)
        except json.JSONDecodeError:
            continue

        if entry.get("type") != "match":
            continue

        data = entry.get("data", {})
        path_text = data.get("path", {}).get("text")
        if not path_text:
            continue

        try:
            relative_path = str(Path(path_text).resolve().relative_to(repo_root))
        except ValueError:
            relative_path = path_text

        line_number = int(data.get("line_number", 0))
        lines = data.get("lines", {}).get("text", "")
        submatches = data.get("submatches", [])
        column = int(submatches[0].get("start", 0)) if submatches else 0

        match: Match = {
            "path": relative_path,
            "line": line_number,
            "column": column,
            "preview": lines.strip()[:MAX_PREVIEW_CHARS],
        }
        matches.append(match)

        if len(matches) >= max_results:
            truncated = True
            break

    return matches, truncated


__all__ = ["search_text"]
