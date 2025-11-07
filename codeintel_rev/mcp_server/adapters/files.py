"""File and scope management adapter.

Provides file listing, reading, and scope configuration.
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Sequence
from pathlib import Path

from codeintel_rev.config.settings import load_settings
from codeintel_rev.io.path_utils import (
    PathOutsideRepositoryError,
    resolve_within_repo,
)
from codeintel_rev.mcp_server.schemas import ScopeIn


def set_scope(scope: ScopeIn) -> dict:
    """Set query scope for subsequent operations.

    Parameters
    ----------
    scope : ScopeIn
        Scope configuration with repos, branches, paths, languages.

    Returns
    -------
    dict
        Effective scope configuration.

    Examples
    --------
    >>> result = set_scope({"repos": ["myrepo"], "languages": ["python"]})
    >>> result["status"]
    'ok'
    """
    return {
        "effective_scope": scope,
        "status": "ok",
    }


def list_paths(
    path: str | None = None,
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
    max_results: int = 1000,
) -> dict:
    """List files in repository.

    Parameters
    ----------
    path : str | None
        Starting path relative to repo root (None = root).
    include_globs : list[str] | None
        Glob patterns to include (e.g., ["*.py"]).
    exclude_globs : list[str] | None
        Glob patterns to exclude (e.g., ["__pycache__", "*.pyc"]).
    max_results : int
        Maximum number of files to return.

    Returns
    -------
    dict
        File listing with paths and metadata.

    Examples
    --------
    >>> result = list_paths(path="src", include_globs=["*.py"])
    >>> isinstance(result["items"], list)
    True
    Notes
    -----
    The traversal skips directories that match the default or user supplied
    exclusion globs (for example ``**/.git/**``) so that large dependency
    folders are pruned without visiting their contents.
    """
    settings = load_settings()
    repo_root = Path(settings.paths.repo_root).expanduser().resolve()
    search_root, error = _resolve_search_root(repo_root, path)
    if search_root is None:
        return {"items": [], "total": 0, "error": error or "Path not found or not a directory"}

    # Default excludes
    default_excludes = [
        "**/.git",
        "**/.git/**",
        "**/.venv",
        "**/.venv/**",
        "**/__pycache__",
        "**/__pycache__/**",
        "**/node_modules",
        "**/node_modules/**",
        "**/.pytest_cache",
        "**/.pytest_cache/**",
        "**/.ruff_cache",
        "**/.ruff_cache/**",
        "**/.mypy_cache",
        "**/.mypy_cache/**",
        "**/*.pyc",
        "**/*.pyo",
    ]
    excludes = (exclude_globs or []) + default_excludes
    includes = include_globs or ["**"]

    # Walk directory
    items = []
    for current_root, dirnames, filenames in os.walk(search_root):
        try:
            resolved_root = Path(current_root).resolve()
        except OSError:
            continue

        if _relative_path_str(resolved_root, repo_root) is None:
            continue

        # Prune excluded directories so we do not descend into them.
        for dir_name in list(dirnames):
            dir_path = resolved_root / dir_name
            relative_dir = _relative_path_str(dir_path, repo_root)
            if relative_dir is None:
                dirnames.remove(dir_name)
                continue
            dir_targets = (
                relative_dir,
                f"{relative_dir}/",
                f"./{relative_dir}",
                f"./{relative_dir}/",
            )
            if any(_matches_any(target, excludes) for target in dir_targets):
                dirnames.remove(dir_name)

        for file_name in filenames:
            file_path = resolved_root / file_name
            relative_file = _relative_path_str(file_path, repo_root)
            if relative_file is None:
                continue
            if _matches_any(relative_file, excludes) or _matches_any(
                f"./{relative_file}", excludes
            ):
                continue
            if not _matches_any(relative_file, includes):
                continue

            stat_result = _safe_stat(file_path)
            if stat_result is None:
                continue

            items.append(
                {
                    "path": relative_file,
                    "size": stat_result.st_size,
                    "modified": stat_result.st_mtime,
                }
            )

            if len(items) >= max_results:
                return {
                    "items": items,
                    "total": len(items),
                    "truncated": True,
                }

    return {
        "items": items,
        "total": len(items),
        "truncated": len(items) >= max_results,
    }


def open_file(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict:
    """Read file content with optional line slicing.

    Parameters
    ----------
    path : str
        File path relative to repo root.
    start_line : int | None
        Start line (1-indexed, inclusive). Must be positive when provided.
    end_line : int | None
        End line (1-indexed, inclusive). Must be positive when provided.

    Returns
    -------
    dict
        File content and metadata. When the requested line bounds are invalid,
        an ``{"error": ...}`` payload describing the constraint violation is
        returned instead of file content.

    Notes
    -----
    ``start_line`` and ``end_line`` are inclusive and 1-indexed. If both bounds
    are supplied then ``start_line`` must be less than or equal to ``end_line``.
    Providing a single bound slices from the start or through the end of the
    file respectively.

    Examples
    --------
    >>> result = open_file("README.md", start_line=1, end_line=10)
    >>> "content" in result
    True
    """
    settings = load_settings()
    repo_root = Path(settings.paths.repo_root).expanduser().resolve()
    try:
        file_path = resolve_within_repo(
            repo_root,
            path,
            allow_nonexistent=False,
        )
    except PathOutsideRepositoryError as exc:
        return {"error": str(exc), "path": path}
    except FileNotFoundError:
        return {"error": "File not found", "path": path}

    if not file_path.is_file():
        return {"error": "Not a file", "path": path}

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {"error": "Binary file or encoding error", "path": path}

    # Validate and slice lines if requested
    if start_line is not None and start_line <= 0:
        return {
            "error": "start_line must be a positive integer",
            "path": path,
        }
    if end_line is not None and end_line <= 0:
        return {
            "error": "end_line must be a positive integer",
            "path": path,
        }
    if (
        start_line is not None
        and end_line is not None
        and start_line > end_line
    ):
        return {
            "error": "start_line must be less than or equal to end_line",
            "path": path,
        }

    if start_line is not None or end_line is not None:
        lines = content.splitlines(keepends=True)
        start_idx = (start_line - 1) if start_line is not None else 0
        end_idx = end_line if end_line is not None else len(lines)
        content = "".join(lines[start_idx:end_idx])

    return {
        "path": path,
        "content": content,
        "lines": len(content.splitlines()),
        "size": len(content),
    }


__all__ = ["list_paths", "open_file", "set_scope"]


def _resolve_search_root(repo_root: Path, requested: str | None) -> tuple[Path | None, str | None]:
    if requested is None:
        return repo_root, None
    try:
        search_root = resolve_within_repo(
            repo_root,
            requested,
            allow_nonexistent=False,
        )
    except PathOutsideRepositoryError as exc:
        return None, str(exc)
    except FileNotFoundError:
        return None, "Path not found or not a directory"
    if not search_root.is_dir():
        return None, "Path not found or not a directory"
    return search_root, None


def _matches_any(target: str, patterns: Sequence[str]) -> bool:
    normalized = target.replace("\\", "/")
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in patterns)


def _relative_path_str(path: Path, repo_root: Path) -> str | None:
    try:
        resolved = path.resolve()
    except OSError:
        return None
    try:
        relative_path = resolved.relative_to(repo_root)
    except ValueError:
        return None
    relative_str = relative_path.as_posix()
    return relative_str


def _safe_stat(path: Path):
    try:
        return path.stat()
    except OSError:
        return None
