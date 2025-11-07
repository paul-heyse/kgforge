"""File and scope management adapter.

Provides file listing, reading, and scope configuration.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

from codeintel_rev.config.settings import load_settings
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
    """
    settings = load_settings()
    repo_root = Path(settings.paths.repo_root)

    # Determine search root
    if path:
        search_root = repo_root / path
        if not search_root.exists() or not search_root.is_dir():
            return {"items": [], "total": 0, "error": "Path not found or not a directory"}
    else:
        search_root = repo_root

    # Default excludes
    default_excludes = [
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
        "*.pyc",
        "*.pyo",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
    ]
    excludes = (exclude_globs or []) + default_excludes
    includes = include_globs or ["*"]

    # Walk directory
    items = []
    for file_path in search_root.rglob("*"):
        if not file_path.is_file():
            continue

        # Check excludes
        rel_path = file_path.relative_to(repo_root)
        if any(fnmatch.fnmatch(str(rel_path), pattern) for pattern in excludes):
            continue

        # Check includes
        if not any(fnmatch.fnmatch(str(rel_path), pattern) for pattern in includes):
            continue

        # Add to results
        items.append(
            {
                "path": str(rel_path),
                "size": file_path.stat().st_size,
                "modified": file_path.stat().st_mtime,
            }
        )

        if len(items) >= max_results:
            break

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
        Start line (1-indexed, inclusive).
    end_line : int | None
        End line (1-indexed, inclusive).

    Returns
    -------
    dict
        File content and metadata.

    Examples
    --------
    >>> result = open_file("README.md", start_line=1, end_line=10)
    >>> "content" in result
    True
    """
    settings = load_settings()
    repo_root = Path(settings.paths.repo_root)
    file_path = repo_root / path

    if not file_path.exists():
        return {"error": "File not found", "path": path}

    if not file_path.is_file():
        return {"error": "Not a file", "path": path}

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {"error": "Binary file or encoding error", "path": path}

    # Slice lines if requested
    if start_line is not None or end_line is not None:
        lines = content.splitlines(keepends=True)
        start_idx = (start_line - 1) if start_line else 0
        end_idx = end_line if end_line else len(lines)
        content = "".join(lines[start_idx:end_idx])

    return {
        "path": path,
        "content": content,
        "lines": len(content.splitlines()),
        "size": len(content),
    }


__all__ = ["list_paths", "open_file", "set_scope"]
