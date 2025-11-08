"""Text search adapter using ripgrep.

Fast text search with regex support.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel_rev.app.middleware import get_session_id
from codeintel_rev.mcp_server.schemas import Match
from codeintel_rev.mcp_server.scope_utils import get_effective_scope
from kgfoundry_common.errors import KgFoundryError, VectorSearchError
from kgfoundry_common.logging import get_logger, with_fields
from kgfoundry_common.observability import DurationObservation, MetricsProvider, observe_duration
from kgfoundry_common.subprocess_utils import (
    SubprocessError,
    SubprocessTimeoutError,
    run_subprocess,
)

if TYPE_CHECKING:
    from codeintel_rev.app.config_context import ApplicationContext

SEARCH_TIMEOUT_SECONDS = 30
MAX_PREVIEW_CHARS = 200
GREP_SPLIT_PARTS = 3
COMMAND_NOT_FOUND_RETURN_CODE = 127
COMPONENT_NAME = "codeintel_mcp"
LOGGER = get_logger(__name__)
METRICS = MetricsProvider.default()


def _supports_histogram_labels(histogram: object) -> bool:
    labelnames = getattr(histogram, "_labelnames", None)
    if labelnames is None:
        return True
    try:
        return len(tuple(labelnames)) > 0
    except TypeError:
        return False


_METRICS_ENABLED = _supports_histogram_labels(METRICS.operation_duration_seconds)


class _NoopObservation:
    """Fallback observation when Prometheus metrics are unavailable."""

    def mark_error(self) -> None:
        """No-op error marker."""

    def mark_success(self) -> None:
        """No-op success marker."""


@contextmanager
def _observe(operation: str) -> Iterator[DurationObservation | _NoopObservation]:
    """Yield a metrics observation, falling back to a no-op when metrics are disabled.

    Yields
    ------
    DurationObservation | _NoopObservation
        Metrics observation when Prometheus is configured, otherwise a no-op recorder.
    """
    if not _METRICS_ENABLED:
        yield _NoopObservation()
        return
    try:
        with observe_duration(METRICS, operation, component=COMPONENT_NAME) as observation:
            yield observation
            return
    except ValueError:
        yield _NoopObservation()


def search_text(  # noqa: PLR0913 - context parameter required for dependency injection
    context: ApplicationContext,
    query: str,
    *,
    regex: bool = False,
    case_sensitive: bool = False,
    paths: Sequence[str] | None = None,
    max_results: int = 50,
) -> dict:
    """Fast text search using ripgrep.

    Applies session scope path filters if set via `set_scope`. Explicit `paths`
    parameter overrides session scope.

    Parameters
    ----------
    context : ApplicationContext
        Application context containing repo root and settings.
    query : str
        Search query (literal or regex).
    regex : bool
        Treat query as regex pattern.
    case_sensitive : bool
        Case-sensitive search.
    paths : Sequence[str] | None
        Paths to search in (relative to repo root). Overrides session scope if provided.
    max_results : int
        Maximum number of results.

    Returns
    -------
    dict
        Search matches with locations and previews.

    Examples
    --------
    Basic usage:

    >>> result = search_text(context, "def main", regex=False)
    >>> isinstance(result["matches"], list)
    True

    With session scope:

    >>> set_scope(context, {"include_globs": ["src/**/*.py"]})
    >>> result = search_text(context, "def main")
    >>> # Searches only Python files in src/ directory

    Explicit paths override scope:

    >>> set_scope(context, {"include_globs": ["src/**"]})
    >>> result = search_text(context, "def main", paths=["tests/"])
    >>> # Searches tests/ directory (explicit override), not src/

    Notes
    -----
    Scope Integration:
    - Session scope is retrieved from registry using session ID (set by middleware).
    - If explicit `paths` parameter is provided, it overrides scope's `include_globs`.
    - If no explicit `paths` and scope has `include_globs`, scope paths are used.
    - If neither explicit paths nor scope paths, searches entire repository (ripgrep default).
    """
    repo_root = context.paths.repo_root

    # Retrieve session scope and merge with explicit paths parameter
    session_id = get_session_id()
    scope = get_effective_scope(context, session_id)

    # Determine effective paths: explicit paths override scope
    if paths:
        effective_paths = list(paths)  # Explicit paths override scope
    elif scope and scope.get("include_globs"):
        effective_paths = scope.get("include_globs") or None  # type: ignore[assignment]
    else:
        effective_paths = None  # Search all (ripgrep default)

    LOGGER.debug(
        "Searching text with scope filters",
        extra={
            "session_id": session_id,
            "query": query,
            "explicit_paths": list(paths) if paths else None,
            "scope_paths": scope.get("include_globs") if scope else None,  # type: ignore[typeddict-item]
            "effective_paths": effective_paths,
        },
    )

    cmd = _build_ripgrep_command(
        query=query,
        regex=regex,
        case_sensitive=case_sensitive,
        paths=effective_paths,
        max_results=max_results,
    )

    with _observe("text_search") as observation:
        try:
            stdout = run_subprocess(cmd, cwd=repo_root, timeout=SEARCH_TIMEOUT_SECONDS)
        except SubprocessTimeoutError:
            observation.mark_error()
            error = VectorSearchError(
                "Search timeout",
                context={"query": query},
            )
            return _error_response(error, operation="text_search")
        except SubprocessError as exc:
            if exc.returncode == 1:
                stdout = ""
            elif exc.returncode == COMMAND_NOT_FOUND_RETURN_CODE:
                return _fallback_grep(
                    observation=observation,
                    repo_root=repo_root,
                    query=query,
                    case_sensitive=case_sensitive,
                    max_results=max_results,
                )
            else:
                observation.mark_error()
                error_message = (exc.stderr or "").strip() or str(exc)
                error = VectorSearchError(
                    error_message,
                    cause=exc,
                    context={"query": query, "returncode": exc.returncode},
                )
                return _error_response(error, operation="text_search")
        except ValueError as exc:
            observation.mark_error()
            error = VectorSearchError(
                str(exc),
                cause=exc,
                context={"query": query},
            )
            return _error_response(error, operation="text_search")

        matches, truncated = _parse_ripgrep_output(stdout, repo_root, max_results)
        observation.mark_success()
        return {
            "matches": matches,
            "total": len(matches),
            "truncated": truncated,
        }


def _fallback_grep(
    *,
    observation: DurationObservation | _NoopObservation,
    repo_root: Path,
    query: str,
    case_sensitive: bool,
    max_results: int,
) -> dict:
    """Fallback to basic grep if ripgrep unavailable.

    Parameters
    ----------
    observation : Any
        Metrics observation context used to record success or failure.
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
        Search results or error payload when fallback execution fails.
    """
    command = ["grep", "-r", "-n"]

    if not case_sensitive:
        command.append("-i")

    command.extend(["--", query, "."])

    try:
        stdout = run_subprocess(command, cwd=repo_root, timeout=SEARCH_TIMEOUT_SECONDS)
    except SubprocessTimeoutError:
        observation.mark_error()
        error = VectorSearchError(
            "Search tool unavailable",
            context={"query": query, "tool": "grep"},
        )
        return _error_response(error, operation="text_search_fallback")
    except SubprocessError as exc:
        if exc.returncode == 1:
            stdout = ""
        else:
            observation.mark_error()
            error_message = (exc.stderr or "").strip() or str(exc)
            error = VectorSearchError(
                error_message,
                cause=exc,
                context={"query": query, "tool": "grep", "returncode": exc.returncode},
            )
            return _error_response(error, operation="text_search_fallback")
    except ValueError as exc:
        observation.mark_error()
        error = VectorSearchError(
            str(exc),
            cause=exc,
            context={"query": query, "tool": "grep"},
        )
        return _error_response(error, operation="text_search_fallback")

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

    observation.mark_success()
    return {
        "matches": matches,
        "total": len(matches),
        "truncated": len(matches) >= max_results,
    }


def _error_response(error: KgFoundryError, *, operation: str) -> dict:
    """Return standardized error payload with Problem Details.

    Parameters
    ----------
    error : KgFoundryError
        Error to convert to response.
    operation : str
        Operation name for logging.

    Returns
    -------
    dict
        Error response dictionary with matches, total, error, and problem fields.
    """
    problem = error.to_problem_details(instance=operation)
    with with_fields(
        LOGGER,
        component=COMPONENT_NAME,
        operation=operation,
        error_code=error.code.value,
    ) as adapter:
        adapter.log(error.log_level, error.message, extra={"context": error.context})

    return {
        "matches": [],
        "total": 0,
        "error": error.message,
        "problem": problem,
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

    Parameters
    ----------
    query : str
        Search query string.
    regex : bool
        Whether query is a regex pattern.
    case_sensitive : bool
        Whether search is case-sensitive.
    paths : Sequence[str] | None
        Paths to search in (relative to repo root).
    max_results : int
        Maximum number of results to return.

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

    Parameters
    ----------
    stdout : str
        Ripgrep JSON output.
    repo_root : Path
        Repository root directory.
    max_results : int
        Maximum number of results to parse.

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
