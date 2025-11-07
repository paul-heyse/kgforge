"""FastMCP server with QueryScope tools.

Implements full MCP tool catalog for code intelligence.
"""

from __future__ import annotations

from fastmcp import FastMCP

from codeintel_rev.mcp_server.schemas import AnswerEnvelope, ScopeIn

# Create FastMCP instance
mcp = FastMCP("CodeIntel MCP")


# ==================== Scope & Navigation ====================


@mcp.tool()
def set_scope(scope: ScopeIn) -> dict:
    """Set query scope for subsequent operations.

    Parameters
    ----------
    scope : ScopeIn
        Scope parameters (repos, branches, paths, languages).

    Returns
    -------
    dict
        Effective scope configuration.
    """
    from codeintel_rev.mcp_server.adapters.files import set_scope as _set_scope

    return _set_scope(scope)


@mcp.tool()
def list_paths(
    path: str | None = None,
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
    max_results: int = 1000,
) -> dict:
    """List files in scope.

    Parameters
    ----------
    path : str | None
        Starting path (defaults to repo root).
    include_globs : list[str] | None
        Glob patterns to include.
    exclude_globs : list[str] | None
        Glob patterns to exclude.
    max_results : int
        Maximum results to return.

    Returns
    -------
    dict
        File listing with paths.
    """
    from codeintel_rev.mcp_server.adapters.files import list_paths as _list_paths

    return _list_paths(path, include_globs, exclude_globs, max_results)


@mcp.tool()
def open_file(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict:
    """Read file content.

    Parameters
    ----------
    path : str
        File path.
    start_line : int | None
        Start line (1-indexed, inclusive).
    end_line : int | None
        End line (1-indexed, inclusive).

    Returns
    -------
    dict
        File content and metadata.
    """
    from codeintel_rev.mcp_server.adapters.files import open_file as _open_file

    return _open_file(path, start_line, end_line)


# ==================== Search ====================


@mcp.tool()
def search_text(
    query: str,
    regex: bool = False,
    case_sensitive: bool = False,
    paths: list[str] | None = None,
    max_results: int = 50,
) -> dict:
    """Fast text search (ripgrep-like).

    Parameters
    ----------
    query : str
        Search query.
    regex : bool
        Treat query as regex.
    case_sensitive : bool
        Case-sensitive search.
    paths : list[str] | None
        Paths to search in.
    max_results : int
        Maximum results.

    Returns
    -------
    dict
        Search matches.
    """
    from codeintel_rev.mcp_server.adapters.text_search import search_text as _search_text

    return _search_text(query, regex, case_sensitive, paths, max_results)


@mcp.tool()
async def semantic_search(
    query: str,
    limit: int = 20,
) -> AnswerEnvelope:
    """Semantic code search using embeddings.

    Parameters
    ----------
    query : str
        Natural language or code query.
    limit : int
        Maximum results.

    Returns
    -------
    AnswerEnvelope
        Search results with findings.
    """
    from codeintel_rev.mcp_server.adapters.semantic import semantic_search as _semantic_search

    return await _semantic_search(query, limit)


# ==================== Symbols ====================


@mcp.tool()
def symbol_search(
    query: str,
    kind: str | None = None,
    language: str | None = None,
) -> dict:
    """Search for symbols (functions, classes, etc).

    Parameters
    ----------
    query : str
        Symbol name query.
    kind : str | None
        Symbol kind filter (function, class, variable).
    language : str | None
        Language filter.

    Returns
    -------
    dict
        Symbol matches.
    """
    # TODO: Implement with SCIP/pyrefly
    return {"symbols": [], "total": 0}


@mcp.tool()
def definition_at(
    path: str,
    line: int,
    character: int,
) -> dict:
    """Find definition at position.

    Parameters
    ----------
    path : str
        File path.
    line : int
        Line number (1-indexed).
    character : int
        Character offset (0-indexed).

    Returns
    -------
    dict
        Definition locations.
    """
    # TODO: Implement with pyrefly/SCIP
    return {"locations": []}


@mcp.tool()
def references_at(
    path: str,
    line: int,
    character: int,
) -> dict:
    """Find references at position.

    Parameters
    ----------
    path : str
        File path.
    line : int
        Line number (1-indexed).
    character : int
        Character offset (0-indexed).

    Returns
    -------
    dict
        Reference locations.
    """
    # TODO: Implement with pyrefly/SCIP
    return {"locations": []}


# ==================== Git History ====================


@mcp.tool()
def blame_range(
    path: str,
    start_line: int,
    end_line: int,
) -> dict:
    """Git blame for line range.

    Parameters
    ----------
    path : str
        File path.
    start_line : int
        Start line (1-indexed).
    end_line : int
        End line (1-indexed).

    Returns
    -------
    dict
        Blame entries for each line.
    """
    from codeintel_rev.mcp_server.adapters.history import blame_range as _blame_range

    return _blame_range(path, start_line, end_line)


@mcp.tool()
def file_history(
    path: str,
    limit: int = 50,
) -> dict:
    """Get file commit history.

    Parameters
    ----------
    path : str
        File path.
    limit : int
        Maximum commits.

    Returns
    -------
    dict
        Commit history.
    """
    from codeintel_rev.mcp_server.adapters.history import file_history as _file_history

    return _file_history(path, limit)


# ==================== Resources ====================


@mcp.resource("file://{path}")
def file_resource(path: str) -> str:
    """Serve file content as resource.

    Parameters
    ----------
    path : str
        File path.

    Returns
    -------
    str
        File content.
    """
    # TODO: Implement file reading
    return ""


# ==================== Prompts ====================


@mcp.prompt()
def prompt_code_review(area: str) -> str:
    """Code review prompt template.

    Parameters
    ----------
    area : str
        Code area to review.

    Returns
    -------
    str
        Prompt template.
    """
    return f"Review the code in {area}. Focus on correctness, performance, and style."


# ==================== ASGI App ====================

# Export ASGI app for mounting in FastAPI
# FastMCP 2.3.2+ uses http_app() method
asgi_app = mcp.http_app()

__all__ = ["asgi_app", "mcp"]
