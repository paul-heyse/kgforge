"""Tree-sitter query registry with lazy loading and validation.

This module provides a centralized registry for Tree-sitter query files with:
- Lazy loading and caching of query files
- Query syntax validation against language grammars
- Built-in queries for supported languages
- Extensible design for custom queries

Queries are loaded from the `codeintel/queries/` directory and validated
when first accessed. This ensures that query syntax errors are caught early
and provides clear error messages.

Example Usage
-------------
>>> from codeintel.queries import load_query
>>> query_text = load_query("python")
>>> assert "def.name" in query_text

>>> from codeintel.queries import validate_query
>>> validate_query("python", "(function_definition) @func")
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Final

from tools import get_logger

from codeintel.errors import QuerySyntaxError
from codeintel.indexer.tscore import get_language, load_langs

_logger = get_logger(__name__)

_QUERIES_DIR = Path(__file__).parent
_BUILTIN_QUERIES: Final[dict[str, Path]] = {
    "python": _QUERIES_DIR / "python.scm",
    "json": _QUERIES_DIR / "json.scm",
    "yaml": _QUERIES_DIR / "yaml.scm",
    "toml": _QUERIES_DIR / "toml.scm",
    "markdown": _QUERIES_DIR / "markdown.scm",
}


@cache
def load_query(language: str) -> str:
    """Load and return Tree-sitter query for a language.

    Queries are loaded from the built-in registry and cached. This function
    does not validate the query syntax - use :func:`validate_query` for that.

    Parameters
    ----------
    language : str
        Language identifier (e.g., 'python', 'json').

    Returns
    -------
    str
        Query text in Tree-sitter s-expression format.

    Raises
    ------
    FileNotFoundError
        If no built-in query exists for the language.

    Examples
    --------
    >>> query = load_query("python")
    >>> assert "def.name" in query

    >>> try:
    ...     load_query("nonexistent")
    ... except FileNotFoundError as e:
    ...     assert "nonexistent" in str(e)
    """
    query_path = _BUILTIN_QUERIES.get(language)
    if not query_path:
        available = ", ".join(sorted(_BUILTIN_QUERIES.keys()))
        message = f"No built-in query for language '{language}'. Available: {available}"
        _logger.warning("Query not found", extra={"language": language})
        raise FileNotFoundError(message)

    if not query_path.exists():
        message = f"Query file not found: {query_path}"
        _logger.error(
            "Query file missing", extra={"language": language, "query_path": str(query_path)}
        )
        raise FileNotFoundError(message)

    query_text = query_path.read_text(encoding="utf-8")
    _logger.debug("Loaded query", extra={"language": language, "size_bytes": len(query_text)})
    return query_text


def validate_query(language: str, query_text: str) -> None:
    """Validate that a query compiles for the given language.

    This function attempts to compile the query using the language's grammar.
    If compilation fails, a QuerySyntaxError is raised with details about
    the syntax error. If the language is not supported, LanguageNotSupportedError
    is raised by the underlying language loading functions.

    Parameters
    ----------
    language : str
        Language identifier.
    query_text : str
        Query source code in s-expression format.

    Raises
    ------
    QuerySyntaxError
        If query has syntax errors or references undefined node types.
        This exception is explicitly raised when query compilation fails.

    Examples
    --------
    >>> validate_query("python", "(function_definition) @func")

    >>> try:
    ...     validate_query("python", "(invalid_syntax @")
    ... except QuerySyntaxError as e:
    ...     assert "syntax" in str(e).lower()

    Notes
    -----
    This function loads the language grammar and compiles the query, which
    may be expensive for the first call. Subsequent calls benefit from caching.
    The function explicitly raises QuerySyntaxError when query compilation fails.
    Errors from :func:`codeintel.indexer.tscore.get_language` propagate, so
    callers should handle :class:`codeintel.errors.LanguageNotSupportedError`
    when the language identifier is unknown.
    """
    langs = load_langs()
    lang = get_language(langs, language)

    try:
        lang.query(query_text)
    except Exception as exc:
        message = f"Query syntax error for language '{language}': {exc}"
        _logger.exception("Query validation failed", extra={"language": language})
        raise QuerySyntaxError(message, extensions={"language": language}) from exc
    else:
        _logger.debug("Query validated successfully", extra={"language": language})


def list_available_queries() -> list[str]:
    """Return list of languages with built-in queries.

    Returns
    -------
    list[str]
        Sorted list of language names with available queries.

    Examples
    --------
    >>> queries = list_available_queries()
    >>> "python" in queries
    True
    >>> all(isinstance(lang, str) for lang in queries)
    True
    """
    return sorted(_BUILTIN_QUERIES.keys())


def get_query_path(language: str) -> Path | None:
    """Get the filesystem path for a language's query file.

    Parameters
    ----------
    language : str
        Language identifier.

    Returns
    -------
    Path | None
        Path to the query file, or None if not found.

    Examples
    --------
    >>> path = get_query_path("python")
    >>> assert path is not None
    >>> assert path.suffix == ".scm"

    >>> get_query_path("nonexistent") is None
    True
    """
    return _BUILTIN_QUERIES.get(language)


__all__ = [
    "get_query_path",
    "list_available_queries",
    "load_query",
    "validate_query",
]
