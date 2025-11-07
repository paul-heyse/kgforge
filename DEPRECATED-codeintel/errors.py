"""CodeIntel exception taxonomy aligned with RFC 9457 Problem Details.

This module defines a comprehensive exception hierarchy for CodeIntel operations.
All exceptions inherit from CodeIntelError and automatically map to RFC 9457
Problem Details format for consistent error handling across MCP and HTTP interfaces.

The exception taxonomy follows AGENTS.md principle 1: "Define a small exception
taxonomy, and for HTTP use RFC 9457 Problem Details".
"""

from __future__ import annotations

from typing import Any, ClassVar

from kgfoundry_common.errors import KgFoundryError


class CodeIntelError(KgFoundryError):
    """Base exception for all CodeIntel errors.

    This exception serves as the root of the CodeIntel exception hierarchy.
    All CodeIntel-specific exceptions should inherit from this class to enable
    consistent error handling and automatic Problem Details generation.

    Attributes
    ----------
    problem_type_base : ClassVar[str]
        Base URI for Problem Details type field. All CodeIntel exceptions
        use this as the prefix for their problem type URIs.
    """

    problem_type_base: ClassVar[str] = "urn:kgf:problem:codeintel"


class SandboxError(CodeIntelError):
    """Path resolution violated repository sandbox.

    Raised when a file or directory path resolves outside the repository root,
    preventing potential directory traversal attacks or access to unauthorized files.

    Attributes
    ----------
    problem_type : ClassVar[str]
        Problem type identifier for RFC 9457 Problem Details.
    default_status : ClassVar[int]
        Default HTTP status code (403 Forbidden).
    default_title : ClassVar[str]
        Default error title for Problem Details.

    Examples
    --------
    >>> raise SandboxError("Path outside repository: ../../etc/passwd")
    Traceback (most recent call last):
        ...
    codeintel.errors.SandboxError: Path outside repository: ../../etc/passwd
    """

    problem_type: ClassVar[str] = "sandbox"
    default_status: ClassVar[int] = 403
    default_title: ClassVar[str] = "Sandbox violation"


class LanguageNotSupportedError(CodeIntelError):
    """Requested Tree-sitter language is not available.

    Raised when attempting to use a language that is not in the language manifest
    or has not been properly installed. The exception includes metadata about
    available languages to help with error recovery.

    Attributes
    ----------
    problem_type : ClassVar[str]
        Problem type identifier for RFC 9457 Problem Details.
    default_status : ClassVar[int]
        Default HTTP status code (400 Bad Request).
    default_title : ClassVar[str]
        Default error title for Problem Details.

    Parameters
    ----------
    message : str
        Error message.
    extensions : dict[str, Any], optional
        Additional context including 'requested' language and 'available' languages.

    Examples
    --------
    >>> raise LanguageNotSupportedError(
    ...     "Language 'rust' not supported",
    ...     extensions={"requested": "rust", "available": ["python", "json"]},
    ... )
    Traceback (most recent call last):
        ...
    codeintel.errors.LanguageNotSupportedError: Language 'rust' not supported
    """

    problem_type: ClassVar[str] = "language-not-supported"
    default_status: ClassVar[int] = 400
    default_title: ClassVar[str] = "Language not supported"


class QuerySyntaxError(CodeIntelError):
    """Tree-sitter query has invalid syntax.

    Raised when a Tree-sitter query string cannot be compiled due to syntax errors.
    This typically occurs with malformed s-expressions or references to non-existent
    node types for the target language.

    Attributes
    ----------
    problem_type : ClassVar[str]
        Problem type identifier for RFC 9457 Problem Details.
    default_status : ClassVar[int]
        Default HTTP status code (422 Unprocessable Entity).
    default_title : ClassVar[str]
        Default error title for Problem Details.

    Parameters
    ----------
    message : str
        Error message describing the syntax error.
    extensions : dict[str, Any], optional
        Additional context including 'language' and optionally 'query' text.

    Examples
    --------
    >>> raise QuerySyntaxError(
    ...     "Invalid query syntax: unclosed parenthesis", extensions={"language": "python"}
    ... )
    Traceback (most recent call last):
        ...
    codeintel.errors.QuerySyntaxError: Invalid query syntax: unclosed parenthesis
    """

    problem_type: ClassVar[str] = "query-syntax"
    default_status: ClassVar[int] = 422
    default_title: ClassVar[str] = "Invalid query syntax"


class IndexNotFoundError(CodeIntelError):
    """Persistent index database does not exist.

    Raised when attempting to query the persistent symbol index but the database
    file has not been created. Users should run 'codeintel index build' to create
    the index before using search or reference-finding features.

    Attributes
    ----------
    problem_type : ClassVar[str]
        Problem type identifier for RFC 9457 Problem Details.
    default_status : ClassVar[int]
        Default HTTP status code (404 Not Found).
    default_title : ClassVar[str]
        Default error title for Problem Details.

    Parameters
    ----------
    message : str
        Error message.
    extensions : dict[str, Any], optional
        Additional context including 'index_path' and instructions.

    Examples
    --------
    >>> raise IndexNotFoundError(
    ...     "Index not found. Run 'codeintel index build' to create it.",
    ...     extensions={"index_path": ".kgf/codeintel.db"},
    ... )
    Traceback (most recent call last):
        ...
    codeintel.errors.IndexNotFoundError: Index not found...
    """

    problem_type: ClassVar[str] = "index-not-found"
    default_status: ClassVar[int] = 404
    default_title: ClassVar[str] = "Index not found"


class FileTooLargeError(CodeIntelError):
    """File exceeds processing size limits.

    Raised when attempting to parse or process a file that exceeds the configured
    size limits. This prevents resource exhaustion from processing extremely large
    files. The limit can be adjusted via the CODEINTEL_MAX_AST_BYTES environment
    variable.

    Attributes
    ----------
    problem_type : ClassVar[str]
        Problem type identifier for RFC 9457 Problem Details.
    default_status : ClassVar[int]
        Default HTTP status code (413 Payload Too Large).
    default_title : ClassVar[str]
        Default error title for Problem Details.

    Parameters
    ----------
    path : str
        File path that exceeded the limit.
    size : int
        Actual file size in bytes.
    limit : int
        Configured size limit in bytes.

    Examples
    --------
    >>> raise FileTooLargeError("huge.py", 5000000, 1048576)
    Traceback (most recent call last):
        ...
    codeintel.errors.FileTooLargeError: File 'huge.py' (5000000 bytes)...
    """

    problem_type: ClassVar[str] = "file-too-large"
    default_status: ClassVar[int] = 413
    default_title: ClassVar[str] = "File too large"

    def __init__(self, path: str, size: int, limit: int) -> None:
        self.path = path
        self.size = size
        self.limit = limit
        message = f"File '{path}' ({size} bytes) exceeds limit ({limit} bytes)"
        super().__init__(
            message, extensions={"path": path, "size_bytes": size, "limit_bytes": limit}
        )


class ManifestError(CodeIntelError):
    """Language manifest is missing or malformed.

    Raised when the language manifest file (build/languages.json) cannot be found
    or parsed. This typically occurs if 'python -m codeintel.build_languages' has
    not been run to generate the manifest after installing Tree-sitter packages.

    Attributes
    ----------
    problem_type : ClassVar[str]
        Problem type identifier for RFC 9457 Problem Details.
    default_status : ClassVar[int]
        Default HTTP status code (500 Internal Server Error).
    default_title : ClassVar[str]
        Default error title for Problem Details.

    Parameters
    ----------
    message : str
        Error message.
    cause : Exception | None, optional
        Underlying exception that caused this error.

    Examples
    --------
    >>> raise ManifestError("Language manifest not found at build/languages.json")
    Traceback (most recent call last):
        ...
    codeintel.errors.ManifestError: Language manifest not found...
    """

    problem_type: ClassVar[str] = "manifest-error"
    default_status: ClassVar[int] = 500
    default_title: ClassVar[str] = "Language manifest error"


class IndexCorruptedError(CodeIntelError):
    """Persistent index database is corrupted or incompatible.

    Raised when the index database exists but cannot be read due to corruption,
    schema incompatibility, or other integrity issues. Users should rebuild the
    index using 'codeintel index rebuild --confirm'.

    Attributes
    ----------
    problem_type : ClassVar[str]
        Problem type identifier for RFC 9457 Problem Details.
    default_status : ClassVar[int]
        Default HTTP status code (500 Internal Server Error).
    default_title : ClassVar[str]
        Default error title for Problem Details.

    Parameters
    ----------
    message : str
        Error message describing the corruption.
    extensions : dict[str, Any], optional
        Additional context including 'index_path' and recovery instructions.

    Examples
    --------
    >>> raise IndexCorruptedError(
    ...     "Index corrupted: unable to read schema version",
    ...     extensions={"index_path": ".kgf/codeintel.db"},
    ... )
    Traceback (most recent call last):
        ...
    codeintel.errors.IndexCorruptedError: Index corrupted...
    """

    problem_type: ClassVar[str] = "index-corrupted"
    default_status: ClassVar[int] = 500
    default_title: ClassVar[str] = "Index corrupted"


class RateLimitExceededError(CodeIntelError):
    """Request rate limit exceeded.

    Raised when the MCP server rejects a request due to rate limiting. The client
    should back off and retry after a delay. Rate limits can be adjusted via
    CODEINTEL_RATE_LIMIT_QPS and CODEINTEL_RATE_LIMIT_BURST environment variables.

    Attributes
    ----------
    problem_type : ClassVar[str]
        Problem type identifier for RFC 9457 Problem Details.
    default_status : ClassVar[int]
        Default HTTP status code (429 Too Many Requests).
    default_title : ClassVar[str]
        Default error title for Problem Details.

    Parameters
    ----------
    message : str
        Error message.
    extensions : dict[str, Any], optional
        Additional context including 'qps', 'burst', and 'retry_after_s'.

    Examples
    --------
    >>> raise RateLimitExceededError(
    ...     "Rate limit exceeded: 5 requests per second",
    ...     extensions={"qps": 5.0, "burst": 10, "retry_after_s": 1.0},
    ... )
    Traceback (most recent call last):
        ...
    codeintel.errors.RateLimitExceededError: Rate limit exceeded...
    """

    problem_type: ClassVar[str] = "rate-limit"
    default_status: ClassVar[int] = 429
    default_title: ClassVar[str] = "Rate limit exceeded"


class OperationTimeoutError(CodeIntelError):
    """Operation exceeded configured timeout.

    Raised when a tool execution or index operation exceeds the configured timeout
    limit. This prevents runaway operations from consuming server resources. The
    timeout can be adjusted via CODEINTEL_TOOL_TIMEOUT_S environment variable.

    Attributes
    ----------
    problem_type : ClassVar[str]
        Problem type identifier for RFC 9457 Problem Details.
    default_status : ClassVar[int]
        Default HTTP status code (504 Gateway Timeout).
    default_title : ClassVar[str]
        Default error title for Problem Details.

    Parameters
    ----------
    operation : str
        Name of the operation that timed out.
    timeout_s : float
        Configured timeout in seconds.
    extensions : dict[str, Any], optional
        Additional context.

    Examples
    --------
    >>> raise OperationTimeoutError("ts.query", 10.0, extensions={"file": "huge.py"})
    Traceback (most recent call last):
        ...
    codeintel.errors.OperationTimeoutError: Operation 'ts.query' exceeded timeout...
    """

    problem_type: ClassVar[str] = "timeout"
    default_status: ClassVar[int] = 504
    default_title: ClassVar[str] = "Operation timeout"

    def __init__(
        self, operation: str, timeout_s: float, extensions: dict[str, Any] | None = None
    ) -> None:
        self.operation = operation
        self.timeout_s = timeout_s
        message = f"Operation '{operation}' exceeded timeout of {timeout_s}s"
        ext = {"operation": operation, "timeout_s": timeout_s}
        if extensions:
            ext.update(extensions)
        super().__init__(message, extensions=ext)
