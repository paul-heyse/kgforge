"""Minimal MCP-compatible stdio server exposing Tree-sitter tools."""

from __future__ import annotations

import json
import signal
import sys
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, cast

import anyio
from anyio import to_thread
from codeintel.config import LIMITS
from codeintel.index.store import IndexStore, find_references, search_symbols
from codeintel.mcp_server import tools
from codeintel.mcp_server.ratelimit import TokenBucket
from codeintel.mcp_server.tools import QueryResult
from pydantic import BaseModel, Field, field_validator
from tools import ProblemDetailsParams, build_problem_details, get_logger

_logger = get_logger(__name__)


def _require_non_empty(value: str, field_name: str) -> str:
    """Return ``value`` stripped when non-empty and free of null bytes.

    Parameters
    ----------
    value : str
        Raw field value supplied by the client.
    field_name : str
        Human-readable name used for error messaging.

    Returns
    -------
    str
        Sanitised value with surrounding whitespace removed.

    Raises
    ------
    ValueError
        If ``value`` is empty or contains a null byte.
    """
    stripped = value.strip()
    if not stripped:
        message = f"{field_name} must be non-empty"
        raise ValueError(message)
    if "\x00" in stripped:
        message = f"{field_name} must not contain null bytes"
        raise ValueError(message)
    return stripped


def _require_optional_non_empty(value: str | None, field_name: str) -> str | None:
    """Return an optional string stripped when present and valid.

    This function validates optional string inputs by delegating to
    :func:`_require_non_empty` when a value is provided. It returns None
    when the input is None, allowing for optional field validation in
    Pydantic models.

    Parameters
    ----------
    value : str | None
        Optional string value to validate, or None to skip validation.
    field_name : str
        Human-readable name used for error messaging when validation fails.

    Returns
    -------
    str | None
        Sanitised value with surrounding whitespace removed, or None if
        the input was None.

    Notes
    -----
    This function is designed for validating optional fields in request models.
    It delegates to :func:`_require_non_empty` for the actual validation logic,
    ensuring consistent behavior across required and optional fields.
    """
    if value is None:
        return None
    return _require_non_empty(value, field_name)


def _require_positive(value: int, field_name: str) -> int:
    """Return ``value`` when positive, otherwise raise ``ValueError``.

    This function validates that an integer value is positive (greater than zero).
    It is used for validating limit parameters and other numeric constraints in
    request models.

    Parameters
    ----------
    value : int
        Integer value to validate.
    field_name : str
        Human-readable name used for error messaging when validation fails.

    Returns
    -------
    int
        The validated positive integer.

    Raises
    ------
    ValueError
        If ``value`` is less than one.
    """
    if value < 1:
        message = f"{field_name} must be at least 1"
        raise ValueError(message)
    return value


def _require_within_limit(value: int, field_name: str, maximum: int) -> int:
    """Return ``value`` when bounded by ``maximum`` (inclusive).

    This function validates that an integer value is within a specified range
    [1, maximum]. It first ensures the value is positive, then checks that it
    does not exceed the maximum. Used for validating limit parameters that
    must be bounded to prevent resource exhaustion.

    Parameters
    ----------
    value : int
        Integer value to validate.
    field_name : str
        Human-readable name used for error messaging when validation fails.
    maximum : int
        Maximum allowed value (inclusive upper bound).

    Returns
    -------
    int
        The validated integer within the range [1, maximum].

    Raises
    ------
    ValueError
        If ``value`` is outside the inclusive range ``[1, maximum]``.
    """
    candidate = _require_positive(value, field_name)
    if candidate > maximum:
        message = f"{field_name} must not exceed {maximum}"
        raise ValueError(message)
    return candidate


class QueryRequest(BaseModel):
    """Payload required to execute an arbitrary Tree-sitter query."""

    path: str = Field(..., description="Absolute or repo-relative file path")
    language: str = Field("python", description="Tree-sitter language identifier")
    query: str = Field(..., description="Tree-sitter S-expression to execute")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Return a sanitised path free of null bytes.

        This validator ensures that path fields are non-empty and do not contain
        null bytes, which could be used for path traversal attacks or cause
        filesystem errors.

        Parameters
        ----------
        cls : type[QueryRequest]
            The model class (unused, required by Pydantic).
        v : str
            Raw path value to validate.

        Returns
        -------
        str
            Normalised path string with whitespace stripped.

        Notes
        -----
        Validation errors propagate from :func:`_require_non_empty`.
        """
        return _require_non_empty(v, "path")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Return a sanitised query expression.

        This validator ensures that query fields are non-empty and do not contain
        null bytes, which could cause Tree-sitter parsing errors or security issues.

        Parameters
        ----------
        cls : type[QueryRequest]
            The model class (unused, required by Pydantic).
        v : str
            Raw query value to validate.

        Returns
        -------
        str
            Normalised query string with whitespace stripped.

        Notes
        -----
        Validation errors propagate from :func:`_require_non_empty`.
        """
        return _require_non_empty(v, "query")


class SymbolsRequest(BaseModel):
    """Directory request for summarising Python symbols."""

    directory: str = Field(..., description="Directory to scan for Python modules")

    @field_validator("directory")
    @classmethod
    def validate_directory(cls, v: str) -> str:
        """Return a sanitised directory path.

        This validator ensures that directory fields are non-empty and do not
        contain null bytes, preventing path traversal attacks and filesystem errors.

        Parameters
        ----------
        cls : type[SymbolsRequest]
            The model class (unused, required by Pydantic).
        v : str
            Raw directory value to validate.

        Returns
        -------
        str
            Normalised directory string with whitespace stripped.

        Notes
        -----
        Validation errors propagate from :func:`_require_non_empty`.
        """
        return _require_non_empty(v, "directory")


class CallsRequest(BaseModel):
    """Directory request for call-graph extraction."""

    directory: str = Field(..., description="Directory to scan for call edges")
    language: str = Field("python", description="Language to analyse")
    callee: str | None = Field(
        default=None,
        description="Optional callee name filter",
    )


class ErrorsRequest(BaseModel):
    """Payload describing a file to inspect for syntax errors."""

    path: str = Field(..., description="File to analyse for syntax errors")
    language: str = Field("python", description="Language to analyse")


class ListFilesRequest(BaseModel):
    """Request to list repository files."""

    directory: str | None = Field(None, description="Directory to scan, or None for root")
    glob: str | None = Field(None, description="Optional glob pattern filter")
    limit: int | None = Field(None, description="Maximum number of files to return")

    @field_validator("directory")
    @classmethod
    def validate_directory(cls, v: str | None) -> str | None:
        """Return an optional directory path when valid.

        This validator ensures that optional directory fields are non-empty and
        do not contain null bytes when provided. None values are passed through
        unchanged, allowing for optional directory parameters.

        Parameters
        ----------
        cls : type[ListFilesRequest]
            The model class (unused, required by Pydantic).
        v : str | None
            Raw directory value to validate, or None to skip validation.

        Returns
        -------
        str | None
            Sanitised directory with whitespace stripped, or None if the input
            was None.

        Notes
        -----
        Validation errors propagate from :func:`_require_optional_non_empty`.
        """
        return _require_optional_non_empty(v, "directory")

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v: int | None) -> int | None:
        """Return an optional limit value constrained by server defaults.

        This validator ensures that limit values are positive and do not exceed
        the configured maximum when provided. None values are passed through
        unchanged, allowing for optional limit parameters.

        Parameters
        ----------
        cls : type[ListFilesRequest]
            The model class (unused, required by Pydantic).
        v : int | None
            Raw limit value to validate, or None to skip validation.

        Returns
        -------
        int | None
            Validated limit within the range [1, LIMITS.list_limit_max], or None
            if the input was None.

        Notes
        -----
        Validation errors propagate from :func:`_require_within_limit`.
        """
        if v is None:
            return None
        return _require_within_limit(v, "limit", LIMITS.list_limit_max)


class GetFileRequest(BaseModel):
    """Request to read a file segment."""

    path: str = Field(..., description="Repository-relative file path")
    offset: int = Field(0, description="Byte offset to start reading")
    length: int | None = Field(None, description="Maximum bytes to read")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Return a sanitised repository-relative path.

        This validator ensures that path fields are non-empty and do not contain
        null bytes, which could be used for path traversal attacks or cause
        filesystem errors.

        Parameters
        ----------
        cls : type[GetFileRequest]
            The model class (unused, required by Pydantic).
        v : str
            Raw path value to validate.

        Returns
        -------
        str
            Normalised path string with whitespace stripped.

        Notes
        -----
        Validation errors propagate from :func:`_require_non_empty`.
        """
        return _require_non_empty(v, "path")

    @field_validator("offset")
    @classmethod
    def validate_offset(cls, v: int) -> int:
        """Ensure the offset is non-negative.

        This validator ensures that byte offset values are non-negative, preventing
        invalid file read operations. Negative offsets are not meaningful for
        file reading operations.

        Parameters
        ----------
        cls : type[GetFileRequest]
            The model class (unused, required by Pydantic).
        v : int
            Raw offset value to validate.

        Returns
        -------
        int
            The validated non-negative offset.

        Raises
        ------
        ValueError
            If the offset is negative.
        """
        if v < 0:
            message = "offset must be non-negative"
            raise ValueError(message)
        return v

    @field_validator("length")
    @classmethod
    def validate_length(cls, v: int | None) -> int | None:
        """Ensure the length constraint is positive when provided.

        This validator ensures that length values are positive when provided,
        preventing invalid file read operations. None values are passed through
        unchanged, allowing for unbounded reads.

        Parameters
        ----------
        cls : type[GetFileRequest]
            The model class (unused, required by Pydantic).
        v : int | None
            Raw length value to validate, or None to skip validation.

        Returns
        -------
        int | None
            Validated positive length, or None if the input was None.

        Raises
        ------
        ValueError
            If the provided length is non-positive (less than 1).
            Errors propagate from :func:`_require_positive`.
        """
        if v is None:
            return None
        if v < 1:
            message = "length must be positive"
            raise ValueError(message)
        return v


class OutlineRequest(BaseModel):
    """Request for file outline."""

    path: str = Field(..., description="Repository-relative file path")
    language: str = Field("python", description="Tree-sitter language identifier")


class ASTRequest(BaseModel):
    """Request for AST snapshot."""

    path: str = Field(..., description="Repository-relative file path")
    language: str = Field("python", description="Tree-sitter language identifier")
    format: str = Field("json", description="Output format: json or sexpr")


class SearchSymbolsRequest(BaseModel):
    """Request to search for symbols in the persistent index."""

    query: str = Field(..., description="Search pattern (SQL LIKE)")
    kind: str | None = Field(None, description="Optional symbol kind filter")
    lang: str | None = Field(None, description="Optional language filter")
    limit: int = Field(100, description="Maximum number of results")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Ensure query is non-empty.

        This validator ensures that search query fields are non-empty and do not
        contain null bytes, preventing empty searches and potential security issues.

        Parameters
        ----------
        cls : type[SearchSymbolsRequest]
            The model class (unused, required by Pydantic).
        v : str
            Raw query value to validate.

        Returns
        -------
        str
            Normalised query string with whitespace stripped.

        Notes
        -----
        Validation errors propagate from :func:`_require_non_empty`.
        """
        return _require_non_empty(v, "query")

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v: int) -> int:
        """Constrain search limits to the configured maximum.

        This validator ensures that search limit values are positive and do not
        exceed the configured maximum, preventing resource exhaustion from
        unbounded search results.

        Parameters
        ----------
        cls : type[SearchSymbolsRequest]
            The model class (unused, required by Pydantic).
        v : int
            Raw limit value to validate.

        Returns
        -------
        int
            Validated limit value within the range [1, LIMITS.search_limit_max].

        Notes
        -----
        Validation errors propagate from :func:`_require_within_limit`.
        """
        return _require_within_limit(v, "limit", LIMITS.search_limit_max)


class FindReferencesRequest(BaseModel):
    """Request to find references to a symbol."""

    qualname: str = Field(..., description="Symbol qualname to search for")
    limit: int = Field(100, description="Maximum number of results")

    @field_validator("qualname")
    @classmethod
    def validate_qualname(cls, v: str) -> str:
        """Return a normalised qualname string.

        This validator ensures that qualname fields are non-empty and do not
        contain null bytes, preventing invalid symbol lookups and potential
        security issues.

        Parameters
        ----------
        cls : type[FindReferencesRequest]
            The model class (unused, required by Pydantic).
        v : str
            Raw qualname value to validate.

        Returns
        -------
        str
            Sanitised qualname value with whitespace stripped.

        Notes
        -----
        Validation errors propagate from :func:`_require_non_empty`.
        """
        return _require_non_empty(v, "qualname")

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v: int) -> int:
        """Constrain reference limits to the configured maximum.

        This validator ensures that reference limit values are positive and do not
        exceed the configured maximum, preventing resource exhaustion from
        unbounded reference lookups.

        Parameters
        ----------
        cls : type[FindReferencesRequest]
            The model class (unused, required by Pydantic).
        v : int
            Raw limit value to validate.

        Returns
        -------
        int
            Validated limit value within the range [1, LIMITS.search_limit_max].

        Notes
        -----
        Validation errors propagate from :func:`_require_within_limit`.
        """
        return _require_within_limit(v, "limit", LIMITS.search_limit_max)


class MCPServer:
    """Lightweight JSON-RPC handler for the Model Context Protocol.

    This server implements the MCP (Model Context Protocol) specification, providing
    a standardized interface for code intelligence tools. It handles JSON-RPC messages
    over stdio, dispatches tool calls to Tree-sitter powered handlers, and enforces
    rate limiting and timeouts.

    The server supports multiple tools including:
    - Tree-sitter queries (ts.query)
    - Symbol extraction (ts.symbols)
    - Call graph analysis (ts.calls)
    - Syntax error detection (ts.errors)
    - File operations (code.listFiles, code.getFile)
    - Code navigation (code.getOutline, code.getAST)
    - Symbol search and reference finding (code.searchSymbols, code.findReferences)

    All tool executions are rate-limited and subject to configurable timeouts to
    prevent resource exhaustion. Errors are returned as RFC 9457 Problem Details
    for consistent error handling.

    Attributes
    ----------
    protocol_version : ClassVar[str]
        MCP protocol version string (currently "2024-11-01"). This identifies
        the protocol version supported by this server instance.
    """

    protocol_version: ClassVar[str] = "2024-11-01"

    def __init__(self) -> None:
        self._bucket = TokenBucket(rate=LIMITS.rate_limit_qps, burst=LIMITS.rate_limit_burst)
        self._shutdown_requested = False
        self._tool_handlers = {
            "ts.query": self._tool_ts_query,
            "ts.symbols": self._tool_ts_symbols,
            "ts.calls": self._tool_ts_calls,
            "ts.errors": self._tool_ts_errors,
            "code.listFiles": self._tool_list_files,
            "code.getFile": self._tool_get_file,
            "code.getOutline": self._tool_get_outline,
            "code.getAST": self._tool_get_ast,
            "code.health": self._tool_health,
            "code.searchSymbols": self._tool_search_symbols,
            "code.findReferences": self._tool_find_references,
        }

    def _request_shutdown(self) -> None:
        """Request graceful shutdown of the server."""
        _logger.info("Shutdown requested")
        self._shutdown_requested = True

    async def run(self) -> None:
        """Run the main event loop and process JSON-RPC frames with graceful shutdown support."""

        # Set up signal handlers for graceful shutdown
        def handle_shutdown_signal(signum: int) -> None:
            _logger.warning("Received signal %s, initiating graceful shutdown", signum)
            self._request_shutdown()

        # Register signal handlers
        signal.signal(signal.SIGTERM, lambda sig, _frame: handle_shutdown_signal(sig))
        signal.signal(signal.SIGINT, lambda sig, _frame: handle_shutdown_signal(sig))

        async with anyio.create_task_group() as tg:
            # Cast to help type checker understand the method signature
            tg.start_soon(cast("Any", self._read_loop))

    async def _read_loop(self) -> None:
        """Read and process JSON-RPC messages from stdin."""
        while not self._shutdown_requested:
            line = await to_thread.run_sync(sys.stdin.readline)
            if not line:
                _logger.info("EOF received on stdin, shutting down")
                break
            line = line.strip()
            if not line:
                continue
            if self._shutdown_requested:
                _logger.info("Shutdown requested, stopping read loop")
                break
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                await self._send_error(None, code=-32700, message="Invalid JSON")
                continue
            if "method" in message:
                try:
                    await self._handle_request(message)
                except SystemExit:
                    break

    async def _handle_request(self, message: dict[str, Any]) -> None:
        """Dispatch a JSON-RPC message to the appropriate handler.

        Parameters
        ----------
        message : dict[str, Any]
            JSON-RPC request message containing method name and parameters.

        Raises
        ------
        SystemExit
            If a shutdown request is processed.
        """
        request_id = message.get("id")
        method = message.get("method")
        if method is None:
            await self._send_error(request_id, code=-32600, message="Missing method field")
            return
        params = message.get("params", {})
        if method == "initialize":
            await self._handle_initialize(request_id)
            return
        if method == "tools/list":
            await self._handle_tools_list(request_id)
            return
        if method == "tools/call":
            await self._handle_tools_call(request_id, params)
            return
        if method == "shutdown":
            await self._send_result(request_id, {})
            raise SystemExit(0)
        message_text = f"Unsupported method '{method}'"
        await self._send_error(request_id, code=-32601, message=message_text)

    async def _handle_initialize(self, request_id: object | None) -> None:
        """Handle initialize request.

        Parameters
        ----------
        request_id : object | None
            Request ID for response.
        """
        await self._send_result(
            request_id,
            {
                "protocolVersion": self.protocol_version,
                "server": {"name": "kgfoundry-codeintel", "version": "0.1.0"},
            },
        )

    async def _handle_tools_list(self, request_id: object | None) -> None:
        """Handle tools/list request.

        Parameters
        ----------
        request_id : object | None
            Request ID for response.
        """
        await self._send_result(request_id, {"tools": self._tool_schemas()})

    async def _handle_tools_call(self, request_id: object | None, params: dict[str, Any]) -> None:
        """Handle tools/call request.

        Parameters
        ----------
        request_id : object | None
            Request ID for response.
        params : dict[str, Any]
            Request parameters containing tool name and arguments.
        """
        tool_name = params.get("name")
        if not tool_name:
            await self._send_error(request_id, code=-32600, message="Missing tool name")
            return
        tool_name_str = str(tool_name)
        arguments = params.get("arguments", {})
        handler = self._tool_handlers.get(tool_name_str)
        if handler is None:
            message_text = f"Unknown tool '{tool_name_str}'"
            await self._send_error(request_id, code=-32601, message=message_text)
            return
        # Rate limiting
        if not self._bucket.acquire():
            problem = MCPServer._build_rate_limit_problem(tool_name_str)
            await self._send_error(request_id, code=-32001, message=json.dumps(problem))
            return
        # Execute with timeout
        result = await self._execute_tool_with_timeout(
            handler, arguments, tool_name_str, request_id
        )
        if result is None:
            return
        await self._send_result(request_id, result)

    async def _execute_tool_with_timeout(
        self,
        handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
        arguments: dict[str, Any],
        tool_name: str,
        request_id: object | None,
    ) -> dict[str, Any] | None:
        """Execute tool handler with timeout and error handling.

        Parameters
        ----------
        handler : Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
            Tool handler function.
        arguments : dict[str, Any]
            Tool arguments.
        tool_name : str
            Tool name for error messages.
        request_id : object | None
            Request ID for error responses.

        Returns
        -------
        dict[str, Any] | None
            Tool result or None if error occurred and response was sent.
        """
        result: dict[str, Any] | None = None
        try:
            with anyio.move_on_after(LIMITS.tool_timeout_s) as scope:
                result = await handler(arguments)
            if scope.cancel_called:
                problem = MCPServer._build_timeout_problem(tool_name)
                await self._send_error(request_id, code=-32001, message=json.dumps(problem))
                return None
        except anyio.get_cancelled_exc_class():
            problem = MCPServer._build_cancelled_problem(tool_name)
            await self._send_error(request_id, code=-32001, message=json.dumps(problem))
            return None
        except tools.SandboxError as exc:
            problem = MCPServer._build_sandbox_problem(tool_name, str(exc))
            await self._send_error(request_id, code=-32001, message=json.dumps(problem))
            return None
        except FileNotFoundError as exc:
            problem = MCPServer._build_not_found_problem(tool_name, str(exc))
            await self._send_error(request_id, code=-32001, message=json.dumps(problem))
            return None
        except ValueError as exc:
            problem = MCPServer._build_validation_problem(tool_name, str(exc))
            await self._send_error(request_id, code=-32001, message=json.dumps(problem))
            return None
        return result

    @staticmethod
    def _build_timeout_problem(tool_name: str) -> dict[str, Any]:
        """Build timeout problem details.

        Parameters
        ----------
        tool_name : str
            Tool name that timed out.

        Returns
        -------
        dict[str, Any]
            Problem details dictionary.
        """
        return build_problem_details(
            ProblemDetailsParams(
                type="urn:kgf:problem:codeintel:timeout",
                title="Tool timeout",
                status=504,
                detail=f"Tool '{tool_name}' timed out after {LIMITS.tool_timeout_s}s",
                instance=f"urn:mcp:tool:{tool_name}",
                extensions={"timeout_s": LIMITS.tool_timeout_s},
            )
        )

    @staticmethod
    def _build_cancelled_problem(tool_name: str) -> dict[str, Any]:
        """Build cancelled problem details.

        Parameters
        ----------
        tool_name : str
            Tool name that was cancelled.

        Returns
        -------
        dict[str, Any]
            Problem details dictionary.
        """
        return build_problem_details(
            ProblemDetailsParams(
                type="urn:kgf:problem:codeintel:cancelled",
                title="Cancelled",
                status=499,
                detail=f"Tool '{tool_name}' was cancelled",
                instance=f"urn:mcp:tool:{tool_name}",
            )
        )

    @staticmethod
    def _build_sandbox_problem(tool_name: str, detail: str) -> dict[str, Any]:
        """Build sandbox violation problem details.

        Parameters
        ----------
        tool_name : str
            Tool name that violated sandbox.
        detail : str
            Error detail message.

        Returns
        -------
        dict[str, Any]
            Problem details dictionary.
        """
        return build_problem_details(
            ProblemDetailsParams(
                type="urn:kgf:problem:codeintel:sandbox",
                title="Sandbox violation",
                status=403,
                detail=detail,
                instance=f"urn:mcp:tool:{tool_name}",
                extensions={"code": "KGF-CI-SANDBOX"},
            )
        )

    @staticmethod
    def _build_not_found_problem(tool_name: str, detail: str) -> dict[str, Any]:
        """Build file not found problem details.

        Parameters
        ----------
        tool_name : str
            Tool name that failed.
        detail : str
            Error detail message.

        Returns
        -------
        dict[str, Any]
            Problem details dictionary.
        """
        return build_problem_details(
            ProblemDetailsParams(
                type="urn:kgf:problem:codeintel:not-found",
                title="File not found",
                status=404,
                detail=detail,
                instance=f"urn:mcp:tool:{tool_name}",
                extensions={"code": "KGF-CI-NOT-FOUND"},
            )
        )

    @staticmethod
    def _build_validation_problem(tool_name: str, detail: str) -> dict[str, Any]:
        """Build validation error problem details.

        Parameters
        ----------
        tool_name : str
            Tool name that failed validation.
        detail : str
            Error detail message.

        Returns
        -------
        dict[str, Any]
            Problem details dictionary.
        """
        return build_problem_details(
            ProblemDetailsParams(
                type="urn:kgf:problem:codeintel:validation",
                title="Validation error",
                status=400,
                detail=detail,
                instance=f"urn:mcp:tool:{tool_name}",
                extensions={"code": "KGF-CI-VALIDATION"},
            )
        )

    @staticmethod
    def _build_rate_limit_problem(tool_name: str) -> dict[str, Any]:
        """Build rate limit problem details.

        Parameters
        ----------
        tool_name : str
            Tool name that was rate limited.

        Returns
        -------
        dict[str, Any]
            Problem details dictionary.
        """
        return build_problem_details(
            ProblemDetailsParams(
                type="urn:kgf:problem:codeintel:rate-limit",
                title="Too many requests",
                status=429,
                detail=f"Rate limit exceeded: {LIMITS.rate_limit_qps} QPS",
                instance=f"urn:mcp:tool:{tool_name}",
                extensions={"rate_limit_qps": LIMITS.rate_limit_qps},
            )
        )

    @staticmethod
    def _build_internal_error_problem(tool_name: str) -> dict[str, Any]:
        """Build internal error problem details.

        Parameters
        ----------
        tool_name : str
            Tool name that failed.

        Returns
        -------
        dict[str, Any]
            Problem details dictionary.
        """
        return build_problem_details(
            ProblemDetailsParams(
                type="urn:kgf:problem:codeintel:internal",
                title="Internal error",
                status=500,
                detail=f"Tool '{tool_name}' returned no result",
                instance=f"urn:mcp:tool:{tool_name}",
                extensions={"code": "KGF-CI-INTERNAL"},
            )
        )

    async def _send_result(self, request_id: object | None, result: dict[str, object]) -> None:
        """Emit a JSON-RPC success response."""
        payload = {"jsonrpc": "2.0", "id": request_id, "result": result}
        await self._write(payload)

    async def _send_error(self, request_id: object | None, *, code: int, message: str) -> None:
        """Emit a JSON-RPC error response."""
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
        await self._write(payload)

    @staticmethod
    async def _write(message: dict[str, object]) -> None:
        """Serialise ``message`` to stdout."""
        data = json.dumps(message, separators=(",", ":"))

        # Wrap in functions with explicit return types to help type checker
        # Use cast for pyrefly compatibility with to_thread.run_sync
        def write_output() -> int:
            return sys.stdout.write(data + "\n")

        def flush_output() -> None:
            sys.stdout.flush()

        await to_thread.run_sync(cast("Any", write_output))
        await to_thread.run_sync(cast("Any", flush_output))

    @staticmethod
    def tool_schemas() -> list[dict[str, object]]:
        """Return tool metadata suitable for documentation tooling.

        Returns
        -------
        list[dict[str, object]]
            Tool schemas compliant with the MCP specification.
        """
        return MCPServer._tool_schemas()

    @staticmethod
    def _tool_schemas() -> list[dict[str, object]]:
        """Describe the available tool contracts.

        Returns
        -------
        list[dict[str, object]]
            Tool metadata compliant with the MCP schema.
        """
        return [
            {
                "name": "ts.query",
                "description": "Run a Tree-sitter query against a file.",
                "inputSchema": QueryRequest.model_json_schema(),
            },
            {
                "name": "ts.symbols",
                "description": "List Python symbol definitions in a directory.",
                "inputSchema": SymbolsRequest.model_json_schema(),
            },
            {
                "name": "ts.calls",
                "description": "Enumerate call expressions within a directory.",
                "inputSchema": CallsRequest.model_json_schema(),
            },
            {
                "name": "ts.errors",
                "description": "Report syntax errors detected by Tree-sitter.",
                "inputSchema": ErrorsRequest.model_json_schema(),
            },
            {
                "name": "code.listFiles",
                "description": "List repo files with optional filters.",
                "inputSchema": ListFilesRequest.model_json_schema(),
            },
            {
                "name": "code.getFile",
                "description": "Read a file segment (UTF-8).",
                "inputSchema": GetFileRequest.model_json_schema(),
            },
            {
                "name": "code.getOutline",
                "description": "Return an outline (functions/classes) for a file.",
                "inputSchema": OutlineRequest.model_json_schema(),
            },
            {
                "name": "code.getAST",
                "description": "Return a bounded AST snapshot.",
                "inputSchema": ASTRequest.model_json_schema(),
            },
            {
                "name": "code.health",
                "description": "Return server health status.",
                "inputSchema": {},
            },
            {
                "name": "code.searchSymbols",
                "description": "Search for symbols in the persistent index.",
                "inputSchema": SearchSymbolsRequest.model_json_schema(),
            },
            {
                "name": "code.findReferences",
                "description": "Find references to a symbol by qualname.",
                "inputSchema": FindReferencesRequest.model_json_schema(),
            },
        ]

    @staticmethod
    async def _tool_ts_query(payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the Tree-sitter query tool.

        Parameters
        ----------
        payload : dict[str, Any]
            Tool arguments containing path, language, and query string.

        Returns
        -------
        dict[str, Any]
            Serialised query captures or error Problem Details.

        Notes
        -----
        Requires CODEINTEL_ENABLE_TS_QUERY=1 to be enabled.
        """
        if not LIMITS.enable_ts_query:
            problem = build_problem_details(
                ProblemDetailsParams(
                    type="urn:kgf:problem:codeintel:disabled",
                    title="TS query disabled",
                    status=403,
                    detail="Set CODEINTEL_ENABLE_TS_QUERY=1 to enable",
                    instance="urn:mcp:tool:ts.query",
                    extensions={"code": "KGF-CI-TSQ-DISABLED"},
                )
            )
            return {"status": "error", "problem": problem}
        request = QueryRequest.model_validate(payload)

        # Wrap in function for pyrefly compatibility
        def _run_query() -> QueryResult:
            return tools.run_ts_query(
                request.path,
                language=request.language,
                query=request.query,
            )

        result = await to_thread.run_sync(cast("Any", _run_query))
        return {"status": "ok", "captures": result.captures}

    @staticmethod
    async def _tool_ts_symbols(payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the symbol inventory tool.

        Parameters
        ----------
        payload : dict[str, Any]
            Tool arguments containing directory path.

        Returns
        -------
        dict[str, Any]
            Files paired with symbol metadata.
        """
        request = SymbolsRequest.model_validate(payload)

        # Wrap in function for pyrefly compatibility
        def _list_symbols() -> list[dict[str, Any]]:
            return tools.list_python_symbols(request.directory)

        symbols = await to_thread.run_sync(cast("Any", _list_symbols))
        return {"status": "ok", "files": symbols}

    @staticmethod
    async def _tool_ts_calls(payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the call-graph extraction tool.

        Parameters
        ----------
        payload : dict[str, Any]
            Tool arguments containing directory path, language, and optional callee filter.

        Returns
        -------
        dict[str, Any]
            Call edges extracted from the source files.
        """
        request = CallsRequest.model_validate(payload)

        # Wrap in function for pyrefly compatibility
        def _list_calls() -> list[dict[str, Any]]:
            return tools.list_calls(
                request.directory,
                language=request.language,
                callee=request.callee,
            )

        edges = await to_thread.run_sync(cast("Any", _list_calls))
        return {"status": "ok", "edges": edges}

    @staticmethod
    async def _tool_ts_errors(payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the syntax error detection tool.

        Parameters
        ----------
        payload : dict[str, Any]
            Tool arguments containing file path and language identifier.

        Returns
        -------
        dict[str, Any]
            Syntax error captures grouped by file.
        """
        request = ErrorsRequest.model_validate(payload)

        # Wrap in function for pyrefly compatibility
        def _list_errors() -> list[dict[str, Any]]:
            return tools.list_errors(
                request.path,
                language=request.language,
            )

        errors = await to_thread.run_sync(cast("Any", _list_errors))
        return {"status": "ok", "errors": errors}

    @staticmethod
    async def _tool_list_files(payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the list files tool.

        Parameters
        ----------
        payload : dict[str, Any]
            Tool arguments containing optional directory, glob, and limit.

        Returns
        -------
        dict[str, Any]
            File list with metadata.
        """
        request = ListFilesRequest.model_validate(payload)

        # Wrap in function for pyrefly compatibility
        def _list_files() -> list[str]:
            return tools.list_files(request.directory, request.glob, request.limit)

        items = await to_thread.run_sync(cast("Any", _list_files))
        return {"status": "ok", "files": items, "meta": {"count": len(items)}}

    @staticmethod
    async def _tool_get_file(payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the get file tool.

        Parameters
        ----------
        payload : dict[str, Any]
            Tool arguments containing path, offset, and length.

        Returns
        -------
        dict[str, Any]
            File content and metadata.
        """
        request = GetFileRequest.model_validate(payload)

        # Wrap in function for pyrefly compatibility
        def _get_file() -> dict[str, Any]:
            return tools.get_file(request.path, request.offset, request.length)

        out = await to_thread.run_sync(cast("Any", _get_file))
        return {"status": "ok", **out}

    @staticmethod
    async def _tool_get_outline(payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the get outline tool.

        Parameters
        ----------
        payload : dict[str, Any]
            Tool arguments containing path and language.

        Returns
        -------
        dict[str, Any]
            File outline with items.
        """
        request = OutlineRequest.model_validate(payload)

        # Wrap in function for pyrefly compatibility
        def _get_outline() -> dict[str, Any]:
            return tools.get_outline(request.path, request.language)

        out = await to_thread.run_sync(cast("Any", _get_outline))
        return {"status": "ok", **out}

    @staticmethod
    async def _tool_get_ast(payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the get AST tool.

        Parameters
        ----------
        payload : dict[str, Any]
            Tool arguments containing path, language, and format.

        Returns
        -------
        dict[str, Any]
            AST representation.
        """
        request = ASTRequest.model_validate(payload)
        fmt = request.format

        # Wrap in function for pyrefly compatibility
        def _get_ast() -> dict[str, Any]:
            return tools.get_ast(request.path, request.language, fmt)

        out = await to_thread.run_sync(cast("Any", _get_ast))
        return {"status": "ok", **out}

    @staticmethod
    async def _tool_health(_payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the health check tool.

        Parameters
        ----------
        _payload : dict[str, Any]
            Empty tool arguments (unused).

        Returns
        -------
        dict[str, Any]
            Health status metrics.
        """

        # Wrap in function for pyrefly compatibility
        def _get_health() -> dict[str, Any]:
            return tools.get_health()

        health = await to_thread.run_sync(cast("Any", _get_health))
        return {"status": "ok", **health}

    @staticmethod
    async def _tool_search_symbols(payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the search symbols tool.

        Parameters
        ----------
        payload : dict[str, Any]
            Tool arguments containing query, optional kind/lang filters, and limit.

        Returns
        -------
        dict[str, Any]
            Search results or error Problem Details if index is unavailable.
        """
        request = SearchSymbolsRequest.model_validate(payload)
        repo_root = tools.REPO_ROOT
        db_path = repo_root / ".kgf" / "codeintel.db"

        # Use cast for pyrefly compatibility with to_thread.run_sync
        def _search() -> list[dict[str, object]]:
            if not db_path.exists():
                msg = (
                    f"Index not found at {db_path}. "
                    "Run 'python -m codeintel.cli index build' to create it."
                )
                raise FileNotFoundError(msg)
            with IndexStore(db_path) as store:
                return search_symbols(
                    store,
                    query=request.query,
                    kind=request.kind,
                    lang=request.lang,
                    limit=request.limit,
                )

        try:
            results = await to_thread.run_sync(cast("Any", _search))
            return {"status": "ok", "symbols": results, "count": len(results)}
        except FileNotFoundError as exc:
            problem = build_problem_details(
                ProblemDetailsParams(
                    type="urn:kgf:problem:codeintel:index-not-found",
                    title="Index not found",
                    status=404,
                    detail=str(exc),
                    instance="urn:mcp:tool:code.searchSymbols",
                    extensions={"code": "KGF-CI-INDEX-MISSING"},
                )
            )
            return {"status": "error", "problem": problem}
        except (OSError, ValueError) as exc:  # pragma: no cover - defensive catch
            problem = build_problem_details(
                ProblemDetailsParams(
                    type="urn:kgf:problem:codeintel:internal",
                    title="Search symbols error",
                    status=500,
                    detail=str(exc),
                    instance="urn:mcp:tool:code.searchSymbols",
                    extensions={"code": "KGF-CI-SEARCH-ERROR"},
                )
            )
            return {"status": "error", "problem": problem}

    @staticmethod
    async def _tool_find_references(payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the find references tool.

        Parameters
        ----------
        payload : dict[str, Any]
            Tool arguments containing qualname and limit.

        Returns
        -------
        dict[str, Any]
            Reference results or error Problem Details if index is unavailable.
        """
        request = FindReferencesRequest.model_validate(payload)
        repo_root = tools.REPO_ROOT
        db_path = repo_root / ".kgf" / "codeintel.db"

        # Use cast for pyrefly compatibility with to_thread.run_sync
        def _find() -> list[dict[str, object]]:
            if not db_path.exists():
                msg = (
                    f"Index not found at {db_path}. "
                    "Run 'python -m codeintel.cli index build' to create it."
                )
                raise FileNotFoundError(msg)
            with IndexStore(db_path) as store:
                return find_references(store, qualname=request.qualname, limit=request.limit)

        try:
            results = await to_thread.run_sync(cast("Any", _find))
            return {"status": "ok", "references": results, "count": len(results)}
        except FileNotFoundError as exc:
            problem = build_problem_details(
                ProblemDetailsParams(
                    type="urn:kgf:problem:codeintel:index-not-found",
                    title="Index not found",
                    status=404,
                    detail=str(exc),
                    instance="urn:mcp:tool:code.findReferences",
                    extensions={"code": "KGF-CI-INDEX-MISSING"},
                )
            )
            return {"status": "error", "problem": problem}
        except (OSError, ValueError) as exc:  # pragma: no cover - defensive catch
            problem = build_problem_details(
                ProblemDetailsParams(
                    type="urn:kgf:problem:codeintel:internal",
                    title="Find references error",
                    status=500,
                    detail=str(exc),
                    instance="urn:mcp:tool:code.findReferences",
                    extensions={"code": "KGF-CI-FIND-ERROR"},
                )
            )
            return {"status": "error", "problem": problem}


async def amain() -> None:
    """Run the MCP server using AnyIO's event loop."""
    server = MCPServer()
    await server.run()


def main() -> None:
    """Entry point for command execution."""
    # Cast to help pyrefly understand the async function signature
    anyio.run(cast("Any", amain))


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
