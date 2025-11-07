"""Minimal MCP-compatible stdio server exposing Tree-sitter tools."""

from __future__ import annotations

import json
import sys
from collections.abc import Awaitable, Callable
from typing import Any, cast

import anyio
from anyio import to_thread
from pydantic import BaseModel, Field
from tools import ProblemDetailsParams, build_problem_details

from codeintel.config import LIMITS
from codeintel.index.store import IndexStore, find_references, search_symbols
from codeintel.mcp_server import tools
from codeintel.mcp_server.ratelimit import TokenBucket


class QueryRequest(BaseModel):
    """Payload required to execute an arbitrary Tree-sitter query."""

    path: str = Field(..., description="Absolute or repo-relative file path")
    language: str = Field("python", description="Tree-sitter language identifier")
    query: str = Field(..., description="Tree-sitter S-expression to execute")


class SymbolsRequest(BaseModel):
    """Directory request for summarising Python symbols."""

    directory: str = Field(..., description="Directory to scan for Python modules")


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


class GetFileRequest(BaseModel):
    """Request to read a file segment."""

    path: str = Field(..., description="Repository-relative file path")
    offset: int = Field(0, description="Byte offset to start reading")
    length: int | None = Field(None, description="Maximum bytes to read")


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


class FindReferencesRequest(BaseModel):
    """Request to find references to a symbol."""

    qualname: str = Field(..., description="Symbol qualname to search for")
    limit: int = Field(100, description="Maximum number of results")


class MCPServer:
    """Lightweight JSON-RPC handler for the Model Context Protocol."""

    protocol_version = "2024-11-01"

    def __init__(self) -> None:
        """Initialize MCP server with tool handlers and rate limiter."""
        self._bucket = TokenBucket(rate=LIMITS.rate_limit_qps, burst=LIMITS.rate_limit_burst)
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

    async def run(self) -> None:
        """Run the main event loop and process JSON-RPC frames."""
        async with anyio.create_task_group() as tg:
            # Cast to help type checker understand the method signature
            tg.start_soon(cast("Any", self._read_loop))

    async def _read_loop(self) -> None:
        """Read and process JSON-RPC messages from stdin."""
        while True:
            line = await to_thread.run_sync(sys.stdin.readline)
            if not line:
                break
            line = line.strip()
            if not line:
                continue
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
        def write_output() -> int:
            return sys.stdout.write(data + "\n")

        def flush_output() -> None:
            sys.stdout.flush()

        await to_thread.run_sync(write_output)
        await to_thread.run_sync(flush_output)

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
        result = await to_thread.run_sync(
            lambda: tools.run_ts_query(
                request.path,
                language=request.language,
                query=request.query,
            )
        )
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
        symbols = await to_thread.run_sync(lambda: tools.list_python_symbols(request.directory))
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
        edges = await to_thread.run_sync(
            lambda: tools.list_calls(
                request.directory,
                language=request.language,
                callee=request.callee,
            )
        )
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
        errors = await to_thread.run_sync(
            lambda: tools.list_errors(
                request.path,
                language=request.language,
            )
        )
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
        items = await to_thread.run_sync(
            lambda: tools.list_files(request.directory, request.glob, request.limit)
        )
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
        out = await to_thread.run_sync(
            lambda: tools.get_file(request.path, request.offset, request.length)
        )
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
        out = await to_thread.run_sync(lambda: tools.get_outline(request.path, request.language))
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
        out = await to_thread.run_sync(lambda: tools.get_ast(request.path, request.language, fmt))
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
        health = await to_thread.run_sync(tools.get_health)
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
            results = await to_thread.run_sync(_search)
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
            results = await to_thread.run_sync(_find)
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
    anyio.run(amain)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
