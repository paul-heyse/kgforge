"""Minimal MCP-compatible stdio server exposing Tree-sitter tools."""

from __future__ import annotations

import json
import sys
from typing import Any

import anyio
from anyio import to_thread
from pydantic import BaseModel, Field

from codeintel.mcp_server import tools


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


class MCPServer:
    """Lightweight JSON-RPC handler for the Model Context Protocol."""

    protocol_version = "2024-11-01"

    def __init__(self) -> None:
        self._tool_handlers = {
            "ts.query": self._tool_ts_query,
            "ts.symbols": self._tool_ts_symbols,
            "ts.calls": self._tool_ts_calls,
            "ts.errors": self._tool_ts_errors,
        }

    async def run(self) -> None:
        """Run the main event loop and process JSON-RPC frames."""
        async with anyio.create_task_group() as tg:
            tg.start_soon(self._read_loop)

    async def _read_loop(self) -> None:
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
            await self._send_result(
                request_id,
                {
                    "protocolVersion": self.protocol_version,
                    "server": {"name": "kgfoundry-ts", "version": "0.1.0"},
                },
            )
            return
        if method == "tools/list":
            await self._send_result(request_id, {"tools": self._tool_schemas()})
            return
        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            handler = self._tool_handlers.get(str(tool_name))
            if handler is None:
                message_text = f"Unknown tool '{tool_name}'"
                await self._send_error(request_id, code=-32601, message=message_text)
                return
            try:
                result = await handler(arguments)
            except (FileNotFoundError, ValueError) as exc:  # pragma: no cover - runtime guard
                await self._send_error(request_id, code=-32001, message=str(exc))
                return
            await self._send_result(request_id, result)
            return
        if method == "shutdown":
            await self._send_result(request_id, {})
            raise SystemExit(0)
        message_text = f"Unsupported method '{method}'"
        await self._send_error(request_id, code=-32601, message=message_text)

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
        await to_thread.run_sync(sys.stdout.write, data + "\n")
        await to_thread.run_sync(sys.stdout.flush)

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
        ]

    @staticmethod
    async def _tool_ts_query(payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the Tree-sitter query tool.

        Returns
        -------
        dict[str, Any]
            Serialised query captures.
        """
        request = QueryRequest.model_validate(payload)
        result = await to_thread.run_sync(
            lambda: tools.run_ts_query(
                request.path,
                language=request.language,
                query=request.query,
            )
        )
        return {"captures": result.captures}

    @staticmethod
    async def _tool_ts_symbols(payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the symbol inventory tool.

        Returns
        -------
        dict[str, Any]
            Files paired with symbol metadata.
        """
        request = SymbolsRequest.model_validate(payload)
        symbols = await to_thread.run_sync(lambda: tools.list_python_symbols(request.directory))
        return {"files": symbols}

    @staticmethod
    async def _tool_ts_calls(payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the call-graph extraction tool.

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
        return {"edges": edges}

    @staticmethod
    async def _tool_ts_errors(payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the syntax error detection tool.

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
        return {"errors": errors}


async def amain() -> None:
    """Run the MCP server using AnyIO's event loop."""
    server = MCPServer()
    await server.run()


def main() -> None:
    """Entry point for command execution."""
    anyio.run(amain)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
