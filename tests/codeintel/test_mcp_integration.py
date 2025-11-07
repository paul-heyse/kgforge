"""Integration tests for CodeIntel MCP workflows.

These tests exercise end-to-end scenarios including health checks, file operations,
symbol search, and error handling.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import IO

import pytest


def _ensure_pipe(pipe: IO[str] | None, name: str) -> IO[str]:
    """Return a text stream, raising if the subprocess pipe is not configured.

    Parameters
    ----------
    pipe : IO[str] | None
        Pipe handle returned by :class:`subprocess.Popen`.
    name : str
        Human-readable pipe identifier for diagnostics.

    Returns
    -------
    IO[str]
        Validated pipe ready for I/O operations.

    Raises
    ------
    RuntimeError
        If the subprocess was spawned without the requested pipe in ``text`` mode.
    """
    if pipe is None:
        message = f"Subprocess {name} pipe is not configured; pass text=True and PIPE."
        raise RuntimeError(message)
    return pipe


def _send_rpc(proc: subprocess.Popen[str], method: str, params: dict) -> dict:
    """Send a JSON-RPC request and return the response.

    Parameters
    ----------
    proc : subprocess.Popen[str]
        Running MCP server process with pipes configured in ``text`` mode.
    method : str
        JSON-RPC method name to invoke.
    params : dict
        Request parameters passed to the MCP server.

    Returns
    -------
    dict
        JSON-decoded response payload.

    Raises
    ------
    RuntimeError
        If the server terminates before producing a response line.
    """
    request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    stdin = _ensure_pipe(proc.stdin, "stdin")
    stdout = _ensure_pipe(proc.stdout, "stdout")
    stderr = _ensure_pipe(proc.stderr, "stderr")

    stdin.write(json.dumps(request) + "\n")
    stdin.flush()

    response_line = stdout.readline()
    if not response_line:
        exit_code = proc.poll()
        stderr_output = stderr.read().strip() if exit_code is not None else ""
        message = (
            "No response from server"
            if not stderr_output
            else f"MCP server exited with code {exit_code}: {stderr_output}"
        )
        raise RuntimeError(message)

    return json.loads(response_line)


@pytest.fixture
def mcp_server(repo_fixture: Path) -> Iterator[subprocess.Popen[str]]:
    """Start MCP server in subprocess with test configuration.

    Yields
    ------
    subprocess.Popen[str]
        Handle to the running MCP server configured with text-mode pipes.
    """
    # Set high rate limits for testing to minimise flakes from rate limiting.
    env = {
        "KGF_REPO_ROOT": str(repo_fixture),
        "CODEINTEL_RATE_LIMIT_QPS": "100",
        "CODEINTEL_RATE_LIMIT_BURST": "200",
        "CODEINTEL_TOOL_TIMEOUT_S": "30",
    }

    # Ensure sample files exist within the synthetic repository.
    (repo_fixture / "test.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
    (repo_fixture / "test.json").write_text('{"key": "value"}', encoding="utf-8")

    proc_env = os.environ.copy()
    proc_env.update(env)

    proc = subprocess.Popen(
        [sys.executable, "-m", "codeintel.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=proc_env,
        cwd=str(repo_fixture),
    )

    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def test_mcp_tools_list(mcp_server):
    """Test that tools/list returns all expected tools."""
    response = _send_rpc(mcp_server, "tools/list", {})

    assert "result" in response
    assert "tools" in response["result"]

    tools = response["result"]["tools"]
    tool_names = [t["name"] for t in tools]

    # Verify key tools are present
    assert "code.health" in tool_names
    assert "code.listFiles" in tool_names
    assert "code.getFile" in tool_names
    assert "code.getOutline" in tool_names
    assert "code.searchSymbols" in tool_names
    assert "code.findReferences" in tool_names


def test_health_check(mcp_server):
    """Test comprehensive health check diagnostics."""
    response = _send_rpc(
        mcp_server,
        "tools/call",
        {"name": "code.health", "arguments": {}},
    )

    assert "result" in response
    result = response["result"]

    # Verify health status structure
    assert "status" in result
    assert result["status"] in {"healthy", "degraded", "unhealthy"}
    assert "components" in result

    # Verify component checks
    components = result["components"]
    assert "manifest" in components
    assert "grammars" in components
    assert "queries" in components
    assert "sandbox" in components


def test_list_files(mcp_server):
    """Test file listing with filters."""
    response = _send_rpc(
        mcp_server,
        "tools/call",
        {"name": "code.listFiles", "arguments": {"limit": 10}},
    )

    assert "result" in response
    result = response["result"]

    assert "status" in result
    assert "files" in result
    assert isinstance(result["files"], list)

    # Should include our test files
    files = result["files"]
    assert any("test.py" in f for f in files)


def test_get_file(mcp_server):
    """Test reading file contents."""
    response = _send_rpc(
        mcp_server,
        "tools/call",
        {"name": "code.getFile", "arguments": {"path": "test.py"}},
    )

    assert "result" in response
    result = response["result"]

    assert "status" in result
    assert result["status"] == "ok"
    assert "data" in result
    assert "hello" in result["data"]


def test_get_outline(mcp_server):
    """Test code outline extraction."""
    response = _send_rpc(
        mcp_server,
        "tools/call",
        {"name": "code.getOutline", "arguments": {"path": "test.py", "language": "python"}},
    )

    assert "result" in response
    result = response["result"]

    assert "status" in result
    assert result["status"] == "ok"
    assert "items" in result

    # Should find the 'hello' function
    items = result["items"]
    assert any(item.get("name") == "hello" for item in items)


def test_input_validation_empty_path(mcp_server):
    """Test that empty paths are rejected by validation."""
    response = _send_rpc(
        mcp_server,
        "tools/call",
        {"name": "code.getFile", "arguments": {"path": ""}},
    )

    # Should get validation error
    assert "error" in response


def test_input_validation_null_bytes(mcp_server):
    """Test that null bytes in paths are rejected."""
    response = _send_rpc(
        mcp_server,
        "tools/call",
        {"name": "code.getFile", "arguments": {"path": "test\x00.py"}},
    )

    # Should get validation error
    assert "error" in response


def test_input_validation_negative_offset(mcp_server):
    """Test that negative offsets are rejected."""
    response = _send_rpc(
        mcp_server,
        "tools/call",
        {"name": "code.getFile", "arguments": {"path": "test.py", "offset": -1}},
    )

    # Should get validation error
    assert "error" in response


def test_search_symbols_without_index(mcp_server):
    """Test symbol search returns appropriate error when index doesn't exist."""
    response = _send_rpc(
        mcp_server,
        "tools/call",
        {"name": "code.searchSymbols", "arguments": {"query": "hello"}},
    )

    # Should complete but may have empty results or error about missing index
    assert "result" in response or "error" in response


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("code.listFiles", {"directory": "."}),
        ("code.getFile", {"path": "test.py"}),
        ("code.getOutline", {"path": "test.py", "language": "python"}),
        ("code.health", {}),
    ],
)
def test_tool_timeout_handling(mcp_server, tool, arguments):
    """Test that all tools respect timeout configuration."""
    # This is more of a smoke test - tools should complete quickly
    response = _send_rpc(mcp_server, "tools/call", {"name": tool, "arguments": arguments})

    # Should get a response (not timeout)
    assert "result" in response or "error" in response


def test_invalid_json_handling(mcp_server):
    """Test server handles malformed JSON gracefully."""
    stdin = _ensure_pipe(mcp_server.stdin, "stdin")
    stdout = _ensure_pipe(mcp_server.stdout, "stdout")

    stdin.write("not valid json\n")
    stdin.flush()

    response_line = stdout.readline()
    response = json.loads(response_line)

    # Should get JSON parse error
    assert "error" in response
    assert response["error"]["code"] == -32700


def test_unknown_tool(mcp_server):
    """Test that unknown tool names are handled gracefully."""
    response = _send_rpc(
        mcp_server,
        "tools/call",
        {"name": "code.nonexistent", "arguments": {}},
    )

    # Should get error about unknown tool
    assert "error" in response
