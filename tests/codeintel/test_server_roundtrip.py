"""Integration tests for JSON-RPC round-trip communication."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _start_mcp_server(cwd: str) -> subprocess.Popen[str]:
    """Start MCP server subprocess for testing.

    Parameters
    ----------
    cwd : str
        Working directory for the server process.

    Returns
    -------
    subprocess.Popen[str]
        Server subprocess with stdin/stdout/stderr pipes.

    Notes
    -----
    Uses subprocess.Popen with a list of arguments and shell=False,
    which is safe for test code with fixed arguments.
    """
    return subprocess.Popen(
        [sys.executable, "-m", "codeintel.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
    )


def _rpc(proc: subprocess.Popen[str], payload: dict[str, object]) -> dict[str, Any]:
    """Send JSON-RPC request and read response.

    Parameters
    ----------
    proc : subprocess.Popen[str]
        Server subprocess with stdin/stdout pipes.
    payload : dict[str, object]
        JSON-RPC request payload.

    Returns
    -------
    dict[str, Any]
        JSON-RPC response payload.
    """
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    assert line, "no response"
    return json.loads(line)


def test_tools_list(repo_fixture: Path) -> None:
    """Test tools/list method returns tool schemas."""
    proc = _start_mcp_server(str(repo_fixture))
    try:
        msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        res = _rpc(proc, msg)
        assert "result" in res
        assert "tools" in res["result"]
        assert res["result"]["tools"]  # non-empty
    finally:
        proc.kill()
        proc.wait()


def test_get_outline_call(repo_fixture: Path) -> None:
    """Test tools/call with code.getOutline."""
    proc = _start_mcp_server(str(repo_fixture))
    try:
        # tools/list (to discover schema) — optional
        _ = _rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        # tools/call
        call = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "code.getOutline",
                "arguments": {"path": "pkg/mod.py", "language": "python"},
            },
        }
        res = _rpc(proc, call)
        assert "result" in res
        assert res["result"]["status"] == "ok"
        assert "items" in res["result"]
        assert any(i["name"] == "f" for i in res["result"]["items"])
    finally:
        proc.kill()
        proc.wait()


def test_list_files_call(repo_fixture: Path) -> None:
    """Test tools/call with code.listFiles."""
    proc = _start_mcp_server(str(repo_fixture))
    try:
        call = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "code.listFiles", "arguments": {"limit": 5}},
        }
        res = _rpc(proc, call)
        assert "result" in res
        assert res["result"]["status"] == "ok"
        assert "files" in res["result"]
        assert isinstance(res["result"]["files"], list)
    finally:
        proc.kill()
        proc.wait()
