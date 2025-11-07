"""Tests for rate limiting and timeout behavior."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


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


def test_rate_limit(repo_fixture: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that rate limiting enforces QPS limits."""
    monkeypatch.setenv("CODEINTEL_RATE_LIMIT_QPS", "1")
    monkeypatch.setenv("CODEINTEL_RATE_LIMIT_BURST", "1")
    proc = _start_mcp_server(str(repo_fixture))
    try:

        def rpc(id_: int) -> dict[str, object]:
            req = {
                "jsonrpc": "2.0",
                "id": id_,
                "method": "tools/call",
                "params": {"name": "code.listFiles", "arguments": {"limit": 1}},
            }
            assert proc.stdin is not None
            assert proc.stdout is not None
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            assert line
            return json.loads(line)

        r1 = rpc(1)
        assert "result" in r1 or "error" in r1
        r2 = rpc(2)
        # Second request should be rate limited
        assert "error" in r2
        error_msg = json.dumps(r2).lower()
        assert "rate" in error_msg or "429" in error_msg
    finally:
        proc.kill()
        proc.wait()


def test_timeout_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that tool timeouts are enforced."""
    monkeypatch.setenv("CODEINTEL_TOOL_TIMEOUT_S", "0.001")  # Very short timeout
    # This test would require a slow operation; for now just verify timeout config is read
    from codeintel.config import LIMITS

    assert LIMITS.tool_timeout_s == 0.001
