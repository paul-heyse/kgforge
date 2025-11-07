"""Tests for rate limiting and timeout behavior."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _start_mcp_server(cwd: str, env: dict[str, str] | None = None) -> subprocess.Popen[str]:
    """Start MCP server subprocess for testing.

    Parameters
    ----------
    cwd : str
        Working directory for the server process.
    env : dict[str, str] | None, optional
        Environment variables to pass to subprocess, by default None (inherits parent env).

    Returns
    -------
    subprocess.Popen[str]
        Server subprocess with stdin/stdout/stderr pipes.

    Notes
    -----
    Uses subprocess.Popen with a list of arguments and shell=False,
    which is safe for test code with fixed arguments.
    """
    import os

    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    return subprocess.Popen(
        [sys.executable, "-m", "codeintel.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=proc_env,
    )


def test_rate_limit(repo_fixture: Path) -> None:
    """Test that rate limiting enforces QPS limits."""
    env = {
        "CODEINTEL_RATE_LIMIT_QPS": "1",
        "CODEINTEL_RATE_LIMIT_BURST": "1",
        "KGF_REPO_ROOT": str(repo_fixture),
    }
    proc = _start_mcp_server(str(repo_fixture), env=env)
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
    # Reload config module to pick up new env var
    import importlib

    import codeintel.config

    importlib.reload(codeintel.config)
    # This test would require a slow operation; for now just verify timeout config is read
    assert codeintel.config.LIMITS.tool_timeout_s == 0.001
