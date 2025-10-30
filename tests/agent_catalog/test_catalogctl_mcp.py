"""Tests for the stdio JSON-RPC catalog server."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

FIXTURE = Path("tests/fixtures/agent/catalog_sample.json").resolve()
REPO_ROOT = Path(__file__).resolve().parents[2]


def _rpc(process: subprocess.Popen[str], payload: dict[str, Any]) -> dict[str, Any]:
    """Send a JSON-RPC request to ``process`` and decode the response."""
    stdin = process.stdin
    stdout = process.stdout
    if stdin is None or stdout is None:
        message = "catalogctl-mcp stdio streams are unavailable"
        raise RuntimeError(message)
    stdin.write(json.dumps(payload) + "\n")
    stdin.flush()
    line = stdout.readline()
    if not line:
        message = "catalogctl-mcp terminated unexpectedly"
        raise RuntimeError(message)
    return json.loads(line)


def test_catalogctl_mcp_session_round_trip() -> None:
    """The stdio server should respond to catalog queries and shut down cleanly."""
    command = [
        sys.executable,
        "-m",
        "tools.agent_catalog.catalogctl_mcp",
        "--catalog",
        str(FIXTURE),
        "--repo-root",
        str(REPO_ROOT),
    ]
    process = subprocess.Popen(  # noqa: S603 - command uses trusted interpreter
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        response = _rpc(process, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert "result" in response
        capabilities = response["result"]["capabilities"]["procedures"]
        assert "catalog.search" in capabilities

        search_response = _rpc(
            process,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "catalog.search",
                "params": {"query": "demo", "k": 1},
            },
        )
        assert search_response["result"][0]["qname"] == "demo.module.fn"

        shutdown = _rpc(
            process, {"jsonrpc": "2.0", "id": 3, "method": "session.shutdown", "params": {}}
        )
        assert shutdown["result"] is None
    finally:
        if process.stdin:
            process.stdin.close()
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
            process.kill()
