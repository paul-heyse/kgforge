"""Tests for the Python stdio session client."""

from __future__ import annotations

import sys
from pathlib import Path

from kgfoundry.agent_catalog.session import CatalogSession

FIXTURE = Path("tests/fixtures/agent/catalog_sample.json")
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_catalog_session_search_results() -> None:
    """CatalogSession should return search responses from the server."""
    command = [
        sys.executable,
        "-m",
        "tools.agent_catalog.catalogctl_mcp",
        "--catalog",
        str(FIXTURE),
        "--repo-root",
        str(REPO_ROOT),
    ]
    session = CatalogSession(command=command)
    with session:
        capabilities = session.initialize()
        assert "capabilities" in capabilities
        results = session.search("demo", k=1)
        assert results[0]["qname"] == "demo.module.fn"
        anchors = session.open_anchor("4b227777d4dd1fc61c6f884f48641d02")
        assert "editor" in anchors
    session.close()
