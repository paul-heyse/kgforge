"""Integration tests for MCP server.

Tests the full MCP server with tool calls.
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_mcp_server_import() -> None:
    """Test that MCP server can be imported."""
    from codeintel_rev.mcp_server.server import asgi_app, mcp

    assert mcp is not None
    assert asgi_app is not None


@pytest.mark.asyncio
async def test_file_operations() -> None:
    """Test file listing and opening."""
    from codeintel_rev.mcp_server.adapters.files import list_paths, open_file

    # Test list_paths
    result = list_paths(max_results=10)
    assert "items" in result
    assert isinstance(result["items"], list)

    # Test open_file with README
    result = open_file("README.md")
    # May or may not exist depending on test environment
    assert "content" in result or "error" in result


@pytest.mark.asyncio
async def test_text_search() -> None:
    """Test text search functionality."""
    from codeintel_rev.mcp_server.adapters.text_search import search_text

    result = search_text("def", max_results=5)
    assert "matches" in result
    assert isinstance(result["matches"], list)


@pytest.mark.asyncio
async def test_semantic_search_no_index() -> None:
    """Test semantic search gracefully handles missing index."""
    from codeintel_rev.mcp_server.adapters.semantic import semantic_search

    result = await semantic_search("test query", limit=5)
    assert "answer" in result
    assert "findings" in result
    # Should handle missing index gracefully
    assert isinstance(result["findings"], list)


@pytest.mark.asyncio
async def test_git_history() -> None:
    """Test git blame and history."""
    from codeintel_rev.mcp_server.adapters.history import blame_range, file_history

    # Test blame_range
    result = blame_range("README.md", 1, 5)
    assert "blame" in result
    assert isinstance(result["blame"], list)

    # Test file_history
    result = file_history("README.md", limit=5)
    assert "commits" in result
    assert isinstance(result["commits"], list)


@pytest.mark.asyncio
async def test_scope_operations() -> None:
    """Test scope setting."""
    from codeintel_rev.mcp_server.adapters.files import set_scope

    result = set_scope({"repos": ["test"], "languages": ["python"]})
    assert result["status"] == "ok"
    assert "effective_scope" in result
