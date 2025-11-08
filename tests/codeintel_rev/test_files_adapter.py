"""Unit tests for files adapter with scope filtering.

Tests verify that list_paths correctly applies session scope filters
(include_globs, exclude_globs, languages) and respects explicit parameter precedence.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from codeintel_rev.mcp_server.adapters.files import list_paths
from codeintel_rev.mcp_server.schemas import ScopeIn

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_context(tmp_path: Path) -> Mock:
    """Create a mock ApplicationContext for testing.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory for test files.

    Returns
    -------
    Mock
        Mock ApplicationContext with repo_root and paths.
    """
    from codeintel_rev.app.config_context import ResolvedPaths

    context = Mock()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    # Create test files
    (repo_root / "src").mkdir()
    (repo_root / "src" / "main.py").write_text('def main():\n    print("hello")\n')
    (repo_root / "src" / "utils.py").write_text("def helper():\n    pass\n")
    (repo_root / "src" / "app.ts").write_text('function app() {\n    console.log("hello");\n}\n')
    (repo_root / "tests").mkdir()
    (repo_root / "tests" / "test_main.py").write_text("def test_main():\n    assert True\n")
    (repo_root / "README.md").write_text("# Documentation\n")

    paths = ResolvedPaths(
        repo_root=repo_root,
        data_dir=repo_root / "data",
        vectors_dir=repo_root / "data" / "vectors",
        faiss_index=repo_root / "data" / "faiss" / "code.ivfpq.faiss",
        duckdb_path=repo_root / "data" / "catalog.duckdb",
        scip_index=repo_root / "index.scip",
    )
    context.paths = paths

    return context


async def test_list_paths_with_scope_globs(mock_context: Mock) -> None:
    """Test that list_paths applies scope glob filters.

    Verifies that when session scope has include_globs, only files
    matching those patterns are returned.
    """
    scope: ScopeIn = {"include_globs": ["**/*.py"]}

    # Mock session scope retrieval
    with (
        patch(
            "codeintel_rev.mcp_server.adapters.files.get_session_id",
            return_value="test-session-123",
        ),
        patch("codeintel_rev.mcp_server.adapters.files.get_effective_scope", return_value=scope),
    ):
        result = await list_paths(mock_context, path=None, max_results=100)

        # Verify only Python files are returned
        assert "items" in result
        items = result["items"]
        assert isinstance(items, list)
        assert len(items) > 0

        # All returned files should be Python files
        paths_list = [item.get("path", "") for item in items]
        assert all(path.endswith(".py") for path in paths_list if path)
        # Should not include TypeScript or Markdown files
        assert not any(path.endswith(".ts") for path in paths_list if path)
        assert not any(path.endswith(".md") for path in paths_list if path)


async def test_list_paths_with_scope_language(mock_context: Mock) -> None:
    """Test that list_paths applies scope language filters.

    Verifies that when session scope has languages, only files
    matching those language extensions are returned.
    """
    scope: ScopeIn = {"languages": ["python"]}

    # Mock session scope retrieval
    with (
        patch(
            "codeintel_rev.mcp_server.adapters.files.get_session_id",
            return_value="test-session-123",
        ),
        patch("codeintel_rev.mcp_server.adapters.files.get_effective_scope", return_value=scope),
    ):
        result = await list_paths(mock_context, path=None, max_results=100)

        # Verify only Python files are returned
        assert "items" in result
        items = result["items"]
        assert isinstance(items, list)
        assert len(items) > 0

        # All returned files should be Python files
        paths_list = [item.get("path", "") for item in items]
        assert all(path.endswith(".py") for path in paths_list if path)
        # Should not include TypeScript files
        assert not any(path.endswith(".ts") for path in paths_list if path)
