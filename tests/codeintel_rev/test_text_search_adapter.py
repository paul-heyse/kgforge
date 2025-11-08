"""Unit tests for text search adapter with scope filtering.

Tests verify that text search correctly applies session scope filters
and respects explicit parameter precedence.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from codeintel_rev.mcp_server.adapters.text_search import search_text
from codeintel_rev.mcp_server.schemas import ScopeIn


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
    (repo_root / "tests").mkdir()
    (repo_root / "tests" / "test_main.py").write_text("def test_main():\n    assert True\n")

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


def test_search_text_with_scope_paths(mock_context: Mock) -> None:
    """Test that search_text applies scope path filters.

    Verifies that when session scope has include_globs, only files
    matching those patterns are searched.
    """
    scope: ScopeIn = {"include_globs": ["src/**"]}

    # Mock session scope retrieval
    with (
        patch(
            "codeintel_rev.mcp_server.adapters.text_search.get_session_id",
            return_value="test-session-123",
        ),
        patch(
            "codeintel_rev.mcp_server.adapters.text_search.get_effective_scope", return_value=scope
        ),
        patch("codeintel_rev.mcp_server.adapters.text_search.run_subprocess") as mock_run,
    ):
        mock_run.return_value = "src/main.py:1:def main():\n"

        result = search_text(mock_context, "main", max_results=50)

        # Verify ripgrep was called with src/ paths
        assert mock_run.called
        call_args = mock_run.call_args[0] if mock_run.call_args[0] else []
        cmd = call_args[0] if call_args else []
        if isinstance(cmd, list):
            # Check that paths include src/
            paths_in_cmd = [arg for arg in cmd if isinstance(arg, str) and "src" in arg]
            assert len(paths_in_cmd) > 0

        # Verify results
        assert "matches" in result
        assert isinstance(result["matches"], list)


def test_search_text_explicit_override(mock_context: Mock) -> None:
    """Test that explicit paths parameter overrides session scope.

    Verifies that when both session scope and explicit paths are provided,
    explicit paths take precedence.
    """
    scope: ScopeIn = {"include_globs": ["src/**"]}

    # Mock session scope retrieval
    with (
        patch(
            "codeintel_rev.mcp_server.adapters.text_search.get_session_id",
            return_value="test-session-123",
        ),
        patch(
            "codeintel_rev.mcp_server.adapters.text_search.get_effective_scope", return_value=scope
        ),
        patch("codeintel_rev.mcp_server.adapters.text_search.run_subprocess") as mock_run,
    ):
        mock_run.return_value = "tests/test_main.py:1:def test_main():\n"

        # Call with explicit paths (should override scope)
        result = search_text(mock_context, "test", paths=["tests/"], max_results=50)

        # Verify ripgrep was called with tests/ paths (not src/)
        assert mock_run.called
        call_args = mock_run.call_args[0] if mock_run.call_args[0] else []
        cmd = call_args[0] if call_args else []
        if isinstance(cmd, list):
            # Explicit paths should override scope
            assert any("tests" in str(arg) for arg in cmd if isinstance(arg, str))
            # Should not include src/ paths
            assert not any(
                "src" in str(arg) for arg in cmd if isinstance(arg, str) and arg == "src"
            )

        # Verify results
        assert "matches" in result
        assert isinstance(result["matches"], list)
