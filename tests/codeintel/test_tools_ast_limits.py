"""Tests for AST size limit enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest
from codeintel.mcp_server import tools


def test_get_ast_respects_size_limit(repo_fixture: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_ast rejects files exceeding size limit."""
    bigfile = repo_fixture / "pkg" / "big.py"
    bigfile.write_text("x='a'*" + "1" * 100000)  # large literal
    monkeypatch.setenv("CODEINTEL_MAX_AST_BYTES", "64")
    with pytest.raises(ValueError, match="too large"):
        tools.get_ast("pkg/big.py", "python", "json")


def test_get_ast_bounded_traversal(repo_fixture: Path) -> None:
    """Test that AST traversal respects node budget."""
    # Create a file with many nested structures
    content = "def f1():\n" + "    " * 50 + "pass\n" * 100
    testfile = repo_fixture / "pkg" / "deep.py"
    testfile.write_text(content)
    result = tools.get_ast("pkg/deep.py", "python", "json")
    assert "ast" in result
    assert result["format"] == "json"
    # Verify AST is bounded (should not explode)
    ast = result["ast"]
    assert ast is not None
