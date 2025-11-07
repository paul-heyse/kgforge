"""Tests for outline extraction."""

from __future__ import annotations

from pathlib import Path

import pytest
from codeintel.mcp_server import tools


@pytest.mark.usefixtures("repo_fixture")
def test_outline_simple() -> None:
    """Test basic outline extraction for Python file."""
    out = tools.get_outline("pkg/mod.py", "python")
    assert out["path"].endswith("pkg/mod.py")
    assert isinstance(out["items"], list)
    assert any(i["name"] == "f" for i in out["items"])


def test_outline_empty_for_missing_query(repo_fixture: Path) -> None:
    """Test that missing query files return empty outline."""
    testfile = repo_fixture / "test.txt"
    testfile.write_text("plain text")
    out = tools.get_outline("test.txt", "python")
    assert out["items"] == []


def test_outline_respects_limit(repo_fixture: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that outline respects max_outline_items limit."""
    monkeypatch.setenv("CODEINTEL_MAX_OUTLINE_ITEMS", "1")
    # Create file with multiple functions
    content = "def f1():\n    pass\ndef f2():\n    pass\ndef f3():\n    pass\n"
    testfile = repo_fixture / "pkg" / "multi.py"
    testfile.write_text(content)
    out = tools.get_outline("pkg/multi.py", "python")
    assert len(out["items"]) <= 1
