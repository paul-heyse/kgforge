from __future__ import annotations

from pathlib import Path

import pytest
from codeintel.mcp_server import tools


@pytest.mark.usefixtures("repo_fixture")
def test_resolve_path_inside() -> None:
    """Test that valid repository paths resolve correctly."""
    p = tools.resolve_path("pkg/mod.py")
    assert p.exists()
    assert p.is_file()


@pytest.mark.usefixtures("repo_fixture")
def test_resolve_path_outside_raises() -> None:
    """Test that paths outside repository raise SandboxError."""
    with pytest.raises(tools.SandboxError):
        tools.resolve_path("../../etc/passwd")


@pytest.mark.usefixtures("repo_fixture")
def test_resolve_directory_valid() -> None:
    """Test that valid directory paths resolve correctly."""
    d = tools.resolve_directory("pkg")
    assert d.is_dir()
    assert d.exists()


def test_resolve_directory_none_returns_root(repo_fixture: Path) -> None:
    """Test that None directory resolves to repository root."""
    d = tools.resolve_directory(None)
    assert d.is_dir()
    assert str(d) == str(repo_fixture.resolve())


@pytest.mark.usefixtures("repo_fixture")
def test_repo_relative() -> None:
    """Test repo_relative helper returns correct relative paths."""
    p = tools.resolve_path("pkg/mod.py")
    rel = tools.repo_relative(p)
    assert rel == "pkg/mod.py"


def test_repo_relative_outside_raises(tmp_path: Path) -> None:
    """Test that repo_relative raises SandboxError for outside paths."""
    outside = tmp_path / "outside.txt"
    outside.write_text("test")
    with pytest.raises(tools.SandboxError):
        tools.repo_relative(outside)
