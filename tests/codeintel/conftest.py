"""Test fixtures and configuration for CodeIntel tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a synthetic repository for testing.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Pytest temporary path factory.

    Returns
    -------
    Path
        Path to synthetic repository root.
    """
    r = tmp_path_factory.mktemp("repo")
    (r / "pkg").mkdir()
    (r / "pkg" / "mod.py").write_text("class A:\n    def f(self, x):\n        return x\n\n")
    (r / "README.md").write_text("# sample\n\n```python\nprint('hi')\n```\n")
    (r / "pyproject.toml").write_text('[tool]\nname="demo"\n')
    # Create queries directory structure
    queries_dir = r / "codeintel" / "queries"
    queries_dir.mkdir(parents=True)
    # Create minimal Python query file for tests
    (queries_dir / "python.scm").write_text(
        "(function_definition name: (identifier) @def.name) @def.node\n"
    )
    return r


@pytest.fixture(autouse=True)
def set_env(repo_fixture: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Set environment variables and module state for test isolation.

    Parameters
    ----------
    repo_fixture : Path
        Synthetic repository root.
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("KGF_REPO_ROOT", str(repo_fixture))
    monkeypatch.setenv("CODEINTEL_MAX_AST_BYTES", "65536")
    monkeypatch.setenv("CODEINTEL_LIMIT_MAX", "1000")
    monkeypatch.setenv("CODEINTEL_ENABLE_TS_QUERY", "1")  # enable advanced query in tests
    # Update REPO_ROOT in tools module since it's set at import time
    from codeintel.mcp_server import tools

    monkeypatch.setattr(tools, "REPO_ROOT", repo_fixture.resolve())
    monkeypatch.setattr(tools, "QUERIES_DIR", repo_fixture.resolve() / "codeintel" / "queries")
