"""Tests for the agent catalog builder CLI."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Protocol

import pytest
from tools.docs import build_agent_catalog


class SupportsSetEnv(Protocol):
    def setenv(self, name: str, value: str, prepend: str | None = ...) -> None:
        """Set an environment variable for the duration of the test."""


@pytest.fixture()
def repo_root() -> Path:
    """Return the repository root path."""
    return Path(__file__).resolve().parents[2]


def test_help_succeeds(repo_root: Path) -> None:
    """The CLI help output should render without errors."""
    script = repo_root / "tools" / "docs" / "build_agent_catalog.py"
    result = subprocess.run(
        ["uv", "run", "python", str(script), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--output" in result.stdout


def test_build_catalog_smoke(tmp_path: Path, repo_root: Path) -> None:
    """Building the catalog should produce a JSON document."""
    output_path = tmp_path / "agent_catalog.json"
    shard_dir = tmp_path / "shards"
    args = build_agent_catalog.parse_args(
        [
            "--output",
            str(output_path),
            "--schema",
            "docs/_build/schema_agent_catalog.json",
            "--shard-dir",
            str(shard_dir),
        ]
    )
    args.repo_root = repo_root
    builder = build_agent_catalog.AgentCatalogBuilder(args)
    catalog = builder.build()
    assert catalog.packages, "expected at least one package"
    builder.write(catalog, args.output, args.schema)
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["link_policy"]["mode"] in {"editor", "github"}
    assert data["packages"], "expected packages in catalog"
    first_package = data["packages"][0]
    first_module = first_package["modules"][0]
    assert first_module["graph"]["imports"] is not None
    assert first_module["symbols"], "expected symbols for module"


def test_sharded_catalog_load(tmp_path: Path, repo_root: Path) -> None:
    """When thresholds are low the builder should emit shards that can be loaded."""
    output_path = tmp_path / "catalog.json"
    shard_dir = tmp_path / "catalog_shards"
    args = build_agent_catalog.parse_args(
        [
            "--output",
            str(output_path),
            "--schema",
            "docs/_build/schema_agent_catalog.json",
            "--shard-dir",
            str(shard_dir),
            "--max-modules-per-shard",
            "1",
            "--max-symbols-per-shard",
            "1",
        ]
    )
    args.repo_root = repo_root
    builder = build_agent_catalog.AgentCatalogBuilder(args)
    catalog = builder.build()
    builder.write(catalog, args.output, args.schema)
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["shards"] is not None
    loaded = build_agent_catalog.load_catalog(output_path)
    assert loaded["packages"], "expected packages from shard loader"


def test_link_policy_cli_precedence(monkeypatch: SupportsSetEnv, repo_root: Path) -> None:
    """CLI link policy options should override environment variables."""
    monkeypatch.setenv("DOCS_LINK_MODE", "editor")
    monkeypatch.setenv("DOCS_GITHUB_ORG", "env-org")
    monkeypatch.setenv("DOCS_GITHUB_REPO", "env-repo")
    monkeypatch.setenv("DOCS_GITHUB_SHA", "env-sha")
    args = build_agent_catalog.parse_args(
        [
            "--link-mode",
            "github",
            "--github-org",
            "cli-org",
            "--github-repo",
            "cli-repo",
            "--github-sha",
            "cli-sha",
        ]
    )
    args.repo_root = repo_root
    builder = build_agent_catalog.AgentCatalogBuilder(args)
    policy = builder._resolve_link_policy()
    assert policy.mode == "github"
    assert policy.github == {"org": "cli-org", "repo": "cli-repo", "sha": "cli-sha"}
