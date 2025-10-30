"""Tests for the agent catalog builder CLI."""

from __future__ import annotations

import json
import subprocess
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pytest
from tools.docs import build_agent_catalog


class SupportsSetEnv(Protocol):
    def setenv(self, name: str, value: str, prepend: str | None = None) -> None:
        """Set an environment variable for the duration of the test."""


@pytest.fixture()
def fake_embedding_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a deterministic embedding backend for tests."""

    class FakeModel:
        dimension = 6

        def encode(self, sentences: Sequence[str], **_: object) -> npt.NDArray[np.float32]:
            vectors: npt.NDArray[np.float32] = np.zeros(
                (len(sentences), self.dimension), dtype=np.float32
            )
            for idx, sentence in enumerate(sentences):
                seed = sum(ord(char) for char in sentence)
                for column in range(self.dimension):
                    vectors[idx, column] = ((seed + (column + 1) * 17) % 101) / 100.0
                norm = np.linalg.norm(vectors[idx])
                if norm:
                    vectors[idx] /= norm
            return vectors

    def loader(_: str) -> FakeModel:
        return FakeModel()

    monkeypatch.setattr(build_agent_catalog, "_load_embedding_model", loader)


@pytest.fixture()
def repo_root() -> Path:
    """Return the repository root path."""
    return Path(__file__).resolve().parents[2]


def test_help_succeeds(fake_embedding_backend: None, repo_root: Path) -> None:
    """The CLI help output should render without errors."""
    del fake_embedding_backend
    script = repo_root / "tools" / "docs" / "build_agent_catalog.py"
    result = subprocess.run(
        ["uv", "run", "python", str(script), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--output" in result.stdout


def test_build_catalog_smoke(fake_embedding_backend: None, tmp_path: Path, repo_root: Path) -> None:
    """Building the catalog should produce a JSON document."""
    del fake_embedding_backend
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
    assert catalog.semantic_index is not None
    builder.write(catalog, args.output, args.schema)
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["link_policy"]["mode"] in {"editor", "github"}
    assert data["packages"], "expected packages in catalog"
    assert data["semantic_index"]["index"].endswith(".faiss")
    assert data["artifacts"]["semantic_index"].endswith(".faiss")
    assert data["search"]["lexical_fields"]
    first_package = data["packages"][0]
    first_module = first_package["modules"][0]
    assert first_module["graph"]["imports"] is not None
    assert first_module["symbols"], "expected symbols for module"


def test_sharded_catalog_load(
    fake_embedding_backend: None, tmp_path: Path, repo_root: Path
) -> None:
    """When thresholds are low the builder should emit shards that can be loaded."""
    del fake_embedding_backend
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


def test_search_catalog_hybrid(
    fake_embedding_backend: None, tmp_path: Path, repo_root: Path
) -> None:
    """Hybrid search should return ranked results from the semantic index."""
    del fake_embedding_backend
    output_path = tmp_path / "catalog.json"
    args = build_agent_catalog.parse_args(
        [
            "--output",
            str(output_path),
            "--schema",
            "docs/_build/schema_agent_catalog.json",
        ]
    )
    args.repo_root = repo_root
    builder = build_agent_catalog.AgentCatalogBuilder(args)
    catalog = builder.build()
    builder.write(catalog, args.output, args.schema)
    catalog_data = build_agent_catalog.load_catalog(output_path)
    results = build_agent_catalog.search_catalog(
        catalog_data,
        repo_root=repo_root,
        query="catalog",
        k=5,
    )
    assert results, "expected hybrid search results"
    assert all(0.0 <= result.score <= 1.0 for result in results)
    assert any(result.lexical_score > 0.0 or result.vector_score > 0.0 for result in results)


def test_cli_search_outputs_json(
    fake_embedding_backend: None,
    tmp_path: Path,
    repo_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI search mode should print JSON payloads."""
    del fake_embedding_backend
    output_path = tmp_path / "catalog.json"
    args = build_agent_catalog.parse_args(
        [
            "--output",
            str(output_path),
            "--schema",
            "docs/_build/schema_agent_catalog.json",
        ]
    )
    args.repo_root = repo_root
    builder = build_agent_catalog.AgentCatalogBuilder(args)
    catalog = builder.build()
    builder.write(catalog, args.output, args.schema)
    exit_code = build_agent_catalog.main(
        [
            "--repo-root",
            str(repo_root),
            "--output",
            str(output_path),
            "--search-query",
            "catalog",
            "--search-k",
            "2",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert isinstance(payload, list)
    assert payload, "expected CLI search results"
    assert len(payload) <= 2


def test_build_anchors_use_docfacts_end_lineno(tmp_path: Path) -> None:
    """DocFacts ``end_lineno`` metadata should populate anchor ranges."""

    module_path = tmp_path / "pkg" / "module.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(
        textwrap.dedent(
            """
            def example(value: int) -> int:
                """Return ``value`` unchanged."""

                return value
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    module_name = "pkg.module"
    qname = f"{module_name}.example"
    args = build_agent_catalog.parse_args([])
    args.repo_root = tmp_path
    builder = build_agent_catalog.AgentCatalogBuilder(args)
    builder._docfacts_index = {qname: {"qname": qname, "lineno": 1, "end_lineno": 4}}
    analyzer = build_agent_catalog.ModuleAnalyzer(module_name, module_path)
    node = analyzer.get_node(qname)
    anchors = builder._build_anchors(qname, analyzer, node, builder._docfacts_index[qname])
    assert anchors.start_line == 1
    assert anchors.end_line == 4
    assert anchors.remap_order, "expected remap_order entries"
    remap = anchors.remap_order[0]
    assert remap["symbol_id"]
    assert remap["cst_fingerprint"] == anchors.cst_fingerprint
    assert remap["name_arity"] == 1
    assert "example" in (remap.get("nearest_text") or "")


def test_load_faiss_falls_back_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing FAISS modules should fall back to the in-memory implementation."""
    monkeypatch.setattr(
        build_agent_catalog,
        "_FAISS_DEFAULT_MODULES",
        ("not_a_real_module",),
        raising=False,
    )
    module = build_agent_catalog._load_faiss("unit-test")
    index = module.IndexFlatIP(2)
    vectors = np.zeros((1, 2), dtype=np.float32)
    index.add(vectors)
    distances, indices = index.search(vectors, 1)
    assert distances.shape == (1, 1)
    assert indices.shape == (1, 1)


def test_load_faiss_respects_fallback_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabling the fallback should raise a CatalogBuildError when FAISS is missing."""
    monkeypatch.setattr(
        build_agent_catalog,
        "_FAISS_DEFAULT_MODULES",
        ("not_a_real_module",),
        raising=False,
    )
    monkeypatch.setenv("KGF_DISABLE_FAISS_FALLBACK", "1")
    with pytest.raises(build_agent_catalog.CatalogBuildError):
        build_agent_catalog._load_faiss("unit-test")


def test_link_policy_cli_precedence(
    fake_embedding_backend: None, monkeypatch: SupportsSetEnv, repo_root: Path
) -> None:
    """CLI link policy options should override environment variables."""
    del fake_embedding_backend
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
