"""Tests for documentation graph helpers."""

from __future__ import annotations

from pathlib import Path
import shutil

import networkx as nx
import pytest

from tools.docs.build_graphs import collapse_to_packages, style_and_render


def _write_dot(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "digraph G {",
                '  "pkg_alpha.module";',
                '  "pkg_beta.module";',
                '  "pkg_isolated.module";',
                '  "pkg_alpha.module" -> "pkg_beta.module";',
                "}",
            ]
        )
    )


def test_collapse_to_packages_preserves_isolated_nodes(tmp_path: Path) -> None:
    """Ensure collapsing retains packages even when they have no edges."""

    dot_path = tmp_path / "graph.dot"
    _write_dot(dot_path)

    collapsed = collapse_to_packages(dot_path)

    assert set(collapsed.nodes) == {"pkg_alpha", "pkg_beta", "pkg_isolated"}
    assert collapsed.has_edge("pkg_alpha", "pkg_beta")


def test_style_and_render_writes_isolated_nodes(tmp_path: Path) -> None:
    """Rendering includes edge-free packages in the SVG output."""

    if shutil.which("dot") is None:
        pytest.skip("graphviz 'dot' binary is required for rendering tests")

    g = nx.DiGraph()
    g.add_nodes_from(["pkg_alpha", "pkg_beta", "pkg_isolated"])
    g.add_edge("pkg_alpha", "pkg_beta", weight=1)

    analysis = {
        "centrality": {node: 0.0 for node in g.nodes},
        "cycles": [],
        "layer_violations": [],
        "cycle_enumeration_skipped": False,
    }

    out_svg = tmp_path / "graph.svg"
    style_and_render(g, {"packages": {}}, analysis, out_svg)

    svg = out_svg.read_text(encoding="utf-8")
    for name in g.nodes:
        assert name in svg
"""Tests for tools.docs.build_graphs caching helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import tools.docs.build_graphs as build_graphs


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_dirty_worktree_invalidates_cache(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    src_pkg = repo / "src" / "sample_pkg"
    src_pkg.mkdir(parents=True)
    package_file = src_pkg / "__init__.py"
    package_file.write_text("CONST = 1\n", encoding="utf-8")

    out_dir = repo / "docs" / "_build" / "graphs"
    out_dir.mkdir(parents=True)
    cache_dir = repo / ".cache" / "graphs"
    cache_dir.mkdir(parents=True)

    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")

    monkeypatch.setattr(build_graphs, "ROOT", repo)
    monkeypatch.setattr(build_graphs, "SRC", repo / "src")
    monkeypatch.setattr(build_graphs, "OUT", out_dir)

    counters = {"pydeps": 0, "pyrev": 0}

    def fake_pydeps(pkg: str, imports_out: Path, excludes: list[str], max_bacon: int, fmt: str) -> None:
        counters["pydeps"] += 1
        imports_out.write_text(f"deps {counters['pydeps']}\n", encoding="utf-8")

    def fake_pyreverse(pkg: str, out_dir: Path, fmt: str) -> None:
        counters["pyrev"] += 1
        (out_dir / f"{pkg}-uml.{fmt}").write_text(
            f"uml {counters['pyrev']}\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(build_graphs, "build_pydeps_for_package", fake_pydeps)
    monkeypatch.setattr(build_graphs, "build_pyreverse_for_package", fake_pyreverse)

    first = build_graphs.build_one_package(
        "sample_pkg",
        "svg",
        [],
        4,
        cache_dir,
        use_cache=True,
        verbose=False,
    )
    assert first == ("sample_pkg", False, True, True)
    assert counters == {"pydeps": 1, "pyrev": 1}

    second = build_graphs.build_one_package(
        "sample_pkg",
        "svg",
        [],
        4,
        cache_dir,
        use_cache=True,
        verbose=False,
    )
    assert second == ("sample_pkg", True, True, True)
    assert counters == {"pydeps": 1, "pyrev": 1}

    package_file.write_text("CONST = 2\n", encoding="utf-8")

    third = build_graphs.build_one_package(
        "sample_pkg",
        "svg",
        [],
        4,
        cache_dir,
        use_cache=True,
        verbose=False,
    )
    assert third == ("sample_pkg", False, True, True)
    assert counters == {"pydeps": 2, "pyrev": 2}

    fourth = build_graphs.build_one_package(
        "sample_pkg",
        "svg",
        [],
        4,
        cache_dir,
        use_cache=True,
        verbose=False,
    )
    assert fourth == ("sample_pkg", True, True, True)
    assert counters == {"pydeps": 2, "pyrev": 2}
