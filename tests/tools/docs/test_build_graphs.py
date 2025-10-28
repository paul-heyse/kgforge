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
