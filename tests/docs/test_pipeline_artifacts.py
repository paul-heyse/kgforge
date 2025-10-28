"""Ensure documentation pipeline artifacts exist after generation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_test_map_exists_and_nonempty() -> None:
    path = Path("docs/_build/test_map.json")
    assert path.exists(), "run tools/docs/build_test_map.py to generate the test map"
    data = json.loads(path.read_text())
    assert isinstance(data, dict)


def test_observability_jsons_exist() -> None:
    base = Path("docs/_build")
    for name in ("metrics.json", "log_events.json", "traces.json"):
        artifact = base / name
        assert artifact.exists(), f"missing {name}; run tools/docs/scan_observability.py"


def test_graph_build_outputs_are_clean() -> None:
    pytest.importorskip("pydot")
    pytest.importorskip("networkx")
    pytest.importorskip("yaml")

    repo_root = Path(__file__).resolve().parents[2]
    graphs = repo_root / "docs" / "_build" / "graphs"

    if graphs.exists():
        shutil.rmtree(graphs)

    for pattern in ("classes_*.dot", "packages_*.dot"):
        for leftover in repo_root.glob(pattern):
            leftover.unlink()

    env = os.environ.copy()
    python_path = str(repo_root / "src")
    env["PYTHONPATH"] = (
        f"{python_path}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else python_path
    )

    def run_builder(fmt: str) -> None:
        cmd = [
            sys.executable,
            "tools/docs/build_graphs.py",
            "--packages",
            "kf_common",
            "--format",
            fmt,
            "--max-workers",
            "1",
            "--no-cache",
        ]
        subprocess.run(cmd, check=True, cwd=repo_root, env=env)

    if shutil.which("dot") is None:
        pytest.skip("graphviz 'dot' binary not available")

    for fmt in ("svg", "png"):
        run_builder(fmt)
        assert graphs.exists(), "graph outputs missing; run tools/docs/build_graphs.py"
        expected = graphs / f"kf_common-uml.{fmt}"
        assert expected.exists(), f"missing UML graph for {fmt}"
        assert any(graphs.glob(f"*-imports.{fmt}")), "no import graphs present"
        assert not any(graphs.glob("*.dot")), "DOT files should be cleaned from build output"
        assert not any(repo_root.glob("classes_*.dot")), "classes_*.dot leaked to repo root"
        assert not any(repo_root.glob("packages_*.dot")), "packages_*.dot leaked to repo root"

    assert (graphs / "kf_common-uml.svg").exists()
    assert (graphs / "kf_common-uml.png").exists()
