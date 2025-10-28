"""Ensure documentation pipeline artifacts exist after generation."""

from __future__ import annotations

import json
from pathlib import Path


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


def test_graph_svgs_exist() -> None:
    graphs = Path("docs/_build/graphs")
    assert graphs.exists(), "graph outputs missing; run tools/docs/build_graphs.py"
    assert any(graphs.glob("*-imports.svg")), "no import graphs present"
    assert any(graphs.glob("*-uml.svg")), "no UML graphs present"
