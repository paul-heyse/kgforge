"""Tests for the agent analytics artifact builder."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from tools.docs import build_agent_analytics

FIXTURE = Path("tests/fixtures/agent/catalog_sample.json")


def _create_fixture_paths(base: Path) -> None:
    """Populate placeholder files referenced by the sample catalog."""
    for relative in [
        "demo/module.py",
        "site/_build/html/autoapi/src/demo/module/index.html",
        "site/_build/json/autoapi/src/demo/module/index.fjson",
    ]:
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")


def test_build_analytics_reports_metrics(tmp_path: Path) -> None:
    """Analytics payload should include catalog metrics and zero errors when links exist."""
    _create_fixture_paths(tmp_path)
    args = Namespace(
        catalog=FIXTURE,
        output=tmp_path / "analytics.json",
        repo_root=tmp_path,
        link_sample=10,
    )
    payload = build_agent_analytics.build_analytics(args)
    assert payload["version"] == "1.0"
    assert payload["catalog"]["packages"] >= 1
    assert payload["portal"]["sessions"]["builds"] == 1
    assert payload["errors"]["broken_links"] == 0


def test_write_analytics_increments_build_counter(tmp_path: Path) -> None:
    """Persisted analytics should increment the portal build counter."""
    _create_fixture_paths(tmp_path)
    previous = {
        "version": "1.0",
        "portal": {"sessions": {"builds": 2, "unique_users": 5}},
        "errors": {"broken_links": 0},
    }
    output_path = tmp_path / "analytics.json"
    output_path.write_text(json.dumps(previous), encoding="utf-8")
    args = Namespace(
        catalog=FIXTURE,
        output=output_path,
        repo_root=tmp_path,
        link_sample=1,
    )
    build_agent_analytics.write_analytics(args)
    refreshed = json.loads(output_path.read_text(encoding="utf-8"))
    assert refreshed["portal"]["sessions"]["builds"] == 3
    assert refreshed["portal"]["sessions"]["unique_users"] == 5


def test_build_analytics_flags_missing_links(tmp_path: Path) -> None:
    """Missing files should be surfaced via the broken links summary."""
    args = Namespace(
        catalog=FIXTURE,
        output=tmp_path / "analytics.json",
        repo_root=tmp_path,
        link_sample=1,
    )
    payload = build_agent_analytics.build_analytics(args)
    assert payload["errors"]["broken_links"] >= 1
    assert payload["errors"]["details"], "expected broken link details"
