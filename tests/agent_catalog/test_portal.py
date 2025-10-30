"""Snapshot-style checks for the rendered agent portal HTML."""

from __future__ import annotations

from pathlib import Path

from tools.docs.render_agent_portal import render_portal

from kgfoundry.agent_catalog.client import AgentCatalogClient

FIXTURE = Path("tests/fixtures/agent/catalog_sample.json")


def test_render_portal_enriched_sections(tmp_path: Path) -> None:
    """The rendered portal should expose facets, breadcrumbs, and feedback UI."""
    client = AgentCatalogClient.from_path(FIXTURE, repo_root=Path.cwd())
    output = tmp_path / "portal.html"
    render_portal(client, output)
    content = output.read_text(encoding="utf-8")
    assert 'aria-label="Filters"' in content
    assert 'class="module-card"' in content
    assert 'aria-label="Breadcrumb"' in content
    assert "feedback-form" in content
    assert "Tutorials" in content
