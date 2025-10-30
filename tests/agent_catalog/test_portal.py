"""Snapshot-style checks for the rendered agent portal HTML."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter, sleep

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


def test_render_portal_uses_cached_module_cards(tmp_path: Path) -> None:
    """Module card HTML snippets should be reused across renders."""

    client = AgentCatalogClient.from_path(FIXTURE, repo_root=Path.cwd())
    output = tmp_path / "portal.html"
    start = perf_counter()
    render_portal(client, output)
    first_duration = perf_counter() - start
    cache_dir = output.parent / ".module_cache"
    cached_files = sorted(cache_dir.glob("*.html"))
    assert cached_files, "expected cached module cards"
    first_path = cached_files[0]
    before = first_path.stat().st_mtime_ns
    sleep(0.01)
    cached_start = perf_counter()
    render_portal(client, output)
    cached_duration = perf_counter() - cached_start
    after = first_path.stat().st_mtime_ns
    assert after == before, "cached card should not be rewritten when unchanged"
    assert cached_duration <= first_duration + 0.05


def test_render_portal_prunes_unused_cache_entries(tmp_path: Path) -> None:
    """Stale cache entries should be removed after rendering."""

    client = AgentCatalogClient.from_path(FIXTURE, repo_root=Path.cwd())
    output = tmp_path / "portal.html"
    render_portal(client, output)
    cache_dir = output.parent / ".module_cache"
    extra = cache_dir / "stale.html"
    extra.write_text("obsolete", encoding="utf-8")
    render_portal(client, output)
    assert not extra.exists(), "unused cache artefacts should be pruned"
