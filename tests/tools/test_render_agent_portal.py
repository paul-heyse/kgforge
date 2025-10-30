from pathlib import Path

from tools.docs import render_agent_portal

from kgfoundry.agent_catalog.client import AgentCatalogClient

CATALOG_PATH = Path(__file__).resolve().parents[2] / "docs" / "_build" / "agent_catalog.json"
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_render_portal(tmp_path: Path) -> None:
    output_path = tmp_path / "portal" / "index.html"
    client = AgentCatalogClient.from_path(CATALOG_PATH, repo_root=REPO_ROOT)
    render_agent_portal.render_portal(client, output_path)
    html = output_path.read_text(encoding="utf-8")
    assert "Agent Portal" in html
    assert "module" in html
