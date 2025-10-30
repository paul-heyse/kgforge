from pathlib import Path

from kgfoundry.agent_catalog.client import AgentCatalogClient


def _catalog_path() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "_build" / "agent_catalog.json"


def test_client_loads_catalog() -> None:
    client = AgentCatalogClient.from_path(_catalog_path())
    packages = client.list_packages()
    assert packages, "expected at least one package"
    module = packages[0].modules[0]
    symbol = module.symbols[0]
    callers = client.find_callers(symbol.symbol_id)
    assert isinstance(callers, list)
    search_results = client.search("catalog", k=5)
    assert search_results, "expected search results"
    anchors = client.open_anchor(symbol.symbol_id)
    assert "editor" in anchors
