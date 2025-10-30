from pathlib import Path

from kgfoundry.agent_catalog.client import AgentCatalogClient

FIXTURE = Path("tests/fixtures/agent/catalog_sample.json")


def _client() -> AgentCatalogClient:
    return AgentCatalogClient.from_path(FIXTURE, repo_root=Path.cwd())


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


def test_sample_catalog_search_mrr() -> None:
    """Hybrid search should surface target symbols within the top results."""

    client = _client()
    queries = [("demo function", "demo.module.fn")]
    reciprocal_ranks: list[float] = []
    for query, expected in queries:
        results = client.search(query, k=5)
        rank = next((index + 1 for index, item in enumerate(results) if item.qname == expected), None)
        assert rank is not None, f"expected {expected} in results"
        reciprocal_ranks.append(1 / rank)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    assert mrr >= 1.0
