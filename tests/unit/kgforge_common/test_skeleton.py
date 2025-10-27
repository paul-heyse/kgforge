import importlib

import pytest

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover
    TestClient = None  # type: ignore[assignment]


def test_imports():
    for mod in [
        "kgforge_common.models",
        "download.harvester",
        "docling.vlm",
        "docling.hybrid",
        "embeddings_dense.qwen3",
        "embeddings_sparse.splade",
        "embeddings_sparse.bm25",
        "vectorstore_faiss.gpu",
        "ontology.loader",
        "linking.linker",
        "kg_builder.neo4j_store",
        "search_api.app",
        "orchestration.flows",
    ]:
        importlib.import_module(mod)


def test_api_skeleton():
    if TestClient is None:  # pragma: no cover - optional dependency missing
        pytest.skip("fastapi is required for API skeleton test")

    from search_api.app import app

    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
