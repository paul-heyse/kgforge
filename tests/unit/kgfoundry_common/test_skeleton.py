"""Skeleton tests covering public modules and API health checks."""

from __future__ import annotations

import importlib
from collections.abc import Sequence

import pytest


def test_imports() -> None:
    """Ensure commonly referenced modules remain importable."""
    modules: Sequence[str] = [
        "kgfoundry_common.models",
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
    ]
    for module in modules:
        importlib.import_module(module)


def test_api_skeleton() -> None:
    """Verify the API skeleton serves the health endpoint when available."""
    pytest.importorskip("fastapi.testclient")
    testclient_module = importlib.import_module("fastapi.testclient")
    TestClient = testclient_module.TestClient
    from search_api.app import app

    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
