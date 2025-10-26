
import importlib

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
    from fastapi.testclient import TestClient
    from search_api.app import app
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
