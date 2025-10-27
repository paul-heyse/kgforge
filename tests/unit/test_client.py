from fastapi.testclient import TestClient
from kgforge.search_api.app import app
from kgforge.search_client import KGForgeClient


def test_client_calls_api():
    client = TestClient(app)
    c = KGForgeClient(base_url=str(client.base_url))
    assert c.healthz()["status"] == "ok"
    res = c.search("test", k=3)
    assert "results" in res
