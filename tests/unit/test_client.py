from typing import cast

from fastapi.testclient import TestClient
from kgfoundry.search_api.app import app
from kgfoundry.search_client import KGFoundryClient
from kgfoundry.search_client.client import SupportsHttp


def test_client_calls_api() -> None:
    client = TestClient(app)
    c = KGFoundryClient(base_url=str(client.base_url), http=cast(SupportsHttp, client))
    assert c.healthz()["status"] == "ok"
    res = c.search("test", k=3)
    assert "results" in res
