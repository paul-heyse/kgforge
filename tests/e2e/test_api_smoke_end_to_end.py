import os

from fastapi.testclient import TestClient
from kgfoundry.orchestration.fixture_flow import fixture_pipeline
from kgfoundry.search_api.app import app
from kgfoundry.search_client import KGFoundryClient


def test_fixture_and_api_smoke():
    os.environ["KGF_FIXTURE_ROOT"] = "/tmp/kgf_fixture"
    os.environ["KGF_FIXTURE_DB"] = "/tmp/kgf_fixture/catalog.duckdb"
    fixture_pipeline(root="/tmp/kgf_fixture", db_path="/tmp/kgf_fixture/catalog.duckdb")
    client = TestClient(app)
    cli = KGFoundryClient(base_url=str(client.base_url), http=client)
    assert cli.healthz()["status"] == "ok"
    res = cli.search("alignment", k=3)
    assert "results" in res and isinstance(res["results"], list)
