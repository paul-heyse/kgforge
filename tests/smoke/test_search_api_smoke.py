
import os
import re
import pathlib
import pytest

duckdb = pytest.importorskip("duckdb")
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
CHUNKS = str(FIXTURES / "chunks.parquet")

# Import app after optional env wiring (app uses fixtures by default)
from kgforge.search_api.app import app

def pick_query_token():
    con = duckdb.connect(database=':memory:')
    try:
        # sample a few rows for a cheap token pick
        rows = con.execute(f"SELECT text FROM read_parquet('{CHUNKS}', union_by_name=true) LIMIT 50").fetchall()
    finally:
        con.close()
    tok_re = re.compile(r"[A-Za-z]{4,}")
    freq = {}
    for (t,) in rows:
        for w in tok_re.findall(t or ""):
            w = w.lower()
            freq[w] = freq.get(w, 0) + 1
    return max(freq, key=freq.get)

@pytest.mark.smoke
def test_search_endpoint_smoke():
    client = TestClient(app)
    q = pick_query_token()
    resp = client.post("/search", json={"query": q, "k": 5, "explain": False})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data and isinstance(data["results"], list)
    assert len(data["results"]) > 0
    # Check the schema of one result
    r0 = data["results"][0]
    for key in ("doc_id","chunk_id","title","section","score","signals","spans","concepts"):
        assert key in r0
