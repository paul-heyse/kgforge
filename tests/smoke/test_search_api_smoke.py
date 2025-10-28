import pathlib
import re

import pytest

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pytest.skip("fastapi is required for smoke tests", allow_module_level=True)

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from kgfoundry.embeddings_sparse.bm25 import PurePythonBM25

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
CHUNKS = str(FIXTURES / "chunks.parquet")


def pick_query_token() -> str:
    con = duckdb.connect(database=":memory:")
    try:
        # sample a few rows for a cheap token pick
        rows = con.execute(
            f"SELECT text FROM read_parquet('{CHUNKS}', union_by_name=true) LIMIT 50"
        ).fetchall()
    finally:
        con.close()
    tok_re = re.compile(r"[A-Za-z]{4,}")
    freq: dict[str, int] = {}
    for (t,) in rows:
        for w in tok_re.findall(t or ""):
            w = w.lower()
            freq[w] = freq.get(w, 0) + 1
    return max(freq, key=lambda token: freq[token])


@pytest.mark.smoke
def test_search_endpoint_smoke() -> None:
    # Import app after optional env wiring (app uses fixtures by default).
    import importlib
    import kgfoundry.search_api.app as search_app_module

    search_app = importlib.reload(search_app_module)

    search_app_module.bm25 = PurePythonBM25(
        index_dir=str(FIXTURES / "_indices" / "bm25"),
        k1=0.9,
        b=0.4,
    )
    if isinstance(search_app.bm25, PurePythonBM25) and not search_app.bm25.docs:
        con = duckdb.connect(database=":memory:")
        try:
            rows = con.execute(
                f"SELECT chunk_id, section, text FROM read_parquet('{CHUNKS}', union_by_name=true)"
            ).fetchall()
        finally:
            con.close()
        docs = [
            (
                chunk_id,
                {
                    "title": f"Title for {chunk_id}",
                    "section": section or "",
                    "body": text or "",
                },
            )
            for chunk_id, section, text in rows
        ]
        search_app.bm25.build(docs)

    client = TestClient(search_app.app)
    q = pick_query_token()
    resp = client.post("/search", json={"query": q, "k": 5, "explain": False})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data and isinstance(data["results"], list)
    assert len(data["results"]) > 0
    # Check the schema of one result
    r0 = data["results"][0]
    for key in ("doc_id", "chunk_id", "title", "section", "score", "signals", "spans", "concepts"):
        assert key in r0
