
import os
import re
import math
import json
import pathlib
import pytest

duckdb = pytest.importorskip("duckdb")

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
CHUNKS = str(FIXTURES / "chunks.parquet")

# Prefer the dedicated BM25 index used by the API if available
try:
    from kgforge.search_api.bm25_index import BM25Index
    HAVE_API_BM25 = True
except Exception:  # module missing or incomplete
    HAVE_API_BM25 = False

try:
    from kgforge.embeddings_sparse.bm25 import PurePythonBM25
    HAVE_PURE = True
except Exception:
    HAVE_PURE = False

def most_frequent_token(limit=10000):
    con = duckdb.connect(database=':memory:')
    try:
        df = con.execute(f"""
            SELECT text FROM read_parquet('{CHUNKS}', union_by_name=true) LIMIT {limit}
        """).fetchdf()
    finally:
        con.close()
    tok_re = re.compile(r"[A-Za-z]{4,}")
    freq = {}
    for t in df['text']:
        for w in tok_re.findall(t or ""):
            w = w.lower()
            freq[w] = freq.get(w, 0) + 1
    # pick a reasonably common token
    token, _ = max(freq.items(), key=lambda kv: kv[1])
    return token

@pytest.mark.skipif(not (HAVE_API_BM25 or HAVE_PURE), reason="BM25 implementations not importable")
def test_bm25_build_and_search_from_fixtures(tmp_path):
    q = most_frequent_token()

    if HAVE_API_BM25:
        idx = BM25Index.from_parquet(CHUNKS, k1=0.9, b=0.4)
        hits = idx.search(q, k=10)
        assert isinstance(hits, list) and len(hits) > 0
        # ensure scores are sorted
        scores = [s for _, s in hits]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        # ensure returned chunk_ids exist in the dataset
        con = duckdb.connect(database=':memory:')
        try:
            ids = set(r[0] for r in con.execute(f"SELECT chunk_id FROM read_parquet('{CHUNKS}', union_by_name=true)").fetchall())
        finally:
            con.close()
        for cid, _ in hits[:5]:
            assert cid in ids
    elif HAVE_PURE:
        # Fallback pure-python BM25 over fixtures (body only)
        con = duckdb.connect(database=':memory:')
        try:
            rows = con.execute(f"SELECT chunk_id, section, text FROM read_parquet('{CHUNKS}', union_by_name=true)").fetchall()
        finally:
            con.close()
        docs = [(cid, {'body': txt or '', 'section': sec or '', 'title': ''}) for cid, sec, txt in rows]
        bm = PurePythonBM25(index_dir=str(tmp_path))
        bm.build(docs)
        res = bm.search(q, k=10)
        assert len(res) > 0
        # monotonic scores
        scores = [s for _, s in res]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
