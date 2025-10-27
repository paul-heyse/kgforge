import pathlib

import numpy as np
import pytest

duckdb = pytest.importorskip("duckdb")

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
DENSE = str(FIXTURES / "dense_qwen3.parquet")

# Prefer the API adapter if present (uses FAISS+cuVS if available)
try:
    from kgfoundry.search_api.faiss_adapter import HAVE_FAISS, FaissAdapter

    HAVE_ADAPTER = True
except Exception:
    HAVE_ADAPTER = False
    HAVE_FAISS = False

# Fallback: vectorstore_faiss (provides CPU/GPU + brute-force fallback)
try:
    from kgfoundry.vectorstore_faiss.gpu import FaissGpuIndex as VSFaiss

    HAVE_VS = True
except Exception:
    HAVE_VS = False


@pytest.mark.skipif(not (HAVE_ADAPTER or HAVE_VS), reason="FAISS adapters not importable")
def test_dense_search_top1_is_self(tmp_path):
    # Load first vector and id from parquet
    con = duckdb.connect(database=":memory:")
    try:
        row = con.execute(
            f"""
            SELECT chunk_id, vector
            FROM read_parquet('{DENSE}', union_by_name=true)
            LIMIT 1
            """
        ).fetchone()
    finally:
        con.close()
    assert row is not None
    cid, vec = row[0], np.array(row[1], dtype=np.float32)
    # Normalize for cosine/IP search
    vec = vec / (np.linalg.norm(vec) + 1e-9)

    if HAVE_ADAPTER:
        adapter = FaissAdapter(db_path=DENSE, factory="OPQ64,IVF8192,PQ64", metric="ip")
        # Build in the current process to ensure a fresh index
        adapter.build()
        top = adapter.search(vec.reshape(1, -1), k=5)[0]
        top_ids = [cid_s for cid_s, _ in top]
        assert cid in top_ids
        # The exact self-match should be rank 1 in practice
        assert top_ids[0] == cid
    elif HAVE_VS:
        # Build an in-memory FAISS or fallback brute-force index
        # Load more vectors for a small index
        con = duckdb.connect(database=":memory:")
        try:
            rows = con.execute(
                f"""
                SELECT chunk_id, vector
                FROM read_parquet('{DENSE}', union_by_name=true)
                LIMIT 1000
                """
            ).fetchall()
        finally:
            con.close()
        ids = [r[0] for r in rows]
        X = np.stack([np.array(r[1], dtype=np.float32) for r in rows])
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

        idx = VSFaiss(factory="OPQ64,IVF8192,PQ64", gpu=False)  # CPU ok for unit
        idx.train(X)  # no-op in fallback
        idx.add(ids, X)
        res = idx.search(vec.reshape(1, -1), k=5)
        top_ids = [ids[i] for i, _ in res]
        assert cid in top_ids
        assert top_ids[0] == cid
