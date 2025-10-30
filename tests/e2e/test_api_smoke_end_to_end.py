import importlib
import os
import warnings
from pathlib import Path
from typing import cast

import duckdb
import pytest
import yaml
from fastapi.testclient import TestClient
from tests.conftest import HAS_GPU_STACK

from kgfoundry.embeddings_sparse.bm25 import PurePythonBM25
from kgfoundry.orchestration.fixture_flow import fixture_pipeline
from kgfoundry.search_client import KGFoundryClient
from kgfoundry.search_client.client import SupportsHttp

# Mark as GPU and skip automatically when the GPU stack is not available.
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not HAS_GPU_STACK,
        reason="GPU stack (extra 'gpu') not installed/available in this environment",
    ),
]

ROOT = Path(__file__).resolve().parents[2]


def test_fixture_and_api_smoke() -> None:
    root = Path("/tmp/kgf_fixture")
    db_path = root / "catalog" / "catalog.duckdb"
    os.environ["KGF_FIXTURE_ROOT"] = str(root)
    os.environ["KGF_FIXTURE_DB"] = str(db_path)

    cfg = {
        "system": {
            "parquet_root": str(root / "parquet"),
            "duckdb_path": str(db_path),
        },
        "search": {
            "sparse_backend": "pure",
            "k": 5,
            "dense_candidates": 50,
            "sparse_candidates": 50,
            "rrf_k": 20,
            "kg_boosts": {"direct": 0.1, "one_hop": 0.05},
        },
        "sparse_embedding": {
            "bm25": {"index_dir": str(root / "_indices" / "bm25"), "k1": 0.9, "b": 0.4},
            "splade": {"index_dir": str(root / "_indices" / "splade_impact"), "topk": 256},
        },
        "faiss": {"gpu": False, "cuvs": False, "index_factory": "Flat", "nprobe": 1},
    }
    config_path = root / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    (root / "_indices" / "bm25").mkdir(parents=True, exist_ok=True)
    (root / "_indices" / "splade_impact").mkdir(parents=True, exist_ok=True)
    (root / "_indices" / "faiss").mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["KGF_CONFIG"] = str(config_path)

    from kgfoundry.registry.migrate import apply as apply_migrations

    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    apply_migrations(str(db_path), str(ROOT / "registry" / "migrations"))
    fixture_pipeline(root=str(root), db_path=str(db_path))

    import kgfoundry.search_api.app as search_app

    search_app = importlib.reload(search_app)
    if isinstance(search_app.bm25, PurePythonBM25) and not search_app.bm25.docs:
        chunk_pattern = str(root / "parquet" / "chunks" / "**" / "*.parquet")
        con = duckdb.connect(database=":memory:")
        try:
            rows = con.execute(
                f"SELECT chunk_id, section, text FROM read_parquet('{chunk_pattern}', union_by_name=true)"
            ).fetchall()
        finally:
            con.close()
        docs = [
            (
                chunk_id,
                {"title": f"Title for {chunk_id}", "section": section or "", "body": text or ""},
            )
            for chunk_id, section, text in rows
        ]
        search_app.bm25.build(docs)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You should not use the 'timeout' argument with the TestClient",
            category=DeprecationWarning,
        )
        with TestClient(search_app.app) as client:
            cli = KGFoundryClient(base_url=str(client.base_url), http=cast(SupportsHttp, client))
            assert cli.healthz()["status"] == "ok"
            res = cli.search("alignment", k=3)
            assert "results" in res and isinstance(res["results"], list)
