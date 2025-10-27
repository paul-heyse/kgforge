"""Module for orchestration.cli."""

from __future__ import annotations

import json
import os

import numpy as np
import typer
from kgforge.embeddings_sparse.bm25 import get_bm25
from kgforge.vectorstore_faiss.gpu import FaissGpuIndex

app = typer.Typer(help="KGForge orchestration CLI")


@app.command()
def index_bm25(
    chunks_parquet: str = typer.Argument(..., help="Path to Parquet/JSONL with chunks"),
    backend: str = typer.Option("lucene", help="lucene|pure"),
    index_dir: str = typer.Option("./_indices/bm25", help="Output index directory"),
) -> None:
    """Build a BM25 index from chunk fixtures (id, title, section, body)."""
    os.makedirs(index_dir, exist_ok=True)
    # Very small loader that supports JSONL in this skeleton (Parquet in real pipeline).
    docs: list[tuple[str, dict[str, str]]] = []
    if chunks_parquet.endswith(".jsonl"):
        with open(chunks_parquet, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                docs.append(
                    (
                        rec["chunk_id"],
                        {
                            "title": rec.get("title", ""),
                            "section": rec.get("section", ""),
                            "body": rec.get("text", ""),
                        },
                    )
                )
    else:
        # naive: expect a JSON file with list under skeleton; replace with Parquet
        # reader in implementation
        with open(chunks_parquet, encoding="utf-8") as fh:
            data = json.load(fh)
        for rec in data:
            docs.append(
                (
                    rec["chunk_id"],
                    {
                        "title": rec.get("title", ""),
                        "section": rec.get("section", ""),
                        "body": rec.get("text", ""),
                    },
                )
            )
    idx = get_bm25(backend, index_dir, k1=0.9, b=0.4)
    idx.build(docs)
    typer.echo(f"BM25 index built at {index_dir} using backend={backend} ({type(idx).__name__})")


@app.command()
def index_faiss(
    dense_vectors: str = typer.Argument(..., help="Path to dense vectors JSON (skeleton)"),
    index_path: str = typer.Option(
        "./_indices/faiss/shard_000.idx", help="Output index (CPU .idx)"
    ),
) -> None:
    """Train & build FAISS index from fixture dense vectors.

    In this skeleton we accept a JSON with entries: {key: str, vector: List[float]}.
    """
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(dense_vectors, encoding="utf-8") as fh:
        vecs = json.load(fh)
    keys = [r["key"] for r in vecs]
    X = np.array([r["vector"] for r in vecs], dtype="float32")
    # Train and add
    vs = FaissGpuIndex()
    vs.train(X[: min(len(X), 10000)])  # small train set
    vs.add(keys, X)
    # Save CPU form when possible
    try:
        vs.save(index_path, None)
        typer.echo(f"FAISS index saved to {index_path}")
    except Exception as e:
        typer.echo(f"Saved fallback matrix (npz) due to {e!r}")
        vs.save(index_path, None)


@app.command()
def api(port: int = 8080) -> None:
    """Run the FastAPI app."""
    import uvicorn

    uvicorn.run("search_api.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    app()
