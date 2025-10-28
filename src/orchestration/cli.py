"""Cli utilities."""

from __future__ import annotations

import json
import os
from typing import Final

import numpy as np
import typer
from kgfoundry.embeddings_sparse.bm25 import get_bm25

from kgfoundry_common.navmap_types import NavMap
from vectorstore_faiss import gpu as faiss_gpu

__all__ = ["api", "e2e", "index_bm25", "index_faiss"]

__navmap__: Final[NavMap] = {
    "title": "orchestration.cli",
    "synopsis": "Prefect command-line entrypoints for orchestration flows",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@orchestration",
        "stability": "beta",
        "since": "0.1.0",
    },
    "symbols": {
        name: {
            "owner": "@orchestration",
            "stability": "beta",
            "since": "0.1.0",
        }
        for name in __all__
    },
}

app = typer.Typer(help="kgfoundry orchestration CLI")


# [nav:anchor index_bm25]
@app.command()
def index_bm25(
    chunks_parquet: str = typer.Argument(..., help="Path to Parquet/JSONL with chunks"),
    backend: str = typer.Option("lucene", help="lucene|pure"),
    index_dir: str = typer.Option("./_indices/bm25", help="Output index directory"),
) -> None:
    """Compute index bm25.

    Carry out the index bm25 operation.

    Parameters
    ----------
    chunks_parquet : str | None
        Description for ``chunks_parquet``.
    backend : str | None
        Description for ``backend``.
    index_dir : str | None
        Description for ``index_dir``.
    """
    
    
    
    
    
    
    
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


# [nav:anchor index_faiss]
@app.command()
def index_faiss(
    dense_vectors: str = typer.Argument(..., help="Path to dense vectors JSON (skeleton)"),
    index_path: str = typer.Option(
        "./_indices/faiss/shard_000.idx", help="Output index (CPU .idx)"
    ),
) -> None:
    """Compute index faiss.

    Carry out the index faiss operation.

    Parameters
    ----------
    dense_vectors : str | None
        Description for ``dense_vectors``.
    index_path : str | None
        Description for ``index_path``.
    """
    
    
    
    
    
    
    
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(dense_vectors, encoding="utf-8") as fh:
        vecs = json.load(fh)
    keys = [r["key"] for r in vecs]
    vectors = np.array([r["vector"] for r in vecs], dtype="float32")
    # Train and add
    vs = faiss_gpu.FaissGpuIndex()
    vs.train(vectors[: min(len(vectors), 10000)])  # small train set
    vs.add(keys, vectors)
    # Save CPU form when possible
    try:
        vs.save(index_path, None)
        typer.echo(f"FAISS index saved to {index_path}")
    except Exception as e:
        typer.echo(f"Saved fallback matrix (npz) due to {e!r}")
        vs.save(index_path, None)


# [nav:anchor api]
@app.command()
def api(port: int = 8080) -> None:
    """Compute api.

    Carry out the api operation.

    Parameters
    ----------
    port : int | None
        Description for ``port``.
    """
    
    
    
    
    
    
    
    import uvicorn

    uvicorn.run("search_api.app:app", host="0.0.0.0", port=port, reload=False)


# [nav:anchor e2e]
@app.command()
def e2e() -> None:
    """Compute e2e.

    Carry out the e2e operation.

    Raises
    ------
    typer.Exit
        Raised when validation fails.
    """
    
    
    
    
    
    
    
    try:
        from orchestration.flows import e2e_flow
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive messaging
        typer.echo(
            "Prefect is required for the e2e pipeline command. "
            "Install it via `pip install -e '.[gpu]'` or add `prefect` manually.",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    stages = e2e_flow()
    for step in stages:
        typer.echo(step)


if __name__ == "__main__":
    app()
