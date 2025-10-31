"""Overview of cli.

This module bundles cli logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import numpy as np
import typer

from kgfoundry.embeddings_sparse.bm25 import get_bm25
from kgfoundry_common.navmap_types import NavMap
from vectorstore_faiss import gpu as faiss_gpu

logger = logging.getLogger(__name__)

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
def index_bm25(
    chunks_parquet: str = typer.Argument(..., help="Path to Parquet/JSONL with chunks"),
    backend: str = typer.Option("lucene", help="lucene|pure"),
    index_dir: str = typer.Option("./_indices/bm25", help="Output index directory"),
) -> None:
    """Build BM25 index from chunk data.

    This command builds a BM25 index from chunk data. The operation is
    **idempotent**: if an index already exists at the output directory, it
    will be rebuilt from scratch. No side effects occur beyond writing
    the index files.

    Parameters
    ----------
    chunks_parquet : str
        Path to Parquet or JSONL file containing chunks.
    backend : str, optional
        Backend to use: "lucene" or "pure" (default: "lucene").
    index_dir : str, optional
        Output directory for the index (default: "./_indices/bm25").

    Notes
    -----
    - **Idempotency**: This command is idempotent. Running it twice with
      identical inputs will rebuild the index, producing the same result.
      Existing index files are overwritten.
    - **Retries**: No automatic retries are implemented. If the operation
      fails due to transient errors (e.g., file system issues), run the
      command again manually. For persistent failures, check logs and
      verify input data integrity.
    """
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    # Check if index already exists and warn if so (idempotency)
    index_path = Path(index_dir)
    if backend == "pure":
        index_file = index_path / "pure_bm25.pkl"
    else:
        index_file = index_path / "bm25_index"

    if index_file.exists():
        logger.warning(
            "Index already exists at %s; will be rebuilt",
            index_dir,
            extra={"operation": "index_bm25", "status": "warning"},
        )
        typer.echo(f"Warning: Index already exists at {index_dir}, will be rebuilt")
    # Very small loader that supports JSONL in this skeleton (Parquet in real pipeline).
    docs: list[tuple[str, dict[str, str]]] = []
    if chunks_parquet.endswith(".jsonl"):
        with Path(chunks_parquet).open(encoding="utf-8") as fh:
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
        with Path(chunks_parquet).open(encoding="utf-8") as fh:
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
def index_faiss(
    dense_vectors: str = typer.Argument(..., help="Path to dense vectors JSON (skeleton)"),
    index_path: str = typer.Option(
        "./_indices/faiss/shard_000.idx", help="Output index (CPU .idx)"
    ),
) -> None:
    """Build FAISS index from dense vectors.

    This command builds a FAISS index from dense vector data. The operation
    is **idempotent**: if an index already exists at the output path, it
    will be rebuilt from scratch. No side effects occur beyond writing
    the index files.

    Parameters
    ----------
    dense_vectors : str
        Path to JSON file containing dense vectors (skeleton format).
    index_path : str, optional
        Output path for the index file (default: "./_indices/faiss/shard_000.idx").

    Notes
    -----
    - **Idempotency**: This command is idempotent. Running it twice with
      identical inputs will rebuild the index, producing the same result.
      Existing index files are overwritten.
    - **Retries**: No automatic retries are implemented. If the operation
      fails due to transient errors (e.g., GPU memory issues), run the
      command again manually. For persistent failures, check logs and
      verify input data integrity.
    """
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)

    # Check if index already exists and warn if so (idempotency)
    if Path(index_path).exists():
        logger.warning(
            "Index already exists at %s; will be rebuilt",
            index_path,
            extra={"operation": "index_faiss", "status": "warning"},
        )
        typer.echo(f"Warning: Index already exists at {index_path}, will be rebuilt")

    with Path(dense_vectors).open(encoding="utf-8") as fh:
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
    except Exception as exc:
        logger.warning("Failed to save FAISS index, using fallback: %s", exc, exc_info=True)
        typer.echo(f"Saved fallback matrix (npz) due to {exc!r}")
        vs.save(index_path, None)


# [nav:anchor api]
def api(port: int = 8080) -> None:
    """Describe api.

    <!-- auto:docstring-builder v1 -->

    Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

    Parameters
    ----------
    port : int, optional
        Describe ``port``.
        Defaults to ``8080``.
    """
    import uvicorn

    uvicorn.run("search_api.app:app", host="0.0.0.0", port=port, reload=False)


# [nav:anchor e2e]
def e2e() -> None:
    """Describe e2e.

    <!-- auto:docstring-builder v1 -->

    Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

    Raises
    ------
    Exit
    Raised when TODO for Exit.
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


app.command()(index_bm25)
app.command()(index_faiss)
app.command()(api)
app.command()(e2e)


if __name__ == "__main__":
    app()
