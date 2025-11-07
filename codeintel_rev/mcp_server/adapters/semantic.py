"""Semantic search adapter using FAISS GPU and DuckDB.

Implements semantic code search by embedding queries and searching
the FAISS index, then hydrating results from DuckDB.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from codeintel_rev.config.settings import load_settings
from codeintel_rev.io.duckdb_catalog import DuckDBCatalog
from codeintel_rev.io.faiss_manager import FAISSManager
from codeintel_rev.io.vllm_client import VLLMClient
from codeintel_rev.mcp_server.schemas import AnswerEnvelope, Finding, MethodInfo


async def semantic_search(
    query: str,
    limit: int = 20,
) -> AnswerEnvelope:
    """Perform semantic search using embeddings.

    Parameters
    ----------
    query : str
        Natural language or code query.
    limit : int
        Maximum number of results to return.

    Returns
    -------
    AnswerEnvelope
        Search results with findings and metadata.

    Examples
    --------
    >>> result = await semantic_search("parse JSON configuration")
    >>> len(result["findings"]) <= 20
    True
    """
    settings = load_settings()
    start_time = time.time()

    # Initialize clients
    vllm_client = VLLMClient(settings.vllm)
    faiss_mgr = FAISSManager(
        index_path=Path(settings.paths.faiss_index),
        vec_dim=settings.index.vec_dim,
        nlist=settings.index.faiss_nlist,
        use_cuvs=settings.index.use_cuvs,
    )

    # Check if index exists
    if not Path(settings.paths.faiss_index).exists():
        return {
            "answer": "Semantic search not available - index not built",
            "query_kind": "semantic",
            "findings": [],
            "limits": ["FAISS index not found. Run indexing first."],
            "confidence": 0.0,
        }

    # Embed query
    try:
        query_embedding = await asyncio.to_thread(
            vllm_client.embed_single,
            query,
        )
    except Exception as e:
        return {
            "answer": f"Embedding service unavailable: {e}",
            "query_kind": "semantic",
            "findings": [],
            "limits": ["vLLM embedding service error"],
            "confidence": 0.0,
        }

    # Load FAISS index and search
    try:
        faiss_mgr.load_cpu_index()
        faiss_mgr.clone_to_gpu()

        # Search
        import numpy as np

        query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, ids = faiss_mgr.search(query_vec, k=limit)

        # Flatten results (single query)
        result_ids = ids[0].tolist()
        result_scores = distances[0].tolist()

    except Exception as e:
        return {
            "answer": f"FAISS search failed: {e}",
            "query_kind": "semantic",
            "findings": [],
            "limits": ["FAISS search error"],
            "confidence": 0.0,
        }

    # Hydrate from DuckDB
    findings: list[Finding] = []
    try:
        with DuckDBCatalog(
            Path(settings.paths.duckdb_path),
            Path(settings.paths.vectors_dir),
        ) as catalog:
            for chunk_id, score in zip(result_ids, result_scores, strict=True):
                # Skip invalid IDs
                if chunk_id < 0:
                    continue

                # Fetch chunk metadata
                chunk = catalog.get_chunk_by_id(int(chunk_id))
                if not chunk:
                    continue

                # Create finding
                finding: Finding = {
                    "type": "usage",
                    "title": f"{Path(chunk['uri']).name} (score: {score:.3f})",
                    "location": {
                        "uri": chunk["uri"],
                        "start_line": chunk["start_line"],
                        "start_column": 0,
                        "end_line": chunk["end_line"],
                        "end_column": 0,
                    },
                    "snippet": chunk["preview"][:500],
                    "score": float(score),
                    "why": f"Semantic similarity: {score:.3f}",
                }
                findings.append(finding)

    except Exception as e:
        return {
            "answer": f"Database query failed: {e}",
            "query_kind": "semantic",
            "findings": findings,  # Return what we got
            "limits": ["DuckDB hydration error"],
            "confidence": 0.5,
        }

    # Build response
    elapsed_ms = int((time.time() - start_time) * 1000)

    method: MethodInfo = {
        "retrieval": ["semantic", "faiss"],
        "coverage": f"{len(findings)}/{limit} results in {elapsed_ms}ms",
    }

    answer = f"Found {len(findings)} semantically similar code chunks for: {query}"

    return {
        "answer": answer,
        "query_kind": "semantic",
        "method": method,
        "findings": findings,
        "confidence": 0.85 if findings else 0.0,
    }


__all__ = ["semantic_search"]
