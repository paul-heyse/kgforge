"""Semantic search adapter using FAISS GPU and DuckDB.

Implements semantic code search by embedding queries and searching
the FAISS index, then hydrating results from DuckDB.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import httpx
import numpy as np

from codeintel_rev.config.settings import Settings, load_settings
from codeintel_rev.io.duckdb_catalog import DuckDBCatalog
from codeintel_rev.io.faiss_manager import FAISSManager
from codeintel_rev.io.vllm_client import VLLMClient
from codeintel_rev.mcp_server.schemas import AnswerEnvelope, Finding, MethodInfo

SNIPPET_PREVIEW_CHARS = 500


async def semantic_search(query: str, limit: int = 20) -> AnswerEnvelope:
    """Perform semantic search using embeddings.

    Returns
    -------
    AnswerEnvelope
        Semantic search response payload.
    """
    return await asyncio.to_thread(_semantic_search_sync, query, limit)


def _semantic_search_sync(query: str, limit: int) -> AnswerEnvelope:
    start_time = perf_counter()
    settings = load_settings()
    index_path = Path(settings.paths.faiss_index)

    if not index_path.exists():
        return _make_envelope(
            findings=[],
            answer="Semantic search not available - index not built",
            confidence=0.0,
            limits=["FAISS index not found. Run indexing first."],
        )

    embedding, embed_error = _embed_query(VLLMClient(settings.vllm), query)
    if embedding is None or embed_error is not None:
        return _make_envelope(
            findings=[],
            answer=embed_error or "Embedding service unavailable",
            confidence=0.0,
            limits=["vLLM embedding service error"],
        )

    result_ids, result_scores, search_limits, search_error = _run_faiss_search(
        FAISSManager(
            index_path=index_path,
            vec_dim=settings.index.vec_dim,
            nlist=settings.index.faiss_nlist,
            use_cuvs=settings.index.use_cuvs,
        ),
        embedding,
        limit,
    )
    if search_error is not None:
        failure_limits = [*search_limits, "FAISS search error"]
        return _make_envelope(
            findings=[],
            answer=search_error,
            confidence=0.0,
            limits=failure_limits,
        )

    findings, hydrate_error = _hydrate_findings(settings, result_ids, result_scores)
    if hydrate_error is not None:
        failure_limits = [*search_limits, "DuckDB hydration error"]
        return _make_envelope(
            findings=findings,
            answer=f"Database query failed: {hydrate_error}",
            confidence=0.5,
            limits=failure_limits,
            method=_build_method(len(findings), limit, start_time),
        )

    success_answer = f"Found {len(findings)} semantically similar code chunks for: {query}"
    return _make_envelope(
        findings=findings,
        answer=success_answer,
        confidence=0.85 if findings else 0.0,
        limits=list(search_limits) or None,
        method=_build_method(len(findings), limit, start_time),
    )


def _embed_query(client: VLLMClient, query: str) -> tuple[np.ndarray | None, str | None]:
    """Embed query text and return a normalized vector and error message.

    Returns
    -------
    tuple[np.ndarray | None, str | None]
        Pair of (query_vector, error_message). Exactly one element will be ``None``.
    """
    try:
        vector = client.embed_single(query)
    except (RuntimeError, ValueError, httpx.HTTPError) as exc:
        return None, f"Embedding service unavailable: {exc}"

    array = np.array(vector, dtype=np.float32).reshape(1, -1)
    return array, None


def _run_faiss_search(
    faiss_mgr: FAISSManager,
    query_vector: np.ndarray,
    limit: int,
) -> tuple[list[int], list[float], list[str], str | None]:
    """Execute FAISS search and return result identifiers and scores.

    Returns
    -------
    tuple[list[int], list[float], list[str], str | None]
        Result IDs, scores, accumulated limit notices, and optional error message.
    """
    limits: list[str] = []
    try:
        faiss_mgr.load_cpu_index()
    except FileNotFoundError as exc:
        return [], [], limits, f"FAISS index not found: {exc}"
    except RuntimeError as exc:
        return [], [], limits, f"FAISS index load failed: {exc}"

    try:
        gpu_enabled = faiss_mgr.clone_to_gpu()
    except RuntimeError as exc:
        limits.append(str(exc))
        gpu_enabled = False

    if not gpu_enabled and faiss_mgr.gpu_disabled_reason:
        limits.append(faiss_mgr.gpu_disabled_reason)

    try:
        distances, ids = faiss_mgr.search(query_vector, k=limit)
    except RuntimeError as exc:
        return [], [], limits, f"FAISS search failed: {exc}"

    return ids[0].tolist(), distances[0].tolist(), limits, None


def _hydrate_findings(
    settings: Settings,
    chunk_ids: Sequence[int],
    scores: Sequence[float],
) -> tuple[list[Finding], str | None]:
    """Hydrate FAISS search results from DuckDB.

    Returns
    -------
    tuple[list[Finding], str | None]
        Findings constructed from the catalog and optional error message.
    """
    findings: list[Finding] = []
    try:
        with DuckDBCatalog(
            Path(settings.paths.duckdb_path),
            Path(settings.paths.vectors_dir),
        ) as catalog:
            for chunk_id, score in zip(chunk_ids, scores, strict=True):
                if chunk_id < 0:
                    continue
                chunk = catalog.get_chunk_by_id(int(chunk_id))
                if not chunk:
                    continue

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
                    "snippet": chunk["preview"][:SNIPPET_PREVIEW_CHARS],
                    "score": float(score),
                    "why": f"Semantic similarity: {score:.3f}",
                }
                findings.append(finding)
    except (RuntimeError, OSError) as exc:
        return findings, str(exc)

    return findings, None


def _build_method(findings_count: int, limit: int, start_time: float) -> MethodInfo:
    """Build method metadata for the response.

    Returns
    -------
    MethodInfo
        Retrieval metadata describing semantic search coverage.
    """
    elapsed_ms = int((perf_counter() - start_time) * 1000)
    return {
        "retrieval": ["semantic", "faiss"],
        "coverage": f"{findings_count}/{limit} results in {elapsed_ms}ms",
    }


def _make_envelope(
    *,
    findings: Sequence[Finding],
    answer: str,
    confidence: float,
    limits: Sequence[str] | None,
    method: MethodInfo | None = None,
) -> AnswerEnvelope:
    """Construct an AnswerEnvelope with optional metadata.

    Returns
    -------
    AnswerEnvelope
        Response payload ready for MCP clients.
    """
    envelope: AnswerEnvelope = {
        "answer": answer,
        "query_kind": "semantic",
        "findings": list(findings),
        "confidence": confidence,
    }

    if limits:
        envelope["limits"] = list(limits)

    if method is not None:
        envelope["method"] = method

    return envelope


__all__ = ["semantic_search"]
