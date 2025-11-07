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

from codeintel_rev.io.faiss_manager import FAISSManager
from codeintel_rev.io.vllm_client import VLLMClient
from codeintel_rev.mcp_server.schemas import AnswerEnvelope, Finding, MethodInfo
from codeintel_rev.mcp_server.service_context import ServiceContext, get_service_context
from kgfoundry_common.errors import (
    EmbeddingError,
    KgFoundryError,
    KgFoundryErrorConfig,
    VectorSearchError,
)
from kgfoundry_common.logging import get_logger, with_fields
from kgfoundry_common.observability import MetricsProvider, observe_duration
from kgfoundry_common.problem_details import ProblemDetails

SNIPPET_PREVIEW_CHARS = 500
COMPONENT_NAME = "codeintel_mcp"
LOGGER = get_logger(__name__)
METRICS = MetricsProvider.default()


async def semantic_search(query: str, limit: int = 20) -> AnswerEnvelope:
    """Perform semantic search using embeddings.

    Parameters
    ----------
    query : str
        Search query text.
    limit : int, optional
        Maximum number of results to return. Defaults to 20.

    Returns
    -------
    AnswerEnvelope
        Semantic search response payload.
    """
    return await asyncio.to_thread(_semantic_search_sync, query, limit)


def _semantic_search_sync(query: str, limit: int) -> AnswerEnvelope:
    start_time = perf_counter()
    with observe_duration(METRICS, "semantic_search", component=COMPONENT_NAME) as observation:
        context = get_service_context()

        requested_limit = limit
        max_results = max(1, context.settings.limits.max_results)
        effective_limit = max(1, min(requested_limit, max_results))
        truncation_messages: list[str] = []
        if requested_limit <= 0:
            truncation_messages.append(
                f"Requested limit {requested_limit} is not positive; using minimum of 1."
            )
        if requested_limit > max_results:
            truncation_messages.append(
                f"Requested limit {requested_limit} exceeds max_results {max_results}; truncating to {max_results}."
            )

        ready, faiss_limits, faiss_error = context.ensure_faiss_ready()
        limits_metadata = [*faiss_limits, *truncation_messages]
        if not ready:
            failure_limits = [*limits_metadata, "FAISS index not available"]
            error = VectorSearchError(
                faiss_error or "Semantic search not available - index not built",
                config=KgFoundryErrorConfig(
                    context={"faiss_index": str(context.settings.paths.faiss_index)}
                ),
            )
            observation.mark_error()
            return _error_envelope(error, limits=failure_limits)

        embedding, embed_error = _embed_query(context.vllm_client, query)
        if embedding is None or embed_error is not None:
            error = EmbeddingError(
                embed_error or "Embedding service unavailable",
                config=KgFoundryErrorConfig(context={"vllm_url": context.settings.vllm.base_url}),
            )
            observation.mark_error()
            return _error_envelope(
                error,
                limits=[*limits_metadata, "vLLM embedding service error"],
            )

        result_ids, result_scores, search_exc = _run_faiss_search(
            context.faiss_manager,
            embedding,
            effective_limit,
        )
        if search_exc is not None:
            error = VectorSearchError(
                str(search_exc),
                config=KgFoundryErrorConfig(
                    cause=search_exc,
                    context={"faiss_index": str(context.settings.paths.faiss_index)},
                ),
            )
            observation.mark_error()
            return _error_envelope(
                error,
                limits=[*limits_metadata, "FAISS search error"],
            )

        findings, hydrate_exc = _hydrate_findings(context, result_ids, result_scores)
        if hydrate_exc is not None:
            failure_limits = [*limits_metadata, "DuckDB hydration error"]
            error = VectorSearchError(
                str(hydrate_exc),
                config=KgFoundryErrorConfig(
                    cause=hydrate_exc,
                    context={
                        "duckdb_path": str(context.settings.paths.duckdb_path),
                        "vectors_dir": str(context.settings.paths.vectors_dir),
                    },
                ),
            )
            observation.mark_error()
            return _error_envelope(
                error,
                limits=failure_limits,
                method=_build_method(
                    len(findings),
                    requested_limit,
                    effective_limit,
                    start_time,
                ),
            )

        observation.mark_success()
        extras: dict[str, object] = {}
        if limits_metadata:
            extras["limits"] = limits_metadata
        extras["method"] = _build_method(
            len(findings),
            requested_limit,
            effective_limit,
            start_time,
        )
        success_answer = f"Found {len(findings)} semantically similar code chunks for: {query}"
        return _make_envelope(
            findings=findings,
            answer=success_answer,
            confidence=0.85 if findings else 0.0,
            extras=extras,
        )


def _embed_query(client: VLLMClient, query: str) -> tuple[np.ndarray | None, str | None]:
    """Embed query text and return a normalized vector and error message.

    Parameters
    ----------
    client : VLLMClient
        vLLM client for generating embeddings.
    query : str
        Query text to embed.

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
) -> tuple[list[int], list[float], Exception | None]:
    """Execute FAISS search and return result identifiers and scores.

    Returns
    -------
    tuple[list[int], list[float], Exception | None]
        Tuple of (chunk_ids, distances, error). ``error`` is ``None`` when the
        search succeeds; otherwise it contains the triggering exception.
    """
    try:
        distances, ids = faiss_mgr.search(query_vector, k=limit)
    except RuntimeError as exc:
        return [], [], exc

    return ids[0].tolist(), distances[0].tolist(), None


def _hydrate_findings(
    context: ServiceContext,
    chunk_ids: Sequence[int],
    scores: Sequence[float],
) -> tuple[list[Finding], Exception | None]:
    """Hydrate FAISS search results from DuckDB.

    Parameters
    ----------
    context : ServiceContext
        Service context for accessing DuckDB catalog.
    chunk_ids : Sequence[int]
        Chunk identifiers from FAISS search.
    scores : Sequence[float]
        Similarity scores aligned with chunk_ids.

    Returns
    -------
    tuple[list[Finding], Exception | None]
        Findings constructed from the catalog and optional hydration exception.
    """
    findings: list[Finding] = []
    try:
        with context.open_catalog() as catalog:
            valid_ids = [int(chunk_id) for chunk_id in chunk_ids if chunk_id >= 0]
            if not valid_ids:
                return [], None

            records = catalog.query_by_ids(valid_ids)
            chunk_by_id = {int(record["id"]): record for record in records if "id" in record}

            for chunk_id, score in zip(chunk_ids, scores, strict=True):
                if chunk_id < 0:
                    continue
                chunk = chunk_by_id.get(int(chunk_id))
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
        return findings, exc

    return findings, None


def _error_envelope(
    error: KgFoundryError,
    *,
    limits: Sequence[str] | None,
    method: MethodInfo | None = None,
) -> AnswerEnvelope:
    """Build an error envelope including Problem Details metadata.

    Parameters
    ----------
    error : KgFoundryError
        Error to convert to envelope.
    limits : Sequence[str] | None
        Search limitations or warnings.
    method : MethodInfo | None, optional
        Retrieval method metadata. Defaults to None.

    Returns
    -------
    AnswerEnvelope
        Error envelope with Problem Details.
    """
    problem: ProblemDetails = error.to_problem_details(instance="semantic_search")
    with with_fields(
        LOGGER,
        component=COMPONENT_NAME,
        operation="semantic_search",
        error_code=error.code.value,
    ) as adapter:
        adapter.log(error.log_level, error.message, extra={"context": error.context})

    extras: dict[str, object] = {"problem": problem}
    if limits:
        extras["limits"] = list(limits)
    if method is not None:
        extras["method"] = method
    return _make_envelope(
        findings=[],
        answer=error.message,
        confidence=0.0,
        extras=extras,
    )


def _build_method(
    findings_count: int,
    requested_limit: int,
    effective_limit: int,
    start_time: float,
) -> MethodInfo:
    """Build method metadata for the response.

    Parameters
    ----------
    findings_count : int
        Number of findings returned.
    requested_limit : int
        Requested result limit.
    effective_limit : int
        Effective limit applied after clamping.
    start_time : float
        Search start time (monotonic clock).

    Returns
    -------
    MethodInfo
        Retrieval metadata describing semantic search coverage.
    """
    elapsed_ms = int((perf_counter() - start_time) * 1000)
    coverage = f"{findings_count}/{effective_limit} results in {elapsed_ms}ms"
    if requested_limit != effective_limit:
        coverage = f"{coverage} (requested {requested_limit})"
    return {
        "retrieval": ["semantic", "faiss"],
        "coverage": coverage,
    }


def _make_envelope(
    *,
    findings: Sequence[Finding],
    answer: str,
    confidence: float,
    extras: dict[str, object] | None = None,
) -> AnswerEnvelope:
    """Construct an AnswerEnvelope with optional metadata.

    Parameters
    ----------
    findings : Sequence[Finding]
        Search findings.
    answer : str
        Answer text.
    confidence : float
        Confidence score (0.0 to 1.0).
    extras : dict[str, object] | None, optional
        Additional envelope fields (for example ``limits``, ``method``, or
        ``problem`` metadata). Defaults to None.

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

    if extras:
        envelope.update(extras)

    return envelope


__all__ = ["semantic_search"]
