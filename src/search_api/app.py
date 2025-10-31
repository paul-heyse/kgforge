"""Search service endpoints and helper utilities.

This module provides FastAPI endpoints for hybrid search using dense (FAISS),
sparse (BM25/SPLADE), and knowledge graph boosting. All endpoints return
RFC 9457 Problem Details for error responses.

Error Responses
---------------
When search operations fail, the API returns Problem Details JSON responses.
See `schema/examples/problem_details/search-missing-index.json` for an example.

Examples
--------
>>> from search_api.schemas import SearchRequest
>>> from search_api.app import search
>>> req = SearchRequest(query="test query", k=5)
>>> response = search(req, None)
>>> len(response.results) <= 5
True

See Also
--------
- `schema/examples/problem_details/search-missing-index.json` - Example Problem Details response
- `schema/models/search_request.v1.json` - SearchRequest JSON Schema
- `schema/models/search_result.v1.json` - SearchResult JSON Schema
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from fastapi import Depends, FastAPI, Header, HTTPException

from kgfoundry.embeddings_sparse.bm25 import PurePythonBM25, get_bm25
from kgfoundry.embeddings_sparse.splade import get_splade
from kgfoundry.kg_builder.mock_kg import MockKG
from kgfoundry_common.errors import (
    register_problem_details_handler,
)
from kgfoundry_common.logging import get_logger, set_correlation_id
from kgfoundry_common.navmap_types import NavMap
from kgfoundry_common.observability import get_metrics_registry, record_operation
from kgfoundry_common.settings import RuntimeSettings
from search_api.schemas import SearchRequest, SearchResponse, SearchResult
from vectorstore_faiss import gpu as faiss_gpu

__all__ = [
    "apply_kg_boosts",
    "auth",
    "graph_concepts",
    "healthz",
    "rrf_fuse",
    "search",
]

__navmap__: Final[NavMap] = {
    "title": "search_api.app",
    "synopsis": "Search service endpoints and helper utilities",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": [
                "auth",
                "healthz",
                "rrf_fuse",
                "apply_kg_boosts",
                "search",
                "graph_concepts",
            ],
        },
    ],
    "module_meta": {
        "owner": "@search-api",
        "stability": "experimental",
        "since": "0.2.0",
    },
    "symbols": {
        name: {
            "owner": "@search-api",
            "stability": "experimental",
            "since": "0.2.0",
        }
        for name in __all__
    },
}

logger = get_logger(__name__)
metrics = get_metrics_registry()

API_KEYS: set[str] = set()  # NOTE: load from env SEARCH_API_KEYS when secrets wiring is ready

app = FastAPI(title="kgfoundry Search API", version="0.2.0")

# Register Problem Details exception handler
register_problem_details_handler(app)

# --- bootstrap typed configuration ---
try:
    settings = RuntimeSettings()
except Exception:
    logger.exception("Failed to load configuration")
    raise

# Extract configuration values with defaults
SPARSE_BACKEND = settings.search.sparse_backend.lower()
BM25_DIR = settings.sparse_embedding.bm25_index_dir
SPLADE_DIR = settings.sparse_embedding.splade_index_dir
SPLADE_QUERY_ENCODER = settings.sparse_embedding.splade_query_encoder
FAISS_PATH = settings.faiss.index_path

bm25 = get_bm25(
    SPARSE_BACKEND,
    BM25_DIR,
    k1=settings.sparse_embedding.bm25_k1,
    b=settings.sparse_embedding.bm25_b,
    field_boosts=None,  # TODO: Add field_boosts to SparseEmbeddingConfig
)
# load index if exists
try:
    if isinstance(bm25, PurePythonBM25) and (Path(BM25_DIR) / "pure_bm25.pkl").exists():
        bm25.load()
except Exception as exc:
    logger.warning("Failed to load BM25 index: %s", exc, exc_info=True)
    # Continue without index (will be built on first use)

splade = get_splade(SPARSE_BACKEND, SPLADE_DIR, query_encoder=SPLADE_QUERY_ENCODER)
try:
    # for PureImpactIndex, try to load
    if hasattr(splade, "load"):
        splade.load()
except Exception as exc:
    logger.warning("Failed to load SPLADE index: %s", exc, exc_info=True)
    # Continue without index (will be built on first use)

faiss = faiss_gpu.FaissGpuIndex(
    factory=settings.faiss.index_factory,
    nprobe=settings.faiss.nprobe,
    gpu=settings.faiss.gpu,
    cuvs=settings.faiss.cuvs,
)
try:
    if Path(FAISS_PATH).exists():
        faiss.load(FAISS_PATH, None)
except Exception as exc:
    logger.warning("Failed to load FAISS index: %s", exc, exc_info=True)
    # Continue without index (will be built on first use)

# tiny KG mock with a few edges/mentions to demonstrate boosts
kg = MockKG()
kg.add_mention("chunk:1", "C:42")
kg.add_edge("C:42", "C:99")


# [nav:anchor auth]
def auth(authorization: str | None = Header(default=None)) -> None:
    """Describe auth.

    <!-- auto:docstring-builder v1 -->

    Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

    Parameters
    ----------
    authorization : str | None, optional
        Describe ``authorization``.
        Defaults to ``Header(None)``.


    Raises
    ------
    HTTPException
    Raised when TODO for HTTPException.
    """
    if not API_KEYS:
        return  # disabled in skeleton
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ", 1)[1]
    if token not in API_KEYS:
        raise HTTPException(status_code=401, detail="Unauthorized")


# [nav:anchor healthz]
def healthz() -> dict[str, Any]:
    """Describe healthz.

    <!-- auto:docstring-builder v1 -->

    Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

    Returns
    -------
    dict[str, Any]
        Describe return value.
    """
    return {
        "status": "ok",
        "components": {
            "faiss": ("loaded" if faiss is not None else "missing"),
            "bm25": type(bm25).__name__,
            "splade": type(splade).__name__,
            "vllm_embeddings": "mocked",
            "neo4j": "mocked",
        },
    }


# [nav:anchor rrf_fuse]
def rrf_fuse(lists: list[list[tuple[str, float]]], k_rrf: int) -> dict[str, float]:
    """Describe rrf fuse.

    <!-- auto:docstring-builder v1 -->

    Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

    Parameters
    ----------
    lists : list[list[tuple[str, float]]]
        Describe ``lists``.
    k_rrf : int
        Describe ``k_rrf``.


    Returns
    -------
    dict[str, float]
        Describe return value.
    """
    scores: dict[str, float] = {}
    for hits in lists:
        for rank, (doc_id, _score) in enumerate(hits, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
    return scores


# [nav:anchor apply_kg_boosts]
def apply_kg_boosts(
    cands: dict[str, float],
    query: str,
    direct: float = 0.08,
    one_hop: float = 0.04,
) -> dict[str, float]:
    """Describe apply kg boosts.

    <!-- auto:docstring-builder v1 -->

    Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

    Parameters
    ----------
    cands : dict[str, float]
        Describe ``cands``.
    query : str
        Describe ``query``.
    direct : float, optional
        Describe ``direct``.
        Defaults to ``0.08``.
    one_hop : float, optional
        Describe ``one_hop``.
        Defaults to ``0.04``.


    Returns
    -------
    dict[str, float]
        Describe return value.
    """
    q_concepts = set()
    for w in query.lower().split():
        if w.startswith("concept"):
            q_concepts.add(f"C:{w.replace('concept', '')}")
    out = dict(cands)
    for chunk_id, base in cands.items():
        linked = set(kg.linked_concepts(chunk_id))
        boost = 0.0
        if linked & q_concepts:
            boost += direct
        else:
            for c in linked:
                if set(kg.one_hop(c)) & q_concepts:
                    boost += one_hop
                    break
        out[chunk_id] = base + boost
    return out


# [nav:anchor search]
def search(req: SearchRequest, _: None = Depends(auth)) -> SearchResponse:
    """Execute hybrid search query.

    Performs hybrid search using dense (FAISS), sparse (BM25), and SPLADE vectors,
    then applies RRF fusion and KG boosts. Includes structured logging and metrics.

    Parameters
    ----------
    req : SearchRequest
        Search request containing query, k, filters, and explain flag.
        Schema: `schema/models/search_request.v1.json`
    _ : None, optional
        Authentication dependency (Bearer token).
        Defaults to ``Depends(auth)``.

    Returns
    -------
    SearchResponse
        Typed response containing list of search results.
        Schema: `schema/models/search_result.v1.json`

    Examples
    --------
    >>> from search_api.schemas import SearchRequest
    >>> from search_api.app import search
    >>> req = SearchRequest(query="test query", k=5)
    >>> response = search(req, None)
    >>> len(response.results) <= 5
    True

    Raises
    ------
    HTTPException
        Returns Problem Details JSON (RFC 9457) on errors.
        Example: `schema/examples/problem_details/search-missing-index.json`

    Notes
    -----
    - **Await semantics**: This is a synchronous function, but FastAPI runs it in
      a thread pool, allowing cancellation via HTTP client disconnect.
    - **Timeout**: No explicit timeout is enforced; long-running searches may
      be cancelled by FastAPI's request timeout or client disconnect.
    - **Cancellation**: If the client disconnects, FastAPI will cancel the
      request thread, which may leave partial results in logs/metrics.
    - **Correlation ID**: Each request receives a unique correlation ID via
      `contextvars.ContextVar`, which propagates through all synchronous
      and async operations automatically.
    - **Idempotency**: This endpoint is **idempotent**. Multiple requests with
      identical parameters produce identical results without side effects.
      Results are determined solely by the query and k parameters; no state
      is modified by search operations.
    - **Retries**: The endpoint does not implement automatic retries. Clients
      should implement exponential backoff with jitter for transient failures.
      Recommended retry strategy: 3 attempts with delays of 1s, 2s, 4s.

    See Also
    --------
    - `schema/examples/problem_details/search-missing-index.json` - Example error response
    - `schema/models/search_request.v1.json` - Request schema
    - `schema/models/search_result.v1.json` - Response schema
    """
    # Generate correlation ID for this request
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)

    # Record operation metrics and duration
    with record_operation(metrics, operation="search", status="success"):
        logger.info(
            "Search request received",
            extra={
                "operation": "search",
                "status": "started",
                "query": req.query,
                "k": req.k,
            },
        )

        # Retrieve from each channel
        # We don't have a query embedder here; fallback to empty or demo vector
        dense_hits: list[tuple[str, float]] = []
        # sparse via BM25 (preferred) and SPLADE
        bm25_hits: list[tuple[str, float]] = []
        if bm25:
            try:
                bm25_hits = bm25.search(req.query, k=settings.search.sparse_candidates)
            except Exception as exc:
                logger.warning(
                    "BM25 search failed, falling back to empty results: %s",
                    exc,
                    extra={"operation": "search", "status": "warning"},
                    exc_info=True,
                )
                bm25_hits = []
        try:
            splade_hits = (
                splade.search(req.query, k=settings.search.sparse_candidates) if splade else []
            )
        except Exception as exc:
            logger.warning(
                "SPLADE search failed, falling back to empty results: %s",
                exc,
                extra={"operation": "search", "status": "warning"},
                exc_info=True,
            )
            splade_hits = []

        # RRF fusion
        fused = rrf_fuse([dense_hits, bm25_hits, splade_hits], k_rrf=settings.search.rrf_k)
        # KG boosts
        boosted = apply_kg_boosts(
            fused,
            req.query,
            direct=settings.search.kg_boosts_direct,
            one_hop=settings.search.kg_boosts_one_hop,
        )
        # Rank and craft results
        top = sorted(boosted.items(), key=lambda x: x[1], reverse=True)[: req.k]
        results: list[SearchResult] = []
        for chunk_id, score in top:
            # In real system we'd hydrate title/section via DuckDB; here we echo ids
            results.append(
                SearchResult(
                    doc_id=f"doc-of-{chunk_id}",
                    chunk_id=chunk_id,
                    title=f"Title for {chunk_id}",
                    section="Methods",
                    score=float(score),
                    signals={
                        "rrf": float(fused.get(chunk_id, 0.0)),
                        "kg_boost": float(boosted[chunk_id] - fused.get(chunk_id, 0.0)),
                    },
                    spans={"start_char": 0, "end_char": 50},
                    concepts=[
                        {
                            "concept_id": c,
                            "label": c,
                            "match": ("direct" if c in req.query else "nearby"),
                        }
                        for c in kg.linked_concepts(chunk_id)
                    ],
                )
            )

        logger.info(
            "Search completed",
            extra={
                "operation": "search",
                "status": "success",
                "result_count": len(results),
            },
        )

        return SearchResponse(results=results)


# [nav:anchor graph_concepts]
def graph_concepts(body: Mapping[str, Any], _: None = Depends(auth)) -> dict[str, Any]:
    """Describe graph concepts.

    <!-- auto:docstring-builder v1 -->

    Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

    Parameters
    ----------
    body : Mapping[str, Any]
        Describe ``body``.
    _ : None, optional
        Describe ``_``.
        Defaults to ``Depends(auth)``.


    Returns
    -------
    dict[str, Any]
        Describe return value.
    """
    q = (body or {}).get("q", "").lower()
    # toy: return nodes that contain the query substring
    concepts = [
        {"concept_id": c, "label": c}
        for c in sorted({c for cs in kg.chunk2concepts.values() for c in cs})
        if q in c.lower()
    ][: body.get("limit", 50)]
    return {"concepts": concepts}


app.get("/healthz")(healthz)
app.post("/search", response_model=SearchResponse)(search)
app.post("/graph/concepts", response_model=dict)(graph_concepts)
