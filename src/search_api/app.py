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

import json
import uuid
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Final

import jsonschema
from fastapi import Depends, FastAPI, Header, HTTPException
from jsonschema.exceptions import ValidationError as SchemaValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse, Response

from kgfoundry.embeddings_sparse.bm25 import LuceneBM25, PurePythonBM25, get_bm25
from kgfoundry.embeddings_sparse.splade import get_splade
from kgfoundry.kg_builder.mock_kg import MockKG
from kgfoundry_common.errors import (
    SerializationError,
    VectorSearchError,
)
from kgfoundry_common.errors.http import register_problem_details_handler
from kgfoundry_common.logging import get_logger, set_correlation_id, with_fields
from kgfoundry_common.navmap_types import NavMap
from kgfoundry_common.observability import MetricsProvider, observe_duration
from kgfoundry_common.problem_details import JsonValue
from kgfoundry_common.schema_helpers import load_schema
from kgfoundry_common.settings import RuntimeSettings
from search_api.fusion import rrf_fuse
from search_api.schemas import SearchRequest, SearchResponse, SearchResult
from search_api.service import apply_kg_boosts

__all__ = [
    "CorrelationIDMiddleware",
    "ResponseValidationMiddleware",
    "app",
    "auth",
    "graph_concepts",
    "healthz",
    "search",
]

__navmap__: Final[NavMap] = {
    "title": "search_api.app",
    "synopsis": "FastAPI endpoints for hybrid search with FAISS, BM25, and SPLADE",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": [
                "app",
                "search",
                "graph_concepts",
                "healthz",
                "auth",
                "CorrelationIDMiddleware",
                "ResponseValidationMiddleware",
            ],
        }
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
metrics = MetricsProvider.default()

API_KEYS: set[str] = set()  # NOTE: load from env SEARCH_API_KEYS when secrets wiring is ready

app = FastAPI(title="kgfoundry Search API", version="0.2.0")

# Register Problem Details exception handler
register_problem_details_handler(app)


# Correlation ID middleware
class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Middleware to extract and set correlation ID from X-Correlation-ID header.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    app : ASGIApp
        Describe ``app``.
    dispatch : DispatchFunction | None, optional
        Describe ``dispatch``.
        Defaults to ``None``.
    """

    async def dispatch(
        self,
        request: StarletteRequest,
        call_next: Callable[[StarletteRequest], Awaitable[Response]],
    ) -> Response:
        """Extract correlation ID from header or generate new one.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        request : StarletteRequest
            Describe ``request``.
        call_next : [<class 'starlette.requests.Request'>] | Response
            Describe ``call_next``.

        Returns
        -------
        Response
            Describe return value.
        """
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        set_correlation_id(correlation_id)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response


# Response validation middleware
class ResponseValidationMiddleware(BaseHTTPMiddleware):
    """Middleware to validate JSON responses against schema (dev/staging only).

    <!-- auto:docstring-builder v1 -->

    Validates responses against search_response.json schema when enabled.
    Logs validation failures and returns Problem Details on schema mismatch.

    Parameters
    ----------
    app : FastAPI
        Describe ``app``.
    enabled : bool, optional
        Describe ``enabled``.
        Defaults to ``False``.
    schema_path : Path | None, optional
        Describe ``schema_path``.
        Defaults to ``None``.
    """

    def __init__(
        self,
        app: FastAPI,
        *,
        enabled: bool = False,
        schema_path: Path | None = None,
    ) -> None:
        """Initialize response validation middleware.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        app : FastAPI
            FastAPI application instance.
        enabled : bool, optional
            Whether to enable response validation.
            Defaults to False.
            Defaults to ``False``.
        schema_path : Path | NoneType, optional
            Path to search_response.json schema file.
            If None, searches for schema/search/search_response.json.
            Defaults to None.
            Defaults to ``None``.
        """
        super().__init__(app)
        self.enabled = enabled
        if schema_path is None:
            # Default to schema/search/search_response.json relative to repo root
            repo_root = Path(__file__).parent.parent.parent
            schema_path = repo_root / "schema" / "search" / "search_response.json"
        self.schema_path = schema_path
        self.schema: dict[str, JsonValue] | None = None
        if self.enabled and self.schema_path.exists():
            try:
                self.schema = load_schema(self.schema_path)
                logger.info(
                    "Response validation enabled",
                    extra={"schema_path": str(self.schema_path)},
                )
            except Exception as exc:  # noqa: BLE001 - defensive guard for schema loading
                logger.warning(
                    "Failed to load response schema, validation disabled",
                    extra={"schema_path": str(self.schema_path), "error": str(exc)},
                    exc_info=True,
                )
                self.enabled = False

    async def dispatch(
        self,
        request: StarletteRequest,
        call_next: Callable[[StarletteRequest], Awaitable[Response]],
    ) -> Response:
        """Validate response against schema if enabled.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        request : StarletteRequest
            Describe ``request``.
        call_next : [<class 'starlette.requests.Request'>] | Response
            Describe ``call_next``.

        Returns
        -------
        Response
            Describe return value.
        """
        if not self.enabled or self.schema is None:
            return await call_next(request)

        response = await call_next(request)

        # Only validate JSON responses
        if not isinstance(response, JSONResponse):  # type: ignore[misc]  # JSONResponse isinstance check
            return response

        # Only validate /search endpoint responses
        if request.url.path != "/search":
            return response

        # Extract response body from JSONResponse
        try:
            # JSONResponse has a 'body' property that contains the rendered body
            body_bytes = response.body
            response_body: JsonValue = json.loads(body_bytes)
        except (json.JSONDecodeError, AttributeError) as exc:
            with with_fields(logger, operation="response_validation") as log_adapter:
                log_adapter.warning(
                    "Failed to parse response body for validation",
                    extra={"error": str(exc)},
                    exc_info=True,
                )
            # Return original response if parsing fails
            return response

        # Validate against schema
        try:
            jsonschema.validate(instance=response_body, schema=self.schema)
        except SchemaValidationError as exc:
            with with_fields(logger, operation="response_validation") as log_adapter:
                log_adapter.exception(
                    "Response validation failed",
                    extra={
                        "status": "error",
                        "validation_error": exc.message,
                        "path": str(exc.path),
                    },
                )
            # Return Problem Details for validation failure
            error_msg = f"Response validation failed: {exc.message}"
            problem = SerializationError(
                error_msg,
                cause=exc,
                context={"schema_path": str(self.schema_path), "validation_path": str(exc.path)},
            )
            return JSONResponse(
                content=problem.to_problem_details(),
                status_code=500,
                headers=response.headers,
            )

        # Return original response if validation passes
        return response


app.middleware("http")(CorrelationIDMiddleware)  # type: ignore[misc]  # FastAPI middleware registration

# --- bootstrap typed configuration ---
try:
    settings = RuntimeSettings()
except Exception:
    logger.exception("Failed to load settings, using defaults")
    from kgfoundry_common.settings import (
        FaissConfig,
        ObservabilityConfig,
        SearchConfig,
        SparseEmbeddingConfig,
    )

    settings = RuntimeSettings(
        search=SearchConfig(),
        sparse_embedding=SparseEmbeddingConfig(),
        faiss=FaissConfig(),
        observability=ObservabilityConfig(),
    )

# Initialize response validation middleware if enabled
if settings.search.validate_responses:
    # Create a factory function that returns configured middleware
    def _create_response_validator(app: FastAPI) -> ResponseValidationMiddleware:
        """Describe  create response validator.

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        app : FastAPI
        Configure the app.


        Returns
        -------
        ResponseValidationMiddleware
        Describe return value.
        """
        return ResponseValidationMiddleware(app, enabled=True)

    app.middleware("http")(_create_response_validator)

# --- bootstrap search backends ---
kg = MockKG()
bm25: PurePythonBM25 | LuceneBM25 | None = None
splade = None

try:
    bm25 = get_bm25(
        backend=settings.search.sparse_backend,
        index_dir=settings.sparse_embedding.bm25_index_dir,
    )
except Exception as exc:  # noqa: BLE001 - defensive guard for BM25 loading
    logger.warning("BM25 index not available: %s", exc, exc_info=True)

try:
    splade = get_splade(
        backend=settings.search.sparse_backend,
        index_dir=settings.sparse_embedding.splade_index_dir,
        query_encoder=settings.sparse_embedding.splade_query_encoder,
    )
except Exception as exc:  # noqa: BLE001 - defensive guard for SPLADE loading
    logger.warning("SPLADE index not available: %s", exc, exc_info=True)


# [nav:anchor auth]
def auth(
    authorization: str | None = Header(default=None),  # type: ignore[assignment]  # FastAPI Header marker type
) -> None:
    """Validate bearer token authentication.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    authorization : str | NoneType, optional
        Authorization header value (Bearer token).
        Defaults to None.
        Defaults to ``Header(None)``.

    Raises
    ------
    HTTPException
        Returns 401 if token is invalid or missing.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Bearer token")
    token = authorization.split(" ", 1)[1]
    if token not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")


# [nav:anchor healthz]
def healthz() -> dict[str, str | dict[str, str]]:
    """Health check endpoint.

    <!-- auto:docstring-builder v1 -->

    Returns
    -------
    dict[str, str | dict[str, str]]
        Health status with component availability.
    """
    return {
        "status": "ok",
        "components": {
            "bm25": "available" if bm25 else "unavailable",
            "splade": "available" if splade else "unavailable",
            "vllm_embeddings": "mocked",
            "neo4j": "mocked",
        },
    }


# [nav:anchor search]
def search(req: SearchRequest, _: None = Depends(auth)) -> SearchResponse:  # type: ignore[assignment]  # FastAPI Depends marker type
    """Execute hybrid search query.

    <!-- auto:docstring-builder v1 -->

    Combines dense (FAISS), sparse (BM25/SPLADE), and knowledge graph signals
    using Reciprocal Rank Fusion and KG boosts. Returns ranked results with
    structured logging and metrics.

    Parameters
    ----------
    req : SearchRequest
        Search request containing query text, k, and optional facets.
    _ : None, optional
        Authentication dependency (Bearer token).
        Defaults to ``Depends(auth)``.
        Defaults to ``Depends(dependency=<function auth at 0x73d12bcbe020>, use_cache=True)``.
        Defaults to ``Depends(dependency=<function auth at 0x73a51e5e4c20>, use_cache=True)``.
        Defaults to ``Depends(dependency=<function auth at 0x7e8041099800>, use_cache=True)``.
        Defaults to ``Depends(dependency=<function auth at 0x7b8c3c365760>, use_cache=True)``.
        Defaults to ``Depends(dependency=<function auth at 0x72c0c6f89760>, use_cache=True)``.
        Defaults to ``Depends(dependency=<function auth at 0x7876585e9760>, use_cache=True)``.

    Returns
    -------
    SearchResponse
        Search results with metadata and performance metrics.

    Raises
    ------
    VectorSearchError
        Returns Problem Details JSON (RFC 9457) on errors.

    Examples
    --------
    >>> from search_api.schemas import SearchRequest
    >>> from search_api.app import search
    >>> req = SearchRequest(query="vector store", k=5)
    >>> response = search(req, None)
    >>> len(response.results) <= 5
    True
    """
    # Correlation ID is set by middleware, use with_fields for structured logging
    with (
        with_fields(logger, operation="search", query=req.query, k=req.k),
        observe_duration(metrics, "search", component="search_api") as obs,
    ):
        log_adapter = logger  # Use logger directly since with_fields provides context
        log_adapter.info("Search request received", extra={"status": "started"})

        try:
            # Retrieve from each channel
            # We don't have a query embedder here; fallback to empty or demo vector
            dense_hits: list[tuple[str, float]] = []
            # sparse via BM25 (preferred) and SPLADE
            bm25_hits: list[tuple[str, float]] = []
            if bm25:
                try:
                    bm25_hits = bm25.search(req.query, k=settings.search.sparse_candidates)
                except (RuntimeError, ValueError, AttributeError, OSError) as exc:
                    log_adapter.warning(
                        "BM25 search failed, falling back to empty results: %s",
                        exc,
                        extra={"status": "warning"},
                        exc_info=True,
                    )
                    bm25_hits = []
            try:
                splade_hits = (
                    splade.search(req.query, k=settings.search.sparse_candidates) if splade else []
                )
            except (RuntimeError, ValueError, AttributeError, OSError) as exc:
                log_adapter.warning(
                    "SPLADE search failed, falling back to empty results: %s",
                    exc,
                    extra={"status": "warning"},
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
            def sort_key(item: tuple[str, float]) -> float:
                """Sort key function for ranking results."""
                return item[1]

            top = sorted(boosted.items(), key=sort_key, reverse=True)[: req.k]
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

            log_adapter.info(
                "Search completed", extra={"status": "success", "result_count": len(results)}
            )
            obs.success()

            return SearchResponse(results=results)
        except (RuntimeError, ValueError, AttributeError, OSError) as exc:
            obs.error()
            # Convert to VectorSearchError for proper Problem Details handling
            error_msg = f"Search operation failed: {exc}"
            # context accepts Mapping[str, object], dict[str, str | int] is compatible
            context_dict: Mapping[str, object] = {"query": req.query, "k": req.k}
            raise VectorSearchError(error_msg, cause=exc, context=context_dict) from exc


# [nav:anchor graph_concepts]
def graph_concepts(
    body: Mapping[str, JsonValue],
    _: None = Depends(auth),  # type: ignore[assignment]  # FastAPI Depends marker type
) -> dict[str, list[dict[str, str]]]:
    """Retrieve knowledge graph concepts matching query.

    <!-- auto:docstring-builder v1 -->

    Returns concepts from the knowledge graph that match the query string.
    Includes structured logging and error handling.

    Parameters
    ----------
    body : str | object
        Request body containing:
        - `q` (str): Query string to match against concept labels.
        - `limit` (int, optional): Maximum number of concepts to return. Defaults to 50.
    _ : None, optional
        Authentication dependency (Bearer token).
        Defaults to ``Depends(auth)``.
        Defaults to ``Depends(dependency=<function auth at 0x73d12bcbe020>, use_cache=True)``.
        Defaults to ``Depends(dependency=<function auth at 0x73a51e5e4c20>, use_cache=True)``.
        Defaults to ``Depends(dependency=<function auth at 0x7e8041099800>, use_cache=True)``.
        Defaults to ``Depends(dependency=<function auth at 0x7b8c3c365760>, use_cache=True)``.
        Defaults to ``Depends(dependency=<function auth at 0x72c0c6f89760>, use_cache=True)``.
        Defaults to ``Depends(dependency=<function auth at 0x7876585e9760>, use_cache=True)``.

    Returns
    -------
    dict[str, list[dict[str, str]]]
        Dictionary with "concepts" key containing list of concept dictionaries.
        Each concept has "concept_id" and "label" keys.

    Raises
    ------
    VectorSearchError
        Returns Problem Details JSON (RFC 9457) on errors.

    Examples
    --------
    >>> from search_api.app import graph_concepts
    >>> result = graph_concepts({"q": "test", "limit": 10}, None)
    >>> "concepts" in result
    True
    """
    with with_fields(logger, operation="graph_concepts") as log_adapter:
        q: str = ""
        try:
            q = str((body or {}).get("q", "")).lower()
            limit_raw = body.get("limit", 50) if body else 50
            try:
                limit: int = int(limit_raw)  # type: ignore[arg-type]  # limit_raw may be JsonValue
                if limit < 0:
                    limit = 50
            except (ValueError, TypeError):
                limit = 50

            # toy: return nodes that contain the query substring
            concepts: list[dict[str, str]] = [
                {"concept_id": c, "label": c}
                for c in sorted({c for cs in kg.chunk2concepts.values() for c in cs})
                if q in c.lower()
            ][:limit]

            log_adapter.info(
                "Graph concepts retrieved", extra={"status": "success", "count": len(concepts)}
            )
        except (RuntimeError, ValueError, AttributeError, OSError) as exc:
            error_msg = f"Graph concepts operation failed: {exc}"
            # context accepts Mapping[str, object], dict[str, str] is compatible
            context_dict: Mapping[str, object] = {"query": q}
            raise VectorSearchError(error_msg, cause=exc, context=context_dict) from exc
        else:
            return {"concepts": concepts}


app.get("/healthz")(healthz)
app.post("/search", response_model=SearchResponse)(search)
app.post("/graph/concepts", response_model=dict)(graph_concepts)
