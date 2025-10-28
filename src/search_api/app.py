"""App utilities."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any, Final

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException
from kgfoundry.embeddings_sparse.bm25 import PurePythonBM25, get_bm25
from kgfoundry.embeddings_sparse.splade import get_splade
from kgfoundry.kg_builder.mock_kg import MockKG
from kgfoundry.vectorstore_faiss.gpu import FaissGpuIndex

from kgfoundry_common.navmap_types import NavMap
from search_api.schemas import SearchRequest, SearchResult

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
}

logger = logging.getLogger(__name__)

API_KEYS: set[str] = set()  # NOTE: load from env SEARCH_API_KEYS when secrets wiring is ready

app = FastAPI(title="kgfoundry Search API", version="0.2.0")

# --- bootstrap lightweight dependencies from config ---
CFG_PATH = os.environ.get(
    "KGF_CONFIG", os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
)
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

SPARSE_BACKEND = (CFG.get("search", {}).get("sparse_backend", "lucene") or "lucene").lower()
BM25_DIR = CFG.get("sparse_embedding", {}).get("bm25", {}).get("index_dir", "./_indices/bm25")
SPLADE_DIR = (
    CFG.get("sparse_embedding", {}).get("splade", {}).get("index_dir", "./_indices/splade_impact")
)
FAISS_PATH = "./_indices/faiss/shard_000.idx"

bm25 = get_bm25(
    SPARSE_BACKEND,
    BM25_DIR,
    k1=CFG.get("sparse_embedding", {}).get("bm25", {}).get("k1", 0.9),
    b=CFG.get("sparse_embedding", {}).get("bm25", {}).get("b", 0.4),
    field_boosts=CFG.get("sparse_embedding", {}).get("bm25", {}).get("field_boosts", None),
)
# load index if exists
try:
    if isinstance(bm25, PurePythonBM25) and os.path.exists(os.path.join(BM25_DIR, "pure_bm25.pkl")):
        bm25.load()
except Exception:
    pass

splade = get_splade(SPARSE_BACKEND, SPLADE_DIR)
try:
    # for PureImpactIndex, try to load
    if hasattr(splade, "load"):
        splade.load()
except Exception:
    pass

faiss = FaissGpuIndex(
    factory=CFG.get("faiss", {}).get("index_factory", "OPQ64,IVF8192,PQ64"),
    nprobe=int(CFG.get("faiss", {}).get("nprobe", 64)),
    gpu=bool(CFG.get("faiss", {}).get("gpu", True)),
    cuvs=bool(CFG.get("faiss", {}).get("cuvs", True)),
)
try:
    if os.path.exists(FAISS_PATH):
        faiss.load(FAISS_PATH, None)
except Exception:
    pass

# tiny KG mock with a few edges/mentions to demonstrate boosts
kg = MockKG()
kg.add_mention("chunk:1", "C:42")
kg.add_edge("C:42", "C:99")


# [nav:anchor auth]
def auth(authorization: str | None = Header(default=None)) -> None:
    """Compute auth.

    Carry out the auth operation.

    Parameters
    ----------
    authorization : str | None
        Description for ``authorization``.

    Raises
    ------
    HTTPException
        Raised when validation fails.
    """
    
    
    
    
    
    
    
    
    
    if not API_KEYS:
        return  # disabled in skeleton
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ", 1)[1]
    if token not in API_KEYS:
        raise HTTPException(status_code=401, detail="Unauthorized")


# [nav:anchor healthz]
@app.get("/healthz")
def healthz() -> dict[str, Any]:
    """Compute healthz.

    Carry out the healthz operation.

    Returns
    -------
    Mapping[str, Any]
        Description of return value.
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
    """Compute rrf fuse.

    Carry out the rrf fuse operation.

    Parameters
    ----------
    lists : List[List[Tuple[str, float]]]
        Description for ``lists``.
    k_rrf : int
        Description for ``k_rrf``.

    Returns
    -------
    Mapping[str, float]
        Description of return value.
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
    """Compute apply kg boosts.

    Carry out the apply kg boosts operation.

    Parameters
    ----------
    cands : Mapping[str, float]
        Description for ``cands``.
    query : str
        Description for ``query``.
    direct : float | None
        Description for ``direct``.
    one_hop : float | None
        Description for ``one_hop``.

    Returns
    -------
    Mapping[str, float]
        Description of return value.
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
@app.post("/search", response_model=dict)
def search(req: SearchRequest, _: None = Depends(auth)) -> dict[str, Any]:
    """Compute search.

    Carry out the search operation.

    Parameters
    ----------
    req : src.search_api.schemas.SearchRequest
        Description for ``req``.
    _ : None | None
        Description for ``_``.

    Returns
    -------
    Mapping[str, Any]
        Description of return value.
    """
    
    
    
    
    
    
    
    
    
    # Retrieve from each channel
    # We don't have a query embedder here; fallback to empty or demo vector
    dense_hits: list[tuple[str, float]] = []
    # sparse via BM25 (preferred) and SPLADE
    bm25_hits: list[tuple[str, float]] = []
    if bm25:
        try:
            bm25_hits = bm25.search(req.query, k=CFG["search"]["sparse_candidates"])
        except Exception as exc:  # pragma: no cover - defensive fallback for missing indices
            logger.warning("BM25 search failed, falling back to empty results: %s", exc)
            bm25_hits = []
    try:
        splade_hits = (
            splade.search(req.query, k=CFG["search"]["sparse_candidates"]) if splade else []
        )
    except Exception:
        splade_hits = []

    # RRF fusion
    fused = rrf_fuse([dense_hits, bm25_hits, splade_hits], k_rrf=int(CFG["search"]["rrf_k"]))
    # KG boosts
    boosted = apply_kg_boosts(
        fused,
        req.query,
        direct=CFG["search"]["kg_boosts"]["direct"],
        one_hop=CFG["search"]["kg_boosts"]["one_hop"],
    )
    # Rank and craft results
    top = sorted(boosted.items(), key=lambda x: x[1], reverse=True)[: req.k]
    results: list[dict[str, Any]] = []
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
            ).model_dump()
        )
    return {"results": results}


# [nav:anchor graph_concepts]
@app.post("/graph/concepts", response_model=dict)
def graph_concepts(body: Mapping[str, Any], _: None = Depends(auth)) -> dict[str, Any]:
    """Compute graph concepts.

    Carry out the graph concepts operation.

    Parameters
    ----------
    body : Mapping[str, Any]
        Description for ``body``.
    _ : None | None
        Description for ``_``.

    Returns
    -------
    Mapping[str, Any]
        Description of return value.
    """
    
    
    
    
    
    
    
    
    
    q = (body or {}).get("q", "").lower()
    # toy: return nodes that contain the query substring
    concepts = [
        {"concept_id": c, "label": c}
        for c in sorted({c for cs in kg.chunk2concepts.values() for c in cs})
        if q in c.lower()
    ][: body.get("limit", 50)]
    return {"concepts": concepts}
