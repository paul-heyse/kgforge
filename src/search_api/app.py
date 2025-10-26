
from __future__ import annotations
import os, yaml, math
from typing import Optional, Dict, List, Tuple
from fastapi import FastAPI, HTTPException, Depends, Header
from .schemas import SearchRequest, SearchResult
from kgforge.embeddings_sparse.bm25 import get_bm25, PurePythonBM25
from kgforge.embeddings_sparse.splade import get_splade
from kgforge.vectorstore_faiss.gpu import FaissGpuIndex
from kgforge.kg_builder.mock_kg import MockKG

API_KEYS = set()  # TODO: load from env SEARCH_API_KEYS

app = FastAPI(title="KGForge Search API", version="0.2.0")

# --- bootstrap lightweight dependencies from config ---
CFG_PATH = os.environ.get("KGF_CONFIG", os.path.join(os.path.dirname(__file__), "../../config/config.yaml"))
with open(CFG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

SPARSE_BACKEND = (CFG.get("search", {}).get("sparse_backend", "lucene") or "lucene").lower()
BM25_DIR = CFG.get("sparse_embedding", {}).get("bm25", {}).get("index_dir", "./_indices/bm25")
SPLADE_DIR = CFG.get("sparse_embedding", {}).get("splade", {}).get("index_dir", "./_indices/splade_impact")
FAISS_PATH = "./_indices/faiss/shard_000.idx"

bm25 = get_bm25(SPARSE_BACKEND, BM25_DIR,
                k1=CFG.get("sparse_embedding", {}).get("bm25", {}).get("k1", 0.9),
                b=CFG.get("sparse_embedding", {}).get("bm25", {}).get("b", 0.4),
                field_boosts=CFG.get("sparse_embedding", {}).get("bm25", {}).get("field_boosts", None)
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

faiss = FaissGpuIndex(factory=CFG.get("faiss", {}).get("index_factory", "OPQ64,IVF8192,PQ64"),
                      nprobe=int(CFG.get("faiss", {}).get("nprobe", 64)),
                      gpu=bool(CFG.get("faiss", {}).get("gpu", True)),
                      cuvs=bool(CFG.get("faiss", {}).get("cuvs", True)))
try:
    if os.path.exists(FAISS_PATH):
        faiss.load(FAISS_PATH, None)
except Exception:
    pass

# tiny KG mock with a few edges/mentions to demonstrate boosts
kg = MockKG()
kg.add_mention("chunk:1", "C:42")
kg.add_edge("C:42", "C:99")

def auth(authorization: Optional[str] = Header(default=None)) -> None:
    if not API_KEYS:
        return  # disabled in skeleton
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ", 1)[1]
    if token not in API_KEYS:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/healthz")
def healthz():
    return {"status": "ok", "components": {"faiss": ("loaded" if faiss is not None else "missing"),
                                           "bm25": type(bm25).__name__,
                                           "splade": type(splade).__name__,
                                           "vllm_embeddings": "mocked",
                                           "neo4j": "mocked"}}

def rrf_fuse(lists: List[List[Tuple[str, float]]], k_rrf: int) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for hits in lists:
        for rank, (doc_id, _score) in enumerate(hits, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
    return scores

def apply_kg_boosts(cands: Dict[str, float], query: str, direct=0.08, one_hop=0.04) -> Dict[str, float]:
    # toy: map words 'concept42' to concept id
    q_concepts = set()
    for w in query.lower().split():
        if w.startswith("concept"):
            q_concepts.add(f"C:{w.replace('concept','')}")
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

@app.post("/search", response_model=dict)
def search(req: SearchRequest, _=Depends(auth)):
    # Retrieve from each channel
    dense_hits: List[Tuple[str, float]] = []  # we don't have a query embedder here; fallback to empty or demo vector
    # sparse via BM25 (preferred) and SPLADE
    bm25_hits = bm25.search(req.query, k=CFG["search"]["sparse_candidates"]) if bm25 else []
    try:
        splade_hits = splade.search(req.query, k=CFG["search"]["sparse_candidates"]) if splade else []
    except Exception:
        splade_hits = []

    # RRF fusion
    fused = rrf_fuse([dense_hits, bm25_hits, splade_hits], k_rrf=int(CFG["search"]["rrf_k"]))
    # KG boosts
    boosted = apply_kg_boosts(fused, req.query,
                              direct=CFG["search"]["kg_boosts"]["direct"],
                              one_hop=CFG["search"]["kg_boosts"]["one_hop"])
    # Rank and craft results
    top = sorted(boosted.items(), key=lambda x: x[1], reverse=True)[:req.k]
    results = []
    for chunk_id, score in top:
        # In real system we'd hydrate title/section via DuckDB; here we echo ids
        results.append(SearchResult(
            doc_id=f"doc-of-{chunk_id}",
            chunk_id=chunk_id,
            title=f"Title for {chunk_id}",
            section="Methods",
            score=float(score),
            signals={"rrf": float(fused.get(chunk_id, 0.0)), "kg_boost": float(boosted[chunk_id]-fused.get(chunk_id,0.0))},
            spans={"start_char": 0, "end_char": 50},
            concepts=[{"concept_id": c, "label": c, "match": ("direct" if c in req.query else "nearby")} for c in kg.linked_concepts(chunk_id)]
        ).model_dump())
    return {"results": results}

@app.post("/graph/concepts", response_model=dict)
def graph_concepts(body: dict, _=Depends(auth)):
    q = (body or {}).get("q","").lower()
    # toy: return nodes that contain the query substring
    concepts = [{"concept_id": c, "label": c} for c in sorted({c for cs in kg.chunk2concepts.values() for c in cs}) if q in c.lower()][: body.get("limit", 50)]
    return {"concepts": concepts}
