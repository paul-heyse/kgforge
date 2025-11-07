"""Hybrid retrieval with RRF fusion.

Fuses results from BM25, SPLADE, and FAISS using Reciprocal Rank Fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class SearchHit:
    """Search result hit from a single retrieval system.

    Represents one search result from a retrieval system (FAISS, BM25, SPLADE, etc.)
    before fusion. Contains the document/chunk ID, its relevance score, rank
    position, and which retrieval system found it.

    SearchHit objects are created for each retrieval system's results, then
    combined using Reciprocal Rank Fusion (RRF) to produce a unified ranked list.
    The source field allows tracking which system contributed each result, which
    is useful for analysis and debugging.

    Attributes
    ----------
    doc_id : str
        Document or chunk ID that was retrieved. This should match the ID used
        in the index (e.g., chunk ID from DuckDB). Used to look up full chunk
        information after fusion.
    score : float
        Original relevance score from the retrieval system. Score ranges and
        meanings vary by system (FAISS uses cosine similarity, BM25 uses BM25 score,
        etc.). Used for debugging and understanding individual system performance.
    rank : int
        Rank position in the original result list (0-indexed). Rank 0 is the
        top result. Used by RRF to compute fusion scores - lower ranks get higher
        RRF scores.
    source : str
        Identifier for the retrieval system that produced this hit (e.g., "faiss",
        "bm25", "splade", "structural"). Used for analysis, debugging, and
        understanding which systems contribute to final results.
    """

    doc_id: str
    score: float
    rank: int
    source: str


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[SearchHit]],
    k: int = 60,
    top_k: int = 50,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists using RRF.

    Parameters
    ----------
    result_lists : Sequence[Sequence[SearchHit]]
        Lists of search hits from different retrieval systems.
    k : int
        RRF K parameter (higher = more weight to lower ranks).
    top_k : int
        Number of results to return.

    Returns
    -------
    list[tuple[str, float]]
        Fused results as (doc_id, rrf_score) sorted by score descending.

    Notes
    -----
    RRF score for document d is:
        RRF(d) = sum over all systems S of: 1 / (k + rank_S(d))

    where rank_S(d) is the rank of d in system S (1-indexed).
    """
    # Aggregate scores
    scores: dict[str, float] = {}

    for result_list in result_lists:
        for hit in result_list:
            # Rank is 1-indexed for RRF formula
            rank_one_indexed = hit.rank + 1
            rrf_score = 1.0 / (k + rank_one_indexed)

            if hit.doc_id in scores:
                scores[hit.doc_id] += rrf_score
            else:
                scores[hit.doc_id] = rrf_score

    # Sort by score descending
    sorted_results = sorted(scores.items(), key=lambda x: -x[1])

    return sorted_results[:top_k]


def create_hit_list(
    doc_ids: Sequence[str],
    scores: Sequence[float],
    source: str,
) -> list[SearchHit]:
    """Create SearchHit list from retrieval results.

    Parameters
    ----------
    doc_ids : Sequence[str]
        Document IDs in rank order.
    scores : Sequence[float]
        Scores for each document.
    source : str
        Source identifier (e.g., "bm25", "faiss").

    Returns
    -------
    list[SearchHit]
        Search hits with ranks.
    """
    return [
        SearchHit(doc_id=doc_id, score=score, rank=rank, source=source)
        for rank, (doc_id, score) in enumerate(zip(doc_ids, scores, strict=True))
    ]


__all__ = [
    "SearchHit",
    "create_hit_list",
    "reciprocal_rank_fusion",
]
