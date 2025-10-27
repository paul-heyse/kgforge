"""Module for search_api.service.

NavMap:
- rrf_fuse: Reciprocal Rank Fusion skeleton.
- apply_kg_boosts: Apply kg boosts.
- mmr_deduplicate: Mmr deduplicate.
"""

from __future__ import annotations


def rrf_fuse(
    dense: list[tuple[str, float]], sparse: list[tuple[str, float]], k: int = 60
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion skeleton."""
    # NOTE: implement stable RRF across rankers when ranker outputs are wired
    return []


def apply_kg_boosts(fused: list[tuple[str, float]], query: str) -> list[tuple[str, float]]:
    """Apply kg boosts.

    Parameters
    ----------
    fused : List[Tuple[str, float]]
        TODO.
    query : str
        TODO.

    Returns
    -------
    List[Tuple[str, float]]
        TODO.
    """
    # NOTE: apply boosts for direct & one-hop concept matches once KG signals exist
    return fused


def mmr_deduplicate(
    results: list[tuple[str, float]], lambda_: float = 0.7
) -> list[tuple[str, float]]:
    """Mmr deduplicate.

    Parameters
    ----------
    results : List[Tuple[str, float]]
        TODO.
    lambda_ : float
        TODO.

    Returns
    -------
    List[Tuple[str, float]]
        TODO.
    """
    # NOTE: add doc-level diversity via MMR when result scoring is available
    return results
