"""Module for search_api.service."""

from __future__ import annotations


def rrf_fuse(
    dense: list[tuple[str, float]], sparse: list[tuple[str, float]], k: int = 60
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion skeleton."""
    # TODO: implement stable RRF across rankers
    return []


def apply_kg_boosts(fused: list[tuple[str, float]], query: str) -> list[tuple[str, float]]:
    """Apply kg boosts.

    Args:
        fused (List[Tuple[str, float]]): TODO.
        query (str): TODO.

    Returns:
        List[Tuple[str, float]]: TODO.
    """
    # TODO: apply boosts for direct & one-hop concept matches
    return fused


def mmr_deduplicate(
    results: list[tuple[str, float]], lambda_: float = 0.7
) -> list[tuple[str, float]]:
    """Mmr deduplicate.

    Args:
        results (List[Tuple[str, float]]): TODO.
        lambda_ (float): TODO.

    Returns:
        List[Tuple[str, float]]: TODO.
    """
    # TODO: diversity via MMR at doc-level
    return results
