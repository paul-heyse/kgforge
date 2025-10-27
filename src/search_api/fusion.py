"""Module for search_api.fusion.

NavMap:
- rrf_fuse: Rrf fuse.
"""

from __future__ import annotations


def rrf_fuse(rankers: list[list[tuple[str, float]]], k: int = 60) -> dict[str, float]:
    """Rrf fuse.

    Args:
        rankers (List[List[Tuple[str, float]]]): TODO.
        k (int): TODO.

    Returns:
        Dict[str, float]: TODO.
    """
    agg: dict[str, float] = {}
    for ranked in rankers:
        for r, (key, _score) in enumerate(ranked, start=1):
            agg[key] = agg.get(key, 0.0) + 1.0 / (k + r)
    return agg
