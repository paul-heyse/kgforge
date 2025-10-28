"""Fusion utilities."""

from __future__ import annotations

from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["rrf_fuse"]

__navmap__: Final[NavMap] = {
    "title": "search_api.fusion",
    "synopsis": "Reciprocal rank fusion helpers for combining retrieval signals",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@search-api",
        "stability": "experimental",
        "since": "0.2.0",
    },
    "symbols": {
        "rrf_fuse": {
            "owner": "@search-api",
            "stability": "experimental",
            "since": "0.2.0",
        },
    },
}


# [nav:anchor rrf_fuse]
def rrf_fuse(rankers: list[list[tuple[str, float]]], k: int = 60) -> dict[str, float]:
    """
    Compute rrf fuse.
    
    Carry out the rrf fuse operation.
    
    Parameters
    ----------
    rankers : List[List[Tuple[str, float]]]
        Description for ``rankers``.
    k : int | None
        Description for ``k``.
    
    Returns
    -------
    Mapping[str, float]
        Description of return value.
    """
    agg: dict[str, float] = {}
    for ranked in rankers:
        for r, (key, _score) in enumerate(ranked, start=1):
            agg[key] = agg.get(key, 0.0) + 1.0 / (k + r)
    return agg
