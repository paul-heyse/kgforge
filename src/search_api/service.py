"""Module for search_api.service.

NavMap:
- NavMap: Structure describing a module navmap.
- rrf_fuse: Fuse dense and sparse rankings via reciprocal rank fusion.
- apply_kg_boosts: Apply knowledge-graph based boosts to fused search results.
- mmr_deduplicate: De-duplicate search results using maximal marginal….
"""

from __future__ import annotations

from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = ["apply_kg_boosts", "mmr_deduplicate", "rrf_fuse"]

__navmap__: Final[NavMap] = {
    "title": "kgfoundry.search_api.service",
    "synopsis": "Ranking utilities composing the kgfoundry search pipeline",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["rrf_fuse", "apply_kg_boosts", "mmr_deduplicate"],
        }
    ],
    "symbols": {
        "rrf_fuse": {
            "since": "2024.10",
            "stability": "experimental",
            "side_effects": ["none"],
            "thread_safety": "reentrant",
            "async_ok": True,
        },
        "apply_kg_boosts": {
            "since": "2024.10",
            "stability": "experimental",
            "side_effects": ["none"],
            "thread_safety": "reentrant",
            "async_ok": True,
        },
        "mmr_deduplicate": {
            "since": "2024.10",
            "stability": "experimental",
            "side_effects": ["none"],
            "thread_safety": "reentrant",
            "async_ok": True,
        },
    },
    "edit_scopes": {"safe": ["apply_kg_boosts", "mmr_deduplicate"], "risky": ["rrf_fuse"]},
    "tags": ["search", "ranking"],
    "since": "2024.10",
}


# [nav:anchor rrf_fuse]
def rrf_fuse(
    dense: list[tuple[str, float]], sparse: list[tuple[str, float]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse dense and sparse rankings via reciprocal rank fusion."""
    # NOTE: implement stable RRF across rankers when ranker outputs are wired
    return []


# [nav:anchor apply_kg_boosts]
def apply_kg_boosts(fused: list[tuple[str, float]], query: str) -> list[tuple[str, float]]:
    """Apply knowledge-graph based boosts to fused search results."""
    # NOTE: apply boosts for direct & one-hop concept matches once KG signals exist
    return fused


# [nav:anchor mmr_deduplicate]
def mmr_deduplicate(
    results: list[tuple[str, float]], lambda_: float = 0.7
) -> list[tuple[str, float]]:
    """De-duplicate search results using maximal marginal relevance heuristics."""
    # NOTE: add doc-level diversity via MMR when result scoring is available
    return results
