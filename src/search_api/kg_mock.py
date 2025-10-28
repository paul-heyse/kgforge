"""Kg Mock utilities."""

from __future__ import annotations

from typing import Final, TypedDict

from kgfoundry_common.navmap_types import NavMap

__all__ = ["detect_query_concepts", "kg_boost", "linked_concepts_for_text"]

__navmap__: Final[NavMap] = {
    "title": "search_api.kg_mock",
    "synopsis": "Mock KG helpers for search API demos",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": ["detect_query_concepts", "linked_concepts_for_text", "kg_boost"],
        },
    ],
}


class ConceptMeta(TypedDict):
    """Describe ConceptMeta."""

    label: str
    keywords: list[str]


CONCEPTS: Final[dict[str, ConceptMeta]] = {
    "urn:concept:toy:LLM": {
        "label": "Large Language Model",
        "keywords": ["llm", "language model", "transformer"],
    },
    "urn:concept:toy:Alignment": {
        "label": "Alignment",
        "keywords": ["alignment", "safety", "rlhf"],
    },
}


# [nav:anchor detect_query_concepts]
def detect_query_concepts(query: str) -> set[str]:
    """Compute detect query concepts.

    Carry out the detect query concepts operation.

    Parameters
    ----------
    query : str
        Description for ``query``.

    Returns
    -------
    Set[str]
        Description of return value.
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    lowered = query.lower()
    hits: set[str] = set()
    for concept_id, meta in CONCEPTS.items():
        if any(keyword in lowered for keyword in meta["keywords"]):
            hits.add(concept_id)
    return hits


# [nav:anchor linked_concepts_for_text]
def linked_concepts_for_text(text: str) -> list[str]:
    """Compute linked concepts for text.

    Carry out the linked concepts for text operation.

    Parameters
    ----------
    text : str
        Description for ``text``.

    Returns
    -------
    List[str]
        Description of return value.
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    lowered = text.lower()
    hits = []
    for concept_id, meta in CONCEPTS.items():
        if any(keyword in lowered for keyword in meta["keywords"]):
            hits.append(concept_id)
    return hits


# [nav:anchor kg_boost]
def kg_boost(
    query_concepts: list[str],
    chunk_concepts: list[str],
    direct: float = 0.08,
    one_hop: float = 0.04,
) -> float:
    """Compute kg boost.

    Carry out the kg boost operation.

    Parameters
    ----------
    query_concepts : List[str]
        Description for ``query_concepts``.
    chunk_concepts : List[str]
        Description for ``chunk_concepts``.
    direct : float | None
        Description for ``direct``.
    one_hop : float | None
        Description for ``one_hop``.

    Returns
    -------
    float
        Description of return value.
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    _ = one_hop  # placeholder for future graph traversal heuristics
    return direct if set(query_concepts) & set(chunk_concepts) else 0.0
