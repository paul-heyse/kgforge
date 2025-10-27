"""Mock knowledge-graph lookups used by the search API for demos."""

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
    """Metadata describing a mock concept and its keyword matches."""

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
    """Extract concept identifiers from ``query`` using simple heuristics."""
    lowered = query.lower()
    hits: set[str] = set()
    for concept_id, meta in CONCEPTS.items():
        if any(keyword in lowered for keyword in meta["keywords"]):
            hits.add(concept_id)
    return hits


# [nav:anchor linked_concepts_for_text]
def linked_concepts_for_text(text: str) -> list[str]:
    """Derive concept URNs mentioned in a chunk of indexed text."""
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
    """Return a boost weight when query and chunk share a concept."""
    _ = one_hop  # placeholder for future graph traversal heuristics
    return direct if set(query_concepts) & set(chunk_concepts) else 0.0
