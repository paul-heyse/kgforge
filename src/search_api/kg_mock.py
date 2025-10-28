"""Overview of kg mock.

This module bundles kg mock logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

from typing import Final, TypedDict

from kgfoundry_common.navmap_types import NavMap

__all__ = ["detect_query_concepts", "kg_boost", "linked_concepts_for_text"]

__navmap__: Final[NavMap] = {
    "title": "search_api.kg_mock",
    "synopsis": "Mock knowledge graph helpers used by the search API",
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
        name: {
            "owner": "@search-api",
            "stability": "experimental",
            "since": "0.2.0",
        }
        for name in __all__
    },
}


class ConceptMeta(TypedDict):
    """Model the ConceptMeta.

    Represent the conceptmeta data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

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

    Carry out the detect query concepts operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    query : str
    query : str
        Description for ``query``.

    Returns
    -------
    collections.abc.Set
        Description of return value.

    Examples
    --------
    >>> from search_api.kg_mock import detect_query_concepts
    >>> result = detect_query_concepts(...)
    >>> result  # doctest: +ELLIPSIS
    ...
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

    Carry out the linked concepts for text operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    text : str
    text : str
        Description for ``text``.

    Returns
    -------
    List[str]
        Description of return value.

    Examples
    --------
    >>> from search_api.kg_mock import linked_concepts_for_text
    >>> result = linked_concepts_for_text(...)
    >>> result  # doctest: +ELLIPSIS
    ...
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

    Carry out the kg boost operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    query_concepts : List[str]
    query_concepts : List[str]
        Description for ``query_concepts``.
    chunk_concepts : List[str]
    chunk_concepts : List[str]
        Description for ``chunk_concepts``.
    direct : float | None
    direct : float | None, optional, default=0.08
        Description for ``direct``.
    one_hop : float | None
    one_hop : float | None, optional, default=0.04
        Description for ``one_hop``.

    Returns
    -------
    float
        Description of return value.

    Examples
    --------
    >>> from search_api.kg_mock import kg_boost
    >>> result = kg_boost(..., ...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    _ = one_hop  # placeholder for future graph traversal heuristics
    return direct if set(query_concepts) & set(chunk_concepts) else 0.0
