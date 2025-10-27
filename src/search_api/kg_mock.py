"""Module for search_api.kg_mock.

NavMap:
- detect_query_concepts: Detect query concepts.
- linked_concepts_for_text: Linked concepts for text.
- kg_boost: Kg boost.
"""

from __future__ import annotations

_CONCEPTS = {
    "urn:concept:toy:LLM": {
        "label": "Large Language Model",
        "keywords": ["llm", "language model", "transformer"],
    },
    "urn:concept:toy:Alignment": {
        "label": "Alignment",
        "keywords": ["alignment", "safety", "rlhf"],
    },
}


def detect_query_concepts(query: str) -> list[str]:
    """Detect query concepts.

    Parameters
    ----------
    query : str
        TODO.

    Returns
    -------
    List[str]
        TODO.
    """
    q = query.lower()
    hits = []
    for cid, meta in _CONCEPTS.items():
        if any(kw in q for kw in meta["keywords"]):
            hits.append(cid)
    return hits


def linked_concepts_for_text(text: str) -> list[str]:
    """Linked concepts for text.

    Parameters
    ----------
    text : str
        TODO.

    Returns
    -------
    List[str]
        TODO.
    """
    t = text.lower()
    hits = []
    for cid, meta in _CONCEPTS.items():
        if any(kw in t for kw in meta["keywords"]):
            hits.append(cid)
    return hits


def kg_boost(
    query_concepts: list[str],
    chunk_concepts: list[str],
    direct: float = 0.08,
    one_hop: float = 0.04,
) -> float:
    """Kg boost.

    Parameters
    ----------
    query_concepts : List[str]
        TODO.
    chunk_concepts : List[str]
        TODO.
    direct : float
        TODO.
    one_hop : float
        TODO.

    Returns
    -------
    float
        TODO.
    """
    return direct if set(query_concepts) & set(chunk_concepts) else 0.0
