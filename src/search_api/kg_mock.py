"""Module for search_api.kg_mock."""


from __future__ import annotations
from typing import List

_CONCEPTS = {
  "urn:concept:toy:LLM": {"label":"Large Language Model", "keywords":["llm","language model","transformer"]},
  "urn:concept:toy:Alignment": {"label":"Alignment", "keywords":["alignment","safety","rlhf"]},
}
def detect_query_concepts(query: str) -> List[str]:
    """Detect query concepts.

    Args:
        query (str): TODO.

    Returns:
        List[str]: TODO.
    """
    q = query.lower()
    hits = []
    for cid, meta in _CONCEPTS.items():
        if any(kw in q for kw in meta["keywords"]):
            hits.append(cid)
    return hits
def linked_concepts_for_text(text: str) -> List[str]:
    """Linked concepts for text.

    Args:
        text (str): TODO.

    Returns:
        List[str]: TODO.
    """
    t = text.lower(); hits = []
    for cid, meta in _CONCEPTS.items():
        if any(kw in t for kw in meta["keywords"]): hits.append(cid)
    return hits
def kg_boost(query_concepts: List[str], chunk_concepts: List[str], direct: float=0.08, one_hop: float=0.04) -> float:
    """Kg boost.

    Args:
        query_concepts (List[str]): TODO.
        chunk_concepts (List[str]): TODO.
        direct (float): TODO.
        one_hop (float): TODO.

    Returns:
        float: TODO.
    """
    return direct if set(query_concepts) & set(chunk_concepts) else 0.0
