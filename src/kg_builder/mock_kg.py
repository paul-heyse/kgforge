
from __future__ import annotations
from typing import Dict, List, Set

class MockKG:
    """A tiny in-memory KG for demo. Maps chunk_id -> set(concept_id) and concept adjacency."""
    def __init__(self):
        self.chunk2concepts: Dict[str, Set[str]] = {}
        self.neighbors: Dict[str, Set[str]] = {}
    def add_mention(self, chunk_id: str, concept_id: str):
        self.chunk2concepts.setdefault(chunk_id, set()).add(concept_id)
    def add_edge(self, a: str, b: str):
        self.neighbors.setdefault(a, set()).add(b)
        self.neighbors.setdefault(b, set()).add(a)
    def linked_concepts(self, chunk_id: str) -> List[str]:
        return sorted(self.chunk2concepts.get(chunk_id, set()))
    def one_hop(self, concept_id: str) -> List[str]:
        return sorted(self.neighbors.get(concept_id, set()))
