from __future__ import annotations
from typing import List
from kgforge.kgforge_common.models import Concept

class OntologyCatalog:
    def __init__(self, concepts: List[Concept]):
        self.by_id = {c.id: c for c in concepts}

    def neighbors(self, concept_id: str, depth: int = 1) -> List[str]:
        # TODO: return neighbor concept IDs up to depth.
        return []
