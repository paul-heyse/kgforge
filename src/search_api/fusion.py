
from __future__ import annotations
from typing import List, Tuple, Dict

def rrf_fuse(rankers: List[List[Tuple[str, float]]], k: int = 60) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    for ranked in rankers:
        for r, (key, _score) in enumerate(ranked, start=1):
            agg[key] = agg.get(key, 0.0) + 1.0 / (k + r)
    return agg
