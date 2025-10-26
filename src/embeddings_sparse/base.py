from __future__ import annotations
from typing import Protocol, Iterable, Tuple, Dict, List

class SparseEncoder(Protocol):
    name: str
    def encode(self, texts: List[str]) -> List[Tuple[List[int], List[float]]]: ...

class SparseIndex(Protocol):
    def build(self, docs_iterable: Iterable[Tuple[str, Dict]]) -> None: ...
    def search(self, query: str, k: int, fields: Dict | None = None) -> List[Tuple[str, float]]: ...
