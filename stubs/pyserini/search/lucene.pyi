from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

__all__ = ["LuceneHit", "LuceneImpactSearcher", "LuceneSearcher"]

class LuceneHit(Protocol):
    @property
    def docid(self) -> str: ...
    @property
    def score(self) -> float: ...

class LuceneSearcher:
    """Stub Lucene searcher."""

    def __init__(self, index_dir: str, *args: object, **kwargs: object) -> None: ...
    def set_bm25(self, k1: float, b: float) -> None: ...
    def search(self, query: str, k: int) -> Sequence[LuceneHit]: ...

class LuceneImpactSearcher(LuceneSearcher):
    """Stub Lucene impact searcher."""

    def __init__(self, index_dir: str, *args: object, **kwargs: object) -> None: ...
