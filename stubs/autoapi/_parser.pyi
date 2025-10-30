from __future__ import annotations

from collections.abc import Mapping, Sequence

class Parser:
    def __init__(self, search_paths: Sequence[str] | None = None) -> None: ...
    def parse(self, node: object) -> Mapping[str, object] | None: ...
