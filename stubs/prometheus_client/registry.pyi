from __future__ import annotations

from typing import Iterable


class CollectorRegistry:
    """Registry of Prometheus metrics."""

    def __init__(self, auto_describe: bool = ...) -> None: ...

    def register(self, metric: object) -> None: ...

    def unregister(self, metric: object) -> None: ...

    def collect(self) -> Iterable[object]: ...


