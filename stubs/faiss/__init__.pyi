from __future__ import annotations

from typing import Any

__all__ = ["METRIC_INNER_PRODUCT", "GpuClonerOptions", "StandardGpuResources"]

class StandardGpuResources:
    """Stub FAISS GPU resource."""

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class GpuClonerOptions:
    """Stub FAISS GPU cloner options."""

    use_cuvs: bool

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

METRIC_INNER_PRODUCT: int

def index_factory(*args: Any, **kwargs: Any) -> Any: ...
def index_cpu_to_gpu(*args: Any, **kwargs: Any) -> Any: ...
def normalize_L2(*args: Any, **kwargs: Any) -> None: ...
def __getattr__(name: str) -> Any: ...
