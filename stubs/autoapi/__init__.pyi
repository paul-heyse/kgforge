from __future__ import annotations

from typing import Any

__all__ = ["__version__", "__version_info__", "setup"]

__version__: str
__version_info__: tuple[int, int, int]

def setup(app: object) -> dict[str, Any]:
    """Register the AutoAPI extension with Sphinx."""
    ...
