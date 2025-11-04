"""Typing façade for tools and documentation scripts.

This module re-exports the main typing helpers from `kgfoundry_common.typing`,
ensuring that tooling scripts can safely import type annotations without
forcing runtime evaluation of heavy optional dependencies.

## Usage

Use this in documentation and tooling scripts:

    from __future__ import annotations

    from typing import TYPE_CHECKING
    from tools.typing import gate_import, NavMap, ProblemDetails

    if TYPE_CHECKING:
        import numpy as np

    def build_index(vectors: np.ndarray) -> NavMap[str, ProblemDetails]:
        ...
"""

from __future__ import annotations

# Re-export all typing helpers from the canonical source
from kgfoundry_common.typing import (
    TYPE_CHECKING,
    JSONValue,
    NavMap,
    ProblemDetails,
    SymbolID,
    gate_import,
    safe_get_type,
)

__all__ = [
    "TYPE_CHECKING",
    "JSONValue",
    "NavMap",
    "ProblemDetails",
    "SymbolID",
    "gate_import",
    "safe_get_type",
]
