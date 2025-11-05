"""Typing façade for documentation scripts.

This module re-exports the main typing helpers from `kgfoundry_common.typing`,
ensuring that documentation and artifacts generation scripts can safely import
type annotations without forcing runtime evaluation of heavy optional
dependencies like FastAPI, numpy, or FAISS.

## Usage

Use this in documentation scripts:

    from __future__ import annotations

    from typing import TYPE_CHECKING
    from docs.typing import gate_import, NavMap, ProblemDetails

    if TYPE_CHECKING:
        import numpy as np
        from fastapi import FastAPI

    def build_symbol_index(app: FastAPI | None = None) -> NavMap[str, ProblemDetails]:
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
