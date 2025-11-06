"""Documentation toolchain orchestration helpers for KGFoundry."""

from __future__ import annotations

import logging

from docs.toolchain.build_symbol_index import build_symbol_index
from docs.toolchain.config import DocsDeltaConfig, DocsSymbolIndexConfig
from docs.toolchain.symbol_delta import symbol_delta
from docs.toolchain.validate_artifacts import validate_artifacts

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "DocsDeltaConfig",
    "DocsSymbolIndexConfig",
    "build_symbol_index",
    "symbol_delta",
    "validate_artifacts",
]
