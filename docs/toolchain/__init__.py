"""Documentation toolchain for KGFoundry.

This package provides a unified interface for documentation operations with
typed configuration objects instead of positional arguments.

New Public APIs
---------------
- :func:`build_symbol_index` — Build documentation symbol indices with typed config
- :func:`compute_delta` — Generate symbol deltas between indices
- :func:`validate_artifacts` — Validate documentation artifacts
- :class:`DocsSymbolIndexConfig` — Configuration for symbol index building
- :class:`DocsDeltaConfig` — Configuration for delta computation

Examples
--------
Build a symbol index with configuration:

>>> from docs.toolchain.config import DocsSymbolIndexConfig
>>> from docs.toolchain.build_symbol_index import build_symbol_index
>>> config = DocsSymbolIndexConfig(include_private=False, output_format="json")
>>> # index = build_symbol_index(config=config)

Compute a delta with configuration:

>>> from docs.toolchain.config import DocsDeltaConfig
>>> from docs.toolchain.symbol_delta import compute_delta
>>> config = DocsDeltaConfig(include_removals=True, severity_threshold="info")
>>> # delta = compute_delta(config=config, baseline={...}, current={...})
"""

from __future__ import annotations

import logging

from docs.toolchain.build_symbol_index import build_symbol_index
from docs.toolchain.config import DocsDeltaConfig, DocsSymbolIndexConfig
from docs.toolchain.symbol_delta import compute_delta
from docs.toolchain.validate_artifacts import validate_artifacts

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "DocsDeltaConfig",
    "DocsSymbolIndexConfig",
    "build_symbol_index",
    "compute_delta",
    "validate_artifacts",
]
