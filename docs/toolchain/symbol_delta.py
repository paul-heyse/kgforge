"""Compute symbol deltas with typed configuration.

This module provides a new public API for computing deltas between symbol
indices using typed configuration objects instead of positional arguments.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docs.toolchain.config import DocsDeltaConfig


SymbolIndex = Mapping[str, Mapping[str, object]]
SymbolDelta = Mapping[str, object]


def compute_delta(
    *, config: DocsDeltaConfig, baseline: SymbolIndex, current: SymbolIndex
) -> SymbolDelta:
    """Compute a delta between symbol indices with typed configuration.

    This is the new public API for computing symbol deltas that accepts a typed
    configuration object instead of boolean positional arguments.

    Parameters
    ----------
    config : DocsDeltaConfig
        Typed configuration controlling delta computation behavior.
    baseline : Mapping[str, Mapping[str, object]]
        The baseline symbol index to compare against.
    current : Mapping[str, Mapping[str, object]]
        The current symbol index to compare.

    Returns
    -------
    Mapping[str, object]
        Delta containing removals, modifications, and additions.

    Examples
    --------
    >>> from docs.toolchain.config import DocsDeltaConfig
    >>> config = DocsDeltaConfig(include_removals=True)
    >>> # delta = compute_delta(config=config, baseline=..., current=...)
    """
    msg = "compute_delta is a placeholder for Phase 3.2 implementation"
    raise NotImplementedError(msg)


__all__ = [
    "compute_delta",
]
