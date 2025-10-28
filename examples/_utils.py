"""Utility helpers used by Sphinx gallery examples.

The helpers in this module must stay import-safe, deterministic, and CPU-only so
that doctest and gallery generation remain fast. Keep the data tiny and avoid
network or GPU side effects.
"""

from __future__ import annotations


def tiny_corpus() -> list[dict[str, str]]:
    """Return a tiny in-memory corpus shared across gallery examples."""

    return [
        {"id": "1", "text": "cats like naps"},
        {"id": "2", "text": "dogs enjoy walks"},
        {"id": "3", "text": "birds can fly"},
    ]

