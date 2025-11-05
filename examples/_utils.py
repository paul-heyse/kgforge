"""Utility helpers used by Sphinx gallery examples.
===============================================

Explain the utility helpers used by Sphinx gallery examples.

The helpers in this module must stay import-safe, deterministic, and CPU-only so
that doctest and gallery generation remain fast. Keep the data tiny and avoid
network or GPU side effects.

.. tags:: utils, helpers

Metadata
--------

- **Title:** Utility helpers used by Sphinx gallery examples
- **Tags:** utils, helpers
- **Time:** <1s
- **GPU:** no
- **Network:** no

Constraints
-----------

- Time: <1s
- GPU: no
- Network: no
"""

from __future__ import annotations


def tiny_corpus() -> list[dict[str, str]]:
    """Return a tiny in-memory corpus shared across gallery examples.

    Returns
    -------
    list[dict[str, str]]
        Three-document corpus with ``id`` and ``text`` keys.
    """
    return [
        {"id": "1", "text": "cats like naps"},
        {"id": "2", "text": "dogs enjoy walks"},
        {"id": "3", "text": "birds can fly"},
    ]
