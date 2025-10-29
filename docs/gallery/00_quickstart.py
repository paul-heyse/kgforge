"""Quickstart - minimal import smoke test.
======================================

Run the quickstart import smoke test.

Ensure the :mod:`src.kgfoundry` package (installed for import as :mod:`kgfoundry`) can be
imported without side effects.

.. tags:: getting-started, smoke

Metadata
--------

- **Title:** Quickstart - minimal import smoke test
- **Tags:** getting-started, smoke
- **Time:** <2s
- **GPU:** no
- **Network:** no

Constraints
-----------

- Time: <2s
- GPU: no
- Network: no

>>> import importlib
>>> module = importlib.import_module("kgfoundry")
>>> module.__name__
'kgfoundry'
"""

from __future__ import annotations
