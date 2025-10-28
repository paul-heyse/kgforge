"""Quickstart — minimal import smoke test.

.. _gallery_quickstart:

Quickstart — minimal import smoke test
======================================

Ensure the :mod:`kgfoundry` package can be imported without side effects.

.. tags:: getting-started, smoke

**Constraints**

- **Time:** <2s
- **GPU:** no
- **Network:** no

>>> import importlib
>>> module = importlib.import_module("kgfoundry")
>>> module.__name__
'kgfoundry'
"""

from __future__ import annotations
