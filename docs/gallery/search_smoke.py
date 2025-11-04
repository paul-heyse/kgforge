"""Search - tiny corpus smoke (no GPU).
===================================

Exercise the tiny-corpus search smoke flow.

Exercise the bundled tiny corpus utilities without requiring accelerators.

.. tags:: search, smoke

Metadata
--------

- **Title:** Search - tiny corpus smoke (no GPU)
- **Tags:** search, smoke
- **Time:** <2s
- **GPU:** no
- **Network:** no

Constraints
-----------

- Time: <2s
- GPU: no
- Network: no

>>> from examples._utils import tiny_corpus
>>> len(tiny_corpus())
3
"""

from __future__ import annotations
