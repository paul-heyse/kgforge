"""Gallery examples package.

========================

Expose common utilities used by the gallery doctests.

The module makes the ``examples`` namespace importable so doctest snippets can
locate helpers without mutating ``sys.path``.

.. tags:: utils, packaging

Metadata
--------

- **Title:** Gallery examples package
- **Tags:** utils, packaging
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

from examples import _utils as _utils

__all__ = ["_utils"]
