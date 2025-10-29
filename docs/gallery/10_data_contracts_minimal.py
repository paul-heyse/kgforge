"""Data contracts - schema export smoke.
====================================

Exercise the minimal data contract schema export.

Demonstrate exporting a minimal :class:`pydantic.BaseModel` schema.

.. tags:: schema, pydantic

Metadata
--------

- **Title:** Data contracts - schema export smoke
- **Tags:** schema, pydantic
- **Time:** <2s
- **GPU:** no
- **Network:** no

Constraints
-----------

- Time: <2s
- GPU: no
- Network: no

>>> from pydantic import BaseModel
>>> class DemoPayload(BaseModel):
...     value: int
>>> schema = DemoPayload.model_json_schema()
>>> "properties" in schema and schema["properties"]["value"]["type"] == "integer"
True
"""

from __future__ import annotations
