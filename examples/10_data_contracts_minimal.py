"""
Title: Data contracts — schema export smoke
Tags: schema, pydantic
Time: <2s
GPU: no
Network: no

>>> from pydantic import BaseModel
>>> class DemoPayload(BaseModel):
...     value: int
>>> schema = DemoPayload.model_json_schema()
>>> "properties" in schema and schema["properties"]["value"]["type"] == "integer"
True
"""

from __future__ import annotations

