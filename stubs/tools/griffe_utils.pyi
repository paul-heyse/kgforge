from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

@dataclass
class GriffeAPI:
    package: ModuleType
    object_type: type[object]
    loader_type: type[object]
    class_type: type[object]
    function_type: type[object]
    module_type: type[object]

def resolve_griffe() -> GriffeAPI: ...
