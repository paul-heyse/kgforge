from __future__ import annotations

from collections.abc import Iterable

from pytestarch.eval_structure.evaluable_architecture import EvaluableArchitecture
from pytestarch.pytestarch import (
    get_evaluable_architecture,
    get_evaluable_architecture_for_module_objects,
)
from pytestarch.query_language.layered_architecture_rule import LayeredArchitecture, LayerRule

class DiagramRule: ...
class Rule: ...

class _LayerDefinition:
    def containing_modules(self, modules: Iterable[str]) -> _LayerDefinition: ...

__all__ = [
    "DiagramRule",
    "EvaluableArchitecture",
    "LayerRule",
    "LayeredArchitecture",
    "Rule",
    "get_evaluable_architecture",
    "get_evaluable_architecture_for_module_objects",
]
