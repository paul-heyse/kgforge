from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol, runtime_checkable

__all__ = [
    "EvaluableArchitectureFactory",
    "EvaluableArchitectureProtocol",
    "IdentifierProtocol",
    "LayeredArchitectureProtocol",
    "ModuleDependencies",
    "ModuleNameFilterProtocol",
    "get_evaluable_architecture_fn",
    "get_evaluable_architecture_for_module_objects_fn",
    "get_layered_architecture_cls",
    "get_layered_architecture_factory",
    "get_module_name_filter_cls",
    "get_module_name_filter_factory",
]

@runtime_checkable
class IdentifierProtocol(Protocol):
    identifier: str

ModuleDependencies = Mapping[str, Sequence[tuple[IdentifierProtocol, IdentifierProtocol]]]

@runtime_checkable
class ModuleNameFilterProtocol(Protocol):
    def __init__(self, name: str) -> None: ...

@runtime_checkable
class LayeredArchitectureProtocol(Protocol):
    def layer(self, name: str) -> LayeredArchitectureProtocol: ...
    def containing_modules(self, modules: Iterable[str]) -> LayeredArchitectureProtocol: ...
    @property
    def layer_mapping(self) -> Mapping[str, Sequence[ModuleNameFilterProtocol]]: ...

@runtime_checkable
class EvaluableArchitectureProtocol(Protocol):
    modules: tuple[str, ...]

    def get_dependencies(
        self,
        sources: Iterable[ModuleNameFilterProtocol],
        targets: Iterable[ModuleNameFilterProtocol],
    ) -> ModuleDependencies: ...

class EvaluableArchitectureFactory(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> EvaluableArchitectureProtocol: ...

def get_layered_architecture_cls() -> type[LayeredArchitectureProtocol]: ...
def get_module_name_filter_cls() -> type[ModuleNameFilterProtocol]: ...
def get_layered_architecture_factory() -> Callable[[], LayeredArchitectureProtocol]: ...
def get_module_name_filter_factory() -> Callable[[str], ModuleNameFilterProtocol]: ...
def get_evaluable_architecture_fn() -> EvaluableArchitectureFactory: ...
def get_evaluable_architecture_for_module_objects_fn() -> EvaluableArchitectureFactory: ...
