"""Typed facades for the ``pytestarch`` runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib import import_module
from types import ModuleType
from typing import Protocol, cast, runtime_checkable

__all__ = [
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
    """Identifier wrapper returned by ``pytestarch`` dependency edges."""

    identifier: str


ModuleDependencies = Mapping[str, Sequence[tuple[IdentifierProtocol, IdentifierProtocol]]]


@runtime_checkable
class ModuleNameFilterProtocol(Protocol):
    """Filter object for selecting modules via pytestarch queries."""

    def __init__(self, name: str) -> None: ...


@runtime_checkable
class LayeredArchitectureProtocol(Protocol):
    """Fluent builder for layered architecture rules."""

    def layer(self, name: str) -> LayeredArchitectureProtocol: ...

    def containing_modules(self, modules: Iterable[str]) -> LayeredArchitectureProtocol: ...

    @property
    def layer_mapping(self) -> Mapping[str, Sequence[ModuleNameFilterProtocol]]: ...


@runtime_checkable
class EvaluableArchitectureProtocol(Protocol):
    """Evaluable architecture surface consumed by the tooling checks."""

    modules: tuple[str, ...]

    def get_dependencies(
        self,
        sources: Iterable[ModuleNameFilterProtocol],
        targets: Iterable[ModuleNameFilterProtocol],
    ) -> ModuleDependencies: ...


def _import(name: str) -> ModuleType:
    return import_module(name)


def get_layered_architecture_cls() -> type[LayeredArchitectureProtocol]:
    module = _import("pytestarch.query_language.layered_architecture_rule")
    cls = module.LayeredArchitecture
    return cast(type[LayeredArchitectureProtocol], cls)


def get_module_name_filter_cls() -> type[ModuleNameFilterProtocol]:
    module = _import("pytestarch.eval_structure.evaluable_architecture")
    cls = module.ModuleNameFilter
    return cast(type[ModuleNameFilterProtocol], cls)


def get_layered_architecture_factory() -> Callable[[], LayeredArchitectureProtocol]:
    cls = get_layered_architecture_cls()
    return cast(Callable[[], LayeredArchitectureProtocol], cls)


def get_module_name_filter_factory() -> Callable[[str], ModuleNameFilterProtocol]:
    cls = get_module_name_filter_cls()
    return cast(Callable[[str], ModuleNameFilterProtocol], cls)


def get_evaluable_architecture_fn() -> Callable[..., EvaluableArchitectureProtocol]:
    module = _import("pytestarch.pytestarch")
    fn = module.get_evaluable_architecture
    return cast(Callable[..., EvaluableArchitectureProtocol], fn)


def get_evaluable_architecture_for_module_objects_fn() -> (
    Callable[..., EvaluableArchitectureProtocol]
):
    module = _import("pytestarch.pytestarch")
    fn = module.get_evaluable_architecture_for_module_objects
    return cast(Callable[..., EvaluableArchitectureProtocol], fn)
