"""Typed facades for the ``pytestarch`` runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib import import_module
from types import ModuleType
from typing import Protocol, cast, runtime_checkable

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


class EvaluableArchitectureFactory(Protocol):
    """Callable that builds an evaluable architecture object."""

    def __call__(self, *args: object, **kwargs: object) -> EvaluableArchitectureProtocol: ...


def _import(name: str) -> ModuleType:
    return import_module(name)


class _LayeredArchitectureModule(Protocol):
    LayeredArchitecture: type[LayeredArchitectureProtocol]


class _EvaluableArchitectureModule(Protocol):
    ModuleNameFilter: type[ModuleNameFilterProtocol]


class _PytestarchModule(Protocol):
    def get_evaluable_architecture(
        *args: object,
        **kwargs: object,
    ) -> EvaluableArchitectureProtocol: ...

    def get_evaluable_architecture_for_module_objects(
        *args: object,
        **kwargs: object,
    ) -> EvaluableArchitectureProtocol: ...


def get_layered_architecture_cls() -> type[LayeredArchitectureProtocol]:
    module = _import("pytestarch.query_language.layered_architecture_rule")
    candidate = getattr(module, "LayeredArchitecture", None)
    if candidate is None or not isinstance(candidate, type):
        message = "pytestarch layered architecture module is missing LayeredArchitecture class"
        raise TypeError(message)
    typed_module = cast(_LayeredArchitectureModule, module)
    return typed_module.LayeredArchitecture


def get_module_name_filter_cls() -> type[ModuleNameFilterProtocol]:
    module = _import("pytestarch.eval_structure.evaluable_architecture")
    candidate = getattr(module, "ModuleNameFilter", None)
    if candidate is None or not isinstance(candidate, type):
        message = "pytestarch evaluable architecture module is missing ModuleNameFilter class"
        raise TypeError(message)
    typed_module = cast(_EvaluableArchitectureModule, module)
    return typed_module.ModuleNameFilter


def get_layered_architecture_factory() -> Callable[[], LayeredArchitectureProtocol]:
    return get_layered_architecture_cls()


def get_module_name_filter_factory() -> Callable[[str], ModuleNameFilterProtocol]:
    return get_module_name_filter_cls()


def get_evaluable_architecture_fn() -> EvaluableArchitectureFactory:
    module = _import("pytestarch.pytestarch")
    if not hasattr(module, "get_evaluable_architecture"):
        message = "pytestarch module is missing get_evaluable_architecture"
        raise AttributeError(message)
    typed_module = cast(_PytestarchModule, module)
    return cast(EvaluableArchitectureFactory, typed_module.get_evaluable_architecture)


def get_evaluable_architecture_for_module_objects_fn() -> EvaluableArchitectureFactory:
    module = _import("pytestarch.pytestarch")
    if not hasattr(module, "get_evaluable_architecture_for_module_objects"):
        message = "pytestarch module is missing get_evaluable_architecture_for_module_objects"
        raise AttributeError(message)
    typed_module = cast(_PytestarchModule, module)
    return cast(
        EvaluableArchitectureFactory,
        typed_module.get_evaluable_architecture_for_module_objects,
    )
