"""Typed facades for the ``pytestarch`` runtime."""

from __future__ import annotations

import inspect
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

_MISSING = object()


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
    candidate_obj = getattr(module, "LayeredArchitecture", _MISSING)
    if candidate_obj is _MISSING or not inspect.isclass(candidate_obj):
        message = "pytestarch layered architecture module is missing LayeredArchitecture class"
        raise TypeError(message)
    return cast(type[LayeredArchitectureProtocol], candidate_obj)


def get_module_name_filter_cls() -> type[ModuleNameFilterProtocol]:
    module = _import("pytestarch.eval_structure.evaluable_architecture")
    candidate_obj = getattr(module, "ModuleNameFilter", _MISSING)
    if candidate_obj is _MISSING or not inspect.isclass(candidate_obj):
        message = "pytestarch evaluable architecture module is missing ModuleNameFilter class"
        raise TypeError(message)
    return cast(type[ModuleNameFilterProtocol], candidate_obj)


def get_layered_architecture_factory() -> Callable[[], LayeredArchitectureProtocol]:
    return get_layered_architecture_cls()


def get_module_name_filter_factory() -> Callable[[str], ModuleNameFilterProtocol]:
    return get_module_name_filter_cls()


def get_evaluable_architecture_fn() -> Callable[..., EvaluableArchitectureProtocol]:
    module = _import("pytestarch.pytestarch")
    if not hasattr(module, "get_evaluable_architecture"):
        message = "pytestarch module is missing get_evaluable_architecture"
        raise AttributeError(message)
    typed_module = cast(_PytestarchModule, module)
    return typed_module.get_evaluable_architecture


def get_evaluable_architecture_for_module_objects_fn() -> Callable[
    ..., EvaluableArchitectureProtocol
]:
    module = _import("pytestarch.pytestarch")
    if not hasattr(module, "get_evaluable_architecture_for_module_objects"):
        message = "pytestarch module is missing get_evaluable_architecture_for_module_objects"
        raise AttributeError(message)
    typed_module = cast(_PytestarchModule, module)
    return typed_module.get_evaluable_architecture_for_module_objects
