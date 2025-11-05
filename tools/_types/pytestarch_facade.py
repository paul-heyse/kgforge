"""Typed facades for the ``pytestarch`` runtime."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from types import ModuleType

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
    """Filter object for selecting modules via pytestarch queries.

    Constructs a filter bound to ``name``.

    Parameters
    ----------
    name : str
        Module name to filter.
    """


@runtime_checkable
class LayeredArchitectureProtocol(Protocol):
    """Fluent builder for layered architecture rules."""

    def layer(self, name: str) -> LayeredArchitectureProtocol:
        """Return a new builder stage representing ``name``."""
        ...

    def containing_modules(self, modules: Iterable[str]) -> LayeredArchitectureProtocol:
        """Restrict the current layer to ``modules``."""
        ...

    @property
    def layer_mapping(self) -> Mapping[str, Sequence[ModuleNameFilterProtocol]]:
        """Return the configured layer-to-module mapping."""
        ...


@runtime_checkable
class EvaluableArchitectureProtocol(Protocol):
    """Evaluable architecture surface consumed by the tooling checks."""

    modules: tuple[str, ...]

    def get_dependencies(
        self,
        sources: Iterable[ModuleNameFilterProtocol],
        targets: Iterable[ModuleNameFilterProtocol],
    ) -> ModuleDependencies:
        """Return dependencies between ``sources`` and ``targets``."""
        ...


def _import(name: str) -> ModuleType:
    return import_module(name)


class _PytestarchModule(Protocol):
    def get_evaluable_architecture(
        *args: object,
        **kwargs: object,
    ) -> EvaluableArchitectureProtocol:
        """Get evaluable architecture instance.

        Parameters
        ----------
        **kwargs : object
            Keyword arguments (unused).

        Returns
        -------
        EvaluableArchitectureProtocol
            Evaluable architecture instance.
        """
        ...

    def get_evaluable_architecture_for_module_objects(
        *args: object,
        **kwargs: object,
    ) -> EvaluableArchitectureProtocol:
        """Get evaluable architecture for module objects.

        Parameters
        ----------
        **kwargs : object
            Keyword arguments (unused).

        Returns
        -------
        EvaluableArchitectureProtocol
            Evaluable architecture instance.
        """
        ...


def get_layered_architecture_cls() -> type[LayeredArchitectureProtocol]:
    """Return the pytestarch layered architecture class.

    Returns
    -------
    type[LayeredArchitectureProtocol]
        Layered architecture class.

    Raises
    ------
    TypeError
        If pytestarch module is missing LayeredArchitecture class.
    """
    module = _import("pytestarch.query_language.layered_architecture_rule")
    candidate_obj = getattr(module, "LayeredArchitecture", _MISSING)
    if candidate_obj is _MISSING or not inspect.isclass(candidate_obj):
        message = "pytestarch layered architecture module is missing LayeredArchitecture class"
        raise TypeError(message)
    return cast("type[LayeredArchitectureProtocol]", candidate_obj)


def get_module_name_filter_cls() -> type[ModuleNameFilterProtocol]:
    """Return the pytestarch module-name filter class.

    Returns
    -------
    type[ModuleNameFilterProtocol]
        Module name filter class.

    Raises
    ------
    TypeError
        If pytestarch module is missing ModuleNameFilter class.
    """
    module = _import("pytestarch.eval_structure.evaluable_architecture")
    candidate_obj = getattr(module, "ModuleNameFilter", _MISSING)
    if candidate_obj is _MISSING or not inspect.isclass(candidate_obj):
        message = "pytestarch evaluable architecture module is missing ModuleNameFilter class"
        raise TypeError(message)
    return cast("type[ModuleNameFilterProtocol]", candidate_obj)


def get_layered_architecture_factory() -> Callable[[], LayeredArchitectureProtocol]:
    """Return a callable that instantiates the layered architecture builder.

    Returns
    -------
    Callable[[], LayeredArchitectureProtocol]
        Factory function for layered architecture.
    """
    return get_layered_architecture_cls()


def get_module_name_filter_factory() -> Callable[[str], ModuleNameFilterProtocol]:
    """Return a callable that creates module-name filters.

    Returns
    -------
    Callable[[str], ModuleNameFilterProtocol]
        Factory function for module name filters.
    """
    return get_module_name_filter_cls()


def get_evaluable_architecture_fn() -> Callable[..., EvaluableArchitectureProtocol]:
    """Return the pytestarch entry-point for evaluable architectures.

    Returns
    -------
    Callable[..., EvaluableArchitectureProtocol]
        Function to create evaluable architecture.

    Raises
    ------
    AttributeError
        If pytestarch module is missing get_evaluable_architecture.
    """
    module = _import("pytestarch.pytestarch")
    if not hasattr(module, "get_evaluable_architecture"):
        message = "pytestarch module is missing get_evaluable_architecture"
        raise AttributeError(message)
    typed_module = cast("_PytestarchModule", module)
    return typed_module.get_evaluable_architecture


def get_evaluable_architecture_for_module_objects_fn() -> Callable[
    ..., EvaluableArchitectureProtocol
]:
    """Return the evaluable-architecture factory for module objects.

    Returns
    -------
    Callable[..., EvaluableArchitectureProtocol]
        Function to create evaluable architecture from module objects.

    Raises
    ------
    AttributeError
        If pytestarch module is missing get_evaluable_architecture_for_module_objects.
    """
    module = _import("pytestarch.pytestarch")
    if not hasattr(module, "get_evaluable_architecture_for_module_objects"):
        message = "pytestarch module is missing get_evaluable_architecture_for_module_objects"
        raise AttributeError(message)
    typed_module = cast("_PytestarchModule", module)
    return typed_module.get_evaluable_architecture_for_module_objects
