"""Typed facades for optional Sphinx dependencies.

This module provides protocols and loaders for optional Sphinx integrations
(Astroid, AutoAPI, docstring overrides). Guarded imports ensure graceful
degradation when dependencies are missing, with descriptive error messages.

Examples
--------
>>> from docs._types.sphinx_optional import load_optional_dependencies
>>> try:
...     deps = load_optional_dependencies()
... except ImportError as e:
...     print(f"Missing optional dependency: {e}")
"""

from __future__ import annotations

from importlib import import_module
from typing import Protocol, cast, runtime_checkable

from docs._types.astroid_facade import AstroidManagerProtocol
from docs._types.autoapi_parser import AutoapiParserProtocol, coerce_parser_class

__all__ = [
    "AstroidManagerFacade",
    "AutoapiParserFacade",
    "MissingDependencyError",
    "OptionalDependencies",
    "load_optional_dependencies",
]


class MissingDependencyError(ImportError):
    """Raised when an optional dependency required for docs building is missing.

    Parameters
    ----------
    module_name : str
        Name of the missing module.
    reason : str, optional
        Additional context or reason for the requirement.

    Examples
    --------
    >>> raise MissingDependencyError("sphinx_autoapi", "required for API docs generation")
    """

    def __init__(self, module_name: str, reason: str = "") -> None:
        """Initialize the error."""
        message = f"Missing optional dependency: {module_name}"
        if reason:
            message += f" ({reason})"
        super().__init__(message)
        self.module_name = module_name


type AutoapiParserFacade = type[AutoapiParserProtocol]


type AstroidManagerFacade = AstroidManagerProtocol


@runtime_checkable
class OptionalDependencies(Protocol):
    """Protocol aggregating optional dependency facades."""

    @property
    def autoapi_parser(self) -> AutoapiParserFacade:
        """Sphinx AutoAPI parser facade."""
        ...

    @property
    def astroid_manager(self) -> AstroidManagerFacade:
        """Astroid manager facade."""
        ...


class _OptionalDependenciesImpl:
    """Concrete implementation of OptionalDependencies."""

    def __init__(
        self,
        autoapi_parser: AutoapiParserFacade,
        astroid_manager: AstroidManagerFacade,
    ) -> None:
        """Initialize with optional dependency instances."""
        self._autoapi_parser = autoapi_parser
        self._astroid_manager = astroid_manager

    @property
    def autoapi_parser(self) -> AutoapiParserFacade:
        """Return the AutoAPI parser."""
        return self._autoapi_parser

    @property
    def astroid_manager(self) -> AstroidManagerFacade:
        """Return the Astroid manager."""
        return self._astroid_manager


def load_optional_dependencies() -> OptionalDependencies:
    """Load optional Sphinx dependencies with guarded imports.

    Returns
    -------
    OptionalDependencies
        Aggregate facade providing access to optional components.

    Raises
    ------
    MissingDependencyError
        If sphinx_autoapi or astroid is not installed.

    Examples
    --------
    >>> from docs._types.sphinx_optional import load_optional_dependencies
    >>> try:
    ...     deps = load_optional_dependencies()
    ...     parser = deps.autoapi_parser
    ... except MissingDependencyError as e:
    ...     print(f"Cannot load AutoAPI: {e}")
    """
    # Try to import the AutoAPI parser module
    autoapi_import_err_msg = "autoapi is required for AutoAPI docs generation"
    try:
        parser_module = import_module("autoapi._parser")
    except ImportError as e:
        autoapi_error = MissingDependencyError("autoapi._parser", autoapi_import_err_msg)
        raise autoapi_error from e
    parser_cls = coerce_parser_class(parser_module)

    # Try to import astroid
    astroid_import_err_msg = "astroid is required for AST analysis during docs build"
    try:
        astroid = import_module("astroid")
    except ImportError as e:
        astroid_error = MissingDependencyError("astroid", astroid_import_err_msg)
        raise astroid_error from e

    # If we get here, both are available
    # Return facades wrapping the actual modules/objects
    manager_facade = cast("AstroidManagerFacade", astroid.MANAGER)
    return _OptionalDependenciesImpl(
        autoapi_parser=parser_cls,
        astroid_manager=manager_facade,
    )
