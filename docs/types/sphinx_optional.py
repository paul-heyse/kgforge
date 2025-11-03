"""Public wrapper for :mod:`docs._types.sphinx_optional`."""

from __future__ import annotations

from docs._types.sphinx_optional import (
    AstroidManagerFacade,
    AutoapiParserFacade,
    MissingDependencyError,
    OptionalDependencies,
    load_optional_dependencies,
)

__all__: tuple[str, ...] = (
    "AstroidManagerFacade",
    "AutoapiParserFacade",
    "MissingDependencyError",
    "OptionalDependencies",
    "load_optional_dependencies",
)
