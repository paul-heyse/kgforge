"""Typed facades and protocols for Griffe integration.

This module provides runtime-checkable protocols that expose only the Griffe
attributes and methods consumed by the docs pipeline. Facades eliminate `Any`-typed
access and enforce type safety at integration points. Optional dependency imports
are guarded and emit RFC 9457 Problem Details on failure.

Examples
--------
>>> from docs._types.griffe import build_facade
>>> from docs._scripts.shared import detect_environment
>>> env = detect_environment()
>>> facade = build_facade(env)
>>> node = facade.loader.load("kgfoundry")
>>> is_module = node.is_module
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from docs._scripts import shared
from kgfoundry_common.errors import ArtifactDependencyError
from kgfoundry_common.optional_deps import (
    safe_import_autoapi,
    safe_import_sphinx,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from docs._scripts.shared import BuildEnvironment

__all__ = [
    "GriffeFacade",
    "GriffeNode",
    "LoaderFacade",
    "MemberIterator",
    "build_facade",
    "get_autoapi_loader",
    "get_sphinx_loader",
]

logger = logging.getLogger(__name__)


@runtime_checkable
class GriffeNode(Protocol):
    """Runtime-checkable protocol for Griffe Object nodes.

    Exposes only the subset of Griffe Object attributes and methods consumed by
    the docs pipeline.
    """

    @property
    def path(self) -> str:
        """Fully qualified path (e.g., 'pkg.mod.Class.method')."""
        ...

    @property
    def members(self) -> Mapping[str, GriffeNode]:
        """Child members indexed by name."""
        ...

    @property
    def is_package(self) -> bool:
        """True if this node is a package."""
        ...

    @property
    def is_module(self) -> bool:
        """True if this node is a module."""
        ...

    @property
    def kind(self) -> str | None:
        """Symbol kind: 'module', 'class', 'function', 'method', etc."""
        ...

    @property
    def file(self) -> str | None:
        """Relative path to the source file."""
        ...

    @property
    def lineno(self) -> int | None:
        """Starting line number in source file."""
        ...

    @property
    def endlineno(self) -> int | None:
        """Ending line number in source file."""
        ...

    @property
    def signature(self) -> object | None:
        """Signature object (may be stringified for display)."""
        ...


@runtime_checkable
class LoaderFacade(Protocol):
    """Protocol for loading Griffe nodes by package name."""

    def load(self, package: str) -> GriffeNode:
        """Load and return the module graph for the given package.

        Parameters
        ----------
        package : str
            Package name (e.g., 'kgfoundry').

        Returns
        -------
        GriffeNode
            Root node of the module graph.
        """
        ...


@runtime_checkable
class MemberIterator(Protocol):
    """Protocol for iterating over members of a Griffe node."""

    def iter_members(self, node: GriffeNode) -> Iterator[GriffeNode]:
        """Iterate over direct members of the given node.

        Parameters
        ----------
        node : GriffeNode
            Node to iterate members from.

        Yields
        ------
        GriffeNode
            Direct child nodes.
        """
        ...


@runtime_checkable
class GriffeFacade(Protocol):
    """Protocol combining loader and member iterator capabilities."""

    @property
    def loader(self) -> LoaderFacade:
        """Loader instance for fetching module graphs."""
        ...

    @property
    def member_iterator(self) -> MemberIterator:
        """Member iterator instance for traversing nodes."""
        ...


class _GriffeLoaderAdapter:
    """Adapter wrapping a Griffe loader."""

    def __init__(self, griffe_loader: LoaderFacade) -> None:
        """Initialize with a Griffe loader."""
        self._loader = griffe_loader

    def load(self, package: str) -> GriffeNode:
        """Delegate to the Griffe loader."""
        return self._loader.load(package)


class _DefaultMemberIterator:
    """Default member iterator for Griffe nodes."""

    @staticmethod
    def iter_members(node: GriffeNode) -> Iterator[GriffeNode]:
        """Yield direct members of the node."""
        yield from node.members.values()


class _TypedGriffeFacade:
    """Concrete implementation of GriffeFacade."""

    def __init__(self, loader: LoaderFacade) -> None:
        """Initialize with a loader facade."""
        self._loader = loader
        self._member_iterator = _DefaultMemberIterator()

    @property
    def loader(self) -> LoaderFacade:
        """Return the loader facade."""
        return self._loader

    @property
    def member_iterator(self) -> MemberIterator:
        """Return the member iterator."""
        return self._member_iterator


def build_facade(env: BuildEnvironment) -> GriffeFacade:
    """Return a typed facade for Griffe accessing environment metadata.

    Parameters
    ----------
    env : BuildEnvironment
        Configuration object describing the build context.

    Returns
    -------
    GriffeFacade
        Runtime-checkable protocol providing typed access to Griffe loaders.

    Raises
    ------
    ArtifactDependencyError
        If Griffe cannot be loaded or initialized.

    Examples
    --------
    >>> from docs._types.griffe import build_facade
    >>> from docs._scripts.shared import detect_environment
    >>> env = detect_environment()
    >>> facade = build_facade(env)
    >>> node = facade.loader.load("kgfoundry")
    """
    try:
        griffe_loader = shared.make_loader(env)
    except ImportError as exc:
        msg = f"Failed to initialize Griffe loader: {exc}"
        logger.exception("Griffe initialization failed", extra={"reason": str(exc)})
        raise ArtifactDependencyError(msg) from exc

    adapter = _GriffeLoaderAdapter(cast("LoaderFacade", griffe_loader))
    return _TypedGriffeFacade(adapter)


def _load_autoapi() -> object:
    """Load and return the AutoAPI module if available.

    Returns
    -------
    object
        AutoAPI module instance.

    Raises
    ------
    ArtifactDependencyError
        If AutoAPI is not installed or cannot be imported.
    """
    # Use guarded import with Problem Details support
    return safe_import_autoapi()


def get_autoapi_loader() -> object:
    """Load and return the AutoAPI loader if available.

    Returns
    -------
    object
        AutoAPI loader instance.

    Raises
    ------
    ArtifactDependencyError
        If AutoAPI is not installed or cannot be imported, with RFC 9457
        Problem Details and remediation guidance.

    Examples
    --------
    >>> try:
    ...     loader = get_autoapi_loader()
    ...     # Use loader
    ... except ArtifactDependencyError as e:
    ...     print(f"AutoAPI unavailable: {e}")
    """
    # _load_autoapi will raise ArtifactDependencyError with Problem Details
    return _load_autoapi()


def _load_sphinx() -> object:
    """Load and return the Sphinx module if available.

    Returns
    -------
    object
        Sphinx module instance.

    Raises
    ------
    ArtifactDependencyError
        If Sphinx is not installed or cannot be imported.
    """
    # Use guarded import with Problem Details support
    return safe_import_sphinx()


def get_sphinx_loader() -> object:
    """Load and return the Sphinx loader if available.

    Returns
    -------
    object
        Sphinx loader instance.

    Raises
    ------
    ArtifactDependencyError
        If Sphinx is not installed or cannot be imported, with RFC 9457
        Problem Details and remediation guidance.

    Examples
    --------
    >>> try:
    ...     loader = get_sphinx_loader()
    ...     # Use loader
    ... except ArtifactDependencyError as e:
    ...     print(f"Sphinx unavailable: {e}")
    """
    # _load_sphinx will raise ArtifactDependencyError with Problem Details
    return _load_sphinx()
