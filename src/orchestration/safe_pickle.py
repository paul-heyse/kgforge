"""Safe pickle wrapper for temporary FAISS index serialization.

This module provides restricted pickle functionality that only allows
dict, list, and primitive types to prevent arbitrary code execution.

Notes
-----
This is a temporary serialization format pending full FAISS integration.
Production systems should use structured formats like Protocol Buffers or MessagePack.
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import BinaryIO, Protocol, cast

logger = logging.getLogger(__name__)

# Allow-list of safe builtin types for FAISS index serialization
_ALLOWED_TYPES = frozenset(
    {
        "builtins.dict",
        "builtins.list",
        "builtins.tuple",
        "builtins.str",
        "builtins.int",
        "builtins.float",
        "builtins.bool",
        "builtins.NoneType",
    }
)


class UnsafePickleError(ValueError):
    """Raised when pickle encounters disallowed types or suspicious patterns."""

    def __init__(self, message: str, type_name: str | None = None) -> None:
        """Initialize unsafe pickle error.

        Parameters
        ----------
        message : str
            Error description.
        type_name : str | None
            Attempted type name that was rejected.
        """
        super().__init__(message)
        self.type_name = type_name


class _UnpicklerProtocol(Protocol):
    def __init__(
        self,
        file: BinaryIO,
        *,
        fix_imports: bool = ...,
        encoding: str = ...,
        errors: str = ...,
        buffers: object | None = ...,
    ) -> None: ...

    def load(self) -> object: ...

    def find_class(self, module: str, name: str) -> object: ...


class _PickleModule(Protocol):
    Unpickler: type[_UnpicklerProtocol]

    def dump(self, obj: object, file: BinaryIO) -> None: ...

    def dumps(self, obj: object) -> bytes: ...


_stdlib_pickle = cast(_PickleModule, import_module("pickle"))
_StdlibUnpickler = cast(type[_UnpicklerProtocol], _stdlib_pickle.Unpickler)


class SafeUnpickler(_StdlibUnpickler):
    """Unpickler that enforces allow-list of safe types.

    This prevents arbitrary code execution by restricting deserialization
    to primitive types and basic containers (dict, list).
    """

    def find_class(self, module: str, name: str) -> type:
        """Override find_class to enforce allow-list.

        Parameters
        ----------
        module : str
            Module name from pickle stream.
        name : str
            Class name from pickle stream.

        Returns
        -------
        type
            The class object.

        Raises
        ------
        UnsafePickleError
            If the requested class is not in the allow-list.
        """
        full_name = f"{module}.{name}"
        if full_name not in _ALLOWED_TYPES:
            msg = f"Pickle deserialization blocked: {full_name} not in allow-list"
            raise UnsafePickleError(msg, type_name=full_name)
        return cast(type[object], super().find_class(module, name))


def dump(obj: object, file: BinaryIO) -> None:
    """Safely pickle object to file with type validation.

    Only dict, list, tuple, and primitives (str, int, float, bool, None)
    are allowed to prevent arbitrary code execution during deserialization.

    Parameters
    ----------
    obj : object
        Object to serialize. Must be dict, list, or primitives.
    file : BinaryIO
        File-like object opened in binary write mode.

    Raises
    ------
    ValueError
        If object contains disallowed types.

    Examples
    --------
    >>> import tempfile
    >>> data = {"keys": ["a", "b"], "vectors": [1.0, 2.0]}
    >>> with tempfile.NamedTemporaryFile() as f:
    ...     dump(data, f)
    """
    _validate_object(obj)
    _stdlib_pickle.dump(obj, file)


def load(file: BinaryIO) -> object:
    """Safely unpickle object from file with type restrictions.

    Parameters
    ----------
    file : BinaryIO
        File-like object opened in binary read mode.

    Returns
    -------
    object
        Deserialized object (guaranteed primitives/dict/list only).

    Raises
    ------
    UnsafePickleError
        If pickle stream contains disallowed types.

    Examples
    --------
    >>> import tempfile
    >>> data = {"keys": ["a", "b"], "vectors": [1.0, 2.0]}
    >>> with tempfile.NamedTemporaryFile() as f:
    ...     dump(data, f)
    ...     f.seek(0)
    ...     loaded = load(f)
    ...     assert loaded == data
    """
    unpickler = SafeUnpickler(file)
    loaded: object = unpickler.load()
    return loaded


def _validate_object(obj: object, depth: int = 0) -> None:
    """Recursively validate object contains only safe types.

    Parameters
    ----------
    obj : object
        Object to validate.
    depth : int
        Current recursion depth (prevents infinite recursion).

    Raises
    ------
    ValueError
        If object contains disallowed types or exceeds depth limit.
    """
    max_depth = 100
    if depth > max_depth:
        msg = f"Object nesting exceeds maximum depth ({max_depth})"
        raise ValueError(msg)

    # Base case: primitives are always safe
    if isinstance(obj, (str, int, float, bool, type(None))):
        return

    # Recursive cases: containers
    if isinstance(obj, dict):
        for key, value in obj.items():
            _validate_object(key, depth + 1)
            _validate_object(value, depth + 1)
        return

    if isinstance(obj, (list, tuple)):
        for item in obj:
            _validate_object(item, depth + 1)
        return

    # Reject all other types
    msg = f"Object type not allowed for safe pickling: {type(obj).__qualname__}"
    raise ValueError(msg)
