"""Public wrapper for :mod:`docs._scripts.mkdocs_gen_api`."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from typing import cast

_ORIGINAL = import_module("docs._scripts.mkdocs_gen_api")

_PUBLIC_NAMES: tuple[str, ...]
_original_all: object = getattr(_ORIGINAL, "__all__", None)
if isinstance(_original_all, Iterable) and not isinstance(_original_all, (str, bytes)):
    _PUBLIC_NAMES = tuple(str(name) for name in _original_all)
else:
    _PUBLIC_NAMES = tuple(name for name in dir(_ORIGINAL) if not name.startswith("_"))

__all__ = _PUBLIC_NAMES

_NAMESPACE: dict[str, object] = globals()
for _name in __all__:
    _NAMESPACE[_name] = cast(object, getattr(_ORIGINAL, _name))
