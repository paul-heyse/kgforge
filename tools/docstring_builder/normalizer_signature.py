"""Utilities for resolving runtime symbols and introspecting signatures."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any

from tools.docstring_builder.harvest import SymbolHarvest
from tools.docstring_builder.models import (
    SignatureIntrospectionError,
    SymbolResolutionError,
)

__all__ = ["load_module_globals", "resolve_callable", "signature_and_hints"]


def load_module_globals(module_name: str) -> Mapping[str, Any]:
    """Return the globals dictionary for ``module_name``.

    Parameters
    ----------
    module_name:
        Name of the module to import.

    Raises
    ------
    SymbolResolutionError
        If the module cannot be imported.
    """
    if not module_name:
        return {}
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
        message = f"Module {module_name!r} could not be imported"
        raise SymbolResolutionError(message) from exc
    except ImportError as exc:  # pragma: no cover - defensive guard
        message = f"Module {module_name!r} import failed"
        raise SymbolResolutionError(message) from exc
    return module.__dict__


def resolve_callable(symbol: SymbolHarvest) -> Callable[..., Any]:
    """Return the callable referenced by ``symbol``.

    Parameters
    ----------
    symbol:
        Harvested metadata describing the symbol to resolve.

    Raises
    ------
    SymbolResolutionError
        If the symbol or one of its parents cannot be resolved.
    SignatureIntrospectionError
        If the resolved object is not callable.
    """
    try:
        module = importlib.import_module(symbol.module)
    except ModuleNotFoundError as exc:
        message = f"Module {symbol.module!r} not found for {symbol.qname}"
        raise SymbolResolutionError(message) from exc
    except ImportError as exc:  # pragma: no cover - defensive guard
        message = f"Module {symbol.module!r} import failed for {symbol.qname}"
        raise SymbolResolutionError(message) from exc

    module_parts = symbol.module.split(".")
    qname_parts = symbol.qname.split(".")
    attr_parts = qname_parts[len(module_parts) :]
    obj: Any = module
    for attr in attr_parts:
        try:
            obj = getattr(obj, attr)
        except AttributeError as exc:
            message = f"Attribute {attr!r} missing on {obj!r} while resolving {symbol.qname}"
            raise SymbolResolutionError(message) from exc

    if not callable(obj):
        message = f"Resolved object for {symbol.qname} is not callable"
        raise SignatureIntrospectionError(message)
    return obj


def signature_and_hints(
    obj: Callable[..., Any], module_globals: Mapping[str, Any] | ModuleType | None
) -> tuple[inspect.Signature, dict[str, Any]]:
    """Return the signature and evaluated type hints for ``obj``.

    Parameters
    ----------
    obj:
        Callable to introspect.
    module_globals:
        Mapping of module globals to resolve forward references while
        evaluating annotations.

    Raises
    ------
    SignatureIntrospectionError
        If the signature cannot be produced for ``obj``.
    """
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError) as exc:
        message = f"Unable to inspect signature for {obj!r}"
        raise SignatureIntrospectionError(message) from exc

    globalns: dict[str, Any] = {}
    if isinstance(module_globals, ModuleType):
        globalns.update(module_globals.__dict__)
    elif isinstance(module_globals, Mapping):
        globalns.update(module_globals)

    try:
        hints = inspect.get_annotations(obj, globalns=globalns, eval_str=True)
    except (NameError, TypeError, AttributeError, SyntaxError):
        hints = {}
    return signature, hints
