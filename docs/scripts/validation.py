"""Public facade for :mod:`docs._scripts.validation` with lazy imports."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from types import ModuleType

MODULE_PATH = "docs._scripts.validation"

__all__: tuple[str, ...] = ("JsonPayload", "validate_against_schema")


def _load_module() -> ModuleType:
    """Import and return the lazily loaded validation module."""
    return import_module(MODULE_PATH)


def __getattr__(name: str) -> object:
    """Expose attributes from the lazily loaded validation module."""
    if name in __all__:
        module = _load_module()
        return cast("object", getattr(module, name))
    message = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(message)


def __dir__() -> list[str]:
    """Return the set of exported attributes provided by the validation module."""
    module = _load_module()
    exports_obj: object | None = getattr(module, "__all__", None)
    if isinstance(exports_obj, Iterable) and not isinstance(exports_obj, (str, bytes)):
        export_names = [str(name) for name in exports_obj]
    else:
        export_names = [candidate for candidate in dir(module) if not candidate.startswith("_")]
    return sorted(set(export_names))


if TYPE_CHECKING:  # pragma: no cover - typing assistance only
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from typing import Protocol

    JsonPayload = Mapping[str, object] | Sequence[object] | str | int | float | bool | None

    class ValidateAgainstSchema(Protocol):
        def __call__(
            self,
            payload: JsonPayload,
            schema_path: Path,
            *,
            artifact: str,
        ) -> None: ...

    validate_against_schema: ValidateAgainstSchema
