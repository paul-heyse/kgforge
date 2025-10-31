"""Typed settings helpers for repository tooling.

The functions in this module provide a thin wrapper around
``pydantic_settings.BaseSettings`` so tooling modules can load strongly typed
configuration directly from environment variables. Validation errors are
surfaced as :class:`SettingsError` exceptions carrying RFC 9457 Problem Details
payloads, ensuring callers can emit structured responses and fail fast when
required configuration is missing.
"""

from __future__ import annotations

import inspect
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Final, Sequence, TypeVar, cast  # noqa: UP035

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from tools._shared.problem_details import JsonValue, ProblemDetailsDict, build_problem_details

__all__: Final[list[str]] = [
    "SettingsError",
    "ToolRuntimeSettings",
    "get_runtime_settings",
    "load_settings",
]


class SettingsError(RuntimeError):
    """Raised when typed tooling settings fail validation."""

    def __init__(
        self,
        message: str,
        *,
        problem: ProblemDetailsDict,
        errors: Sequence[dict[str, JsonValue]],
    ) -> None:
        super().__init__(message)
        self.problem = problem
        self.errors: tuple[dict[str, JsonValue], ...] = tuple(errors)


_SettingsT = TypeVar("_SettingsT", bound=BaseSettings)

_SETTINGS_CACHE: dict[str, ToolRuntimeSettings] = {}


def load_settings(  # noqa: UP047 - helper preserves generic subclass typing for callers
    settings_factory: Callable[[], _SettingsT],
) -> _SettingsT:
    """Instantiate settings via ``settings_factory`` with structured error handling.

    Parameters
    ----------
    settings_factory : Callable[[], _SettingsT]
        Zero-argument callable that returns a ``BaseSettings`` subclass.

    Returns
    -------
    _SettingsT
        Validated settings instance.

    Raises
    ------
    SettingsError
        Raised when validation fails and the returned errors are converted to Problem Details.
    """
    try:
        return settings_factory()
    except ValidationError as exc:
        if inspect.isclass(settings_factory):
            class_factory = cast(type[BaseSettings], settings_factory)
            settings_name = class_factory.__name__
        else:
            attr_name: object = getattr(settings_factory, "__name__", None)
            settings_name = (
                attr_name if isinstance(attr_name, str) else settings_factory.__class__.__name__
            )

        raw_errors: Sequence[object] = exc.errors()
        error_dicts: tuple[dict[str, JsonValue], ...] = tuple(
            _as_error_dict(err) for err in raw_errors
        )
        extensions: dict[str, JsonValue] = {
            "errors": list(error_dicts),
            "settings_class": settings_name,
        }
        problem = build_problem_details(
            type="https://kgfoundry.dev/problems/tool-settings-invalid",
            title="Invalid tooling settings",
            status=500,
            detail="Failed to load tooling configuration",
            instance=f"urn:tool-settings:{settings_name}:invalid",
            extensions=extensions,
        )
        message = "Failed to load tooling settings"
        raise SettingsError(message, problem=problem, errors=error_dicts) from exc


class ToolRuntimeSettings(BaseSettings):
    """Repository-wide runtime configuration for tooling helpers."""

    model_config = SettingsConfigDict(env_prefix="TOOLS_", case_sensitive=False, extra="ignore")

    exec_allowlist: tuple[str, ...] = Field(
        default=(
            "python*",
            "uv",
            "git",
            "ruff",
            "mypy",
            "pytest",
            "doctoc",
            "docformatter",
            "spectral",
            "openspec",
            "dot",
            "neato",
            "pydeps",
            "pyreverse",
        ),
        description="Glob patterns for executables allowed to run via tools._shared.proc",
    )

    @field_validator("exec_allowlist", mode="before")
    @classmethod
    def _normalise_allowlist(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return cast(tuple[str, ...], cls.model_fields["exec_allowlist"].default)
        if isinstance(value, str):
            tokens = [part.strip() for part in value.split(",") if part.strip()]
            return tuple(tokens)
        if isinstance(value, (list, tuple, set)):
            tokens = [str(part).strip() for part in value if str(part).strip()]
            return tuple(tokens)
        message = "exec_allowlist must be a comma-separated string or sequence"
        raise TypeError(message)

    def is_allowed(self, executable: Path) -> bool:
        """Return ``True`` when ``executable`` matches the configured allow list.

        Parameters
        ----------
        executable : Path
            Executable path to evaluate.

        Returns
        -------
        bool
            ``True`` when ``executable`` is permitted, otherwise ``False``.
        """
        candidate = executable.name
        absolute = str(executable)
        for pattern in self.exec_allowlist:
            if Path(pattern).is_absolute() and absolute == pattern:
                return True
            if fnmatch(candidate, pattern):
                return True
        return False


def get_runtime_settings() -> ToolRuntimeSettings:
    """Return singleton runtime settings for tooling.

    Returns
    -------
    ToolRuntimeSettings
        Cached settings instance populated from environment variables.
    """
    cached = _SETTINGS_CACHE.get("default")
    if cached is None:

        def _factory() -> ToolRuntimeSettings:
            return ToolRuntimeSettings()

        cached = load_settings(_factory)
        _SETTINGS_CACHE["default"] = cached
    return cached


def _as_error_dict(error: object) -> dict[str, JsonValue]:
    """Coerce a validation error object into a JSON-serialisable dictionary.

    Parameters
    ----------
    error : object
        Validation error structure returned by Pydantic.

    Returns
    -------
    dict[str, JsonValue]
        JSON-compatible dictionary representation of ``error``.
    """
    if isinstance(error, dict):
        return {str(key): _to_jsonable(value) for key, value in error.items()}
    return {"detail": _to_jsonable(error)}


def _to_jsonable(value: object) -> JsonValue:
    """Convert ``value`` into a Problem Details-compatible JSON value.

    Parameters
    ----------
    value : object
        Arbitrary Python value emitted during validation.

    Returns
    -------
    JsonValue
        JSON-compatible representation of ``value``.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return repr(value)
