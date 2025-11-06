"""Shared helpers for CLI tooling metadata and configuration.

This module centralises loading of CLI augment metadata and registry entries so
all tooling scripts operate on the same typed primitives. It produces
``CLIConfig`` instances compatible with :mod:`tools.typer_to_openapi_cli` and
raises :class:`CLIConfigError` with RFC 9457 Problem Details payloads when I/O
or validation fails.

Examples
--------
>>> from pathlib import Path
>>> from tools._shared.cli_tooling import CLIToolSettings, load_cli_tooling_context
>>> settings = CLIToolSettings(
...     bin_name="kgf",
...     title="KGFoundry CLI",
...     version="0.0.0",
...     augment_path=Path("openapi/_augment_cli.yaml"),
...     registry_path=Path("tools/mkdocs_suite/api_registry.yaml"),
...     interface_id="orchestration-cli",
... )
>>> # context = load_cli_tooling_context(settings)  # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import importlib
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, NoReturn

import yaml

from tools import get_logger
from tools._shared.problem_details import ProblemDetailsParams, build_problem_details

if TYPE_CHECKING:
    from tools._shared.problem_details import ProblemDetailsDict

LOGGER = get_logger(__name__)

AugmentPayload = dict[str, object]
JSONMapping = Mapping[str, object]

_CLI_CONFIG_PROBLEM_TYPE = "https://kgfoundry.dev/problems/cli-config"
_CLI_CONFIG_PROBLEM_TITLE = "CLI configuration error"

Reader = Callable[[Path], object]


class CLIConfigError(RuntimeError):
    """Raised when CLI configuration metadata cannot be loaded or validated.

    Parameters
    ----------
    problem : ProblemDetailsDict
        RFC 9457 Problem Details payload describing the failure.
    """

    def __init__(self, problem: ProblemDetailsDict) -> None:
        detail = str(problem.get("detail", _CLI_CONFIG_PROBLEM_TITLE))
        super().__init__(detail)
        self.problem = problem


@dataclass(frozen=True, slots=True)
class AugmentConfig:
    """Immutable view of augment metadata used by CLI tooling."""

    path: Path
    payload: AugmentPayload
    operations: Mapping[str, JSONMapping]
    tag_groups: tuple[JSONMapping, ...]

    def get_operation(self, key: str) -> JSONMapping | None:
        """Return the override associated with ``key`` when present.

        Returns
        -------
        JSONMapping | None
            Operation override mapping when available, otherwise ``None``.
        """

        return self.operations.get(key)


@dataclass(frozen=True, slots=True)
class RegistryContext:
    """Immutable registry metadata describing CLI interfaces."""

    path: Path
    interfaces: Mapping[str, JSONMapping]

    def get_interface(self, interface_id: str) -> JSONMapping | None:
        """Return metadata for ``interface_id`` if available.

        Returns
        -------
        JSONMapping | None
            Interface metadata mapping when present, otherwise ``None``.
        """

        return self.interfaces.get(interface_id)


@dataclass(frozen=True, slots=True)
class CLIToolSettings:
    """Settings describing CLI tooling inputs and metadata defaults."""

    bin_name: str
    title: str
    version: str
    augment_path: Path
    registry_path: Path
    interface_id: str | None = None


@dataclass(frozen=True, slots=True)
class CLIToolingContext:
    """Composite context bundling augment, registry, and CLI configuration."""

    augment: AugmentConfig
    registry: RegistryContext
    cli_config: object


def load_cli_tooling_context(
    settings: CLIToolSettings,
    *,
    augment_reader: Reader | None = None,
    registry_reader: Reader | None = None,
    config_factory: Callable[..., object] | None = None,
) -> CLIToolingContext:
    """Return a :class:`CLIToolingContext` for the supplied ``settings``.

    Parameters
    ----------
    settings : CLIToolSettings
        CLI tooling inputs including augment/registry paths and metadata.
    augment_reader : Reader | None, optional
        Optional custom reader for augment files. Defaults to safe YAML reader.
    registry_reader : Reader | None, optional
        Optional custom reader for registry files. Defaults to safe YAML reader.
    config_factory : Callable[..., object] | None, optional
        Factory callable producing ``CLIConfig`` instances. Defaults to the
        canonical :class:`tools.typer_to_openapi_cli.CLIConfig` implementation.

    Returns
    -------
    CLIToolingContext
        Composite context containing augment data, registry metadata, and the
        constructed CLI configuration.

    Raises
    ------
    CLIConfigError
        Raised when augment or registry metadata cannot be loaded or validated.
    """
    augment = load_augment_config(settings.augment_path, reader=augment_reader)
    registry = load_registry_context(settings.registry_path, reader=registry_reader)
    cli_config = build_cli_config(
        augment=augment,
        registry=registry,
        settings=settings,
        config_factory=config_factory,
    )
    return CLIToolingContext(augment=augment, registry=registry, cli_config=cli_config)


def load_augment_config(path: Path, *, reader: Reader | None = None) -> AugmentConfig:
    """Load augment metadata from ``path``.

    Parameters
    ----------
    path : Path
        Filesystem path to the augment YAML document.
    reader : Reader | None, optional
        Optional custom reader used for testing. Defaults to YAML safe load.

    Returns
    -------
    AugmentConfig
        Immutable augment metadata bundle.

    Raises
    ------
    CLIConfigError
        Raised when the file is missing, malformed, or not a mapping.
    """
    resolved = path.resolve()
    if reader is None:
        return _cached_augment(str(resolved))
    return _load_augment(Path(resolved), reader)


def load_registry_context(path: Path, *, reader: Reader | None = None) -> RegistryContext:
    """Load CLI registry metadata from ``path``.

    Parameters
    ----------
    path : Path
        Filesystem path to the registry YAML document.
    reader : Reader | None, optional
        Optional custom reader used for testing. Defaults to YAML safe load.

    Returns
    -------
    RegistryContext
        Immutable registry metadata bundle.

    Raises
    ------
    CLIConfigError
        Raised when the file is missing, malformed, or lacks ``interfaces``.
    """
    resolved = path.resolve()
    if reader is None:
        return _cached_registry(str(resolved))
    return _load_registry(Path(resolved), reader)


def build_cli_config(
    *,
    augment: AugmentConfig,
    registry: RegistryContext,
    settings: CLIToolSettings,
    config_factory: Callable[..., object] | None = None,
) -> object:
    """Return a ``CLIConfig`` constructed from the supplied metadata."""
    factory = config_factory or _resolve_cli_config_factory()
    interface_meta: dict[str, object] | None = None
    if settings.interface_id:
        interface_meta_mapping = registry.get_interface(settings.interface_id)
        if interface_meta_mapping is None:
            _raise_cli_error(
                detail=(
                    f"Interface '{settings.interface_id}' is not defined in registry "
                    f"'{registry.path}'."
                ),
                status=422,
                instance=f"urn:cli:registry:{registry.path.name}",
                extras={
                    "path": str(registry.path),
                    "interface_id": settings.interface_id,
                },
            )
        interface_meta = dict(interface_meta_mapping)
        interface_meta.setdefault("id", settings.interface_id)

    return factory(
        bin_name=settings.bin_name,
        title=settings.title,
        version=settings.version,
        augment=augment.payload,
        interface_id=settings.interface_id,
        interface_meta=interface_meta,
    )


def _resolve_cli_config_factory() -> Callable[..., object]:
    import importlib
    module = importlib.import_module("tools.typer_to_openapi_cli")
    return getattr(module, "CLIConfig")


def _coerce_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): val for key, val in value.items()}


def _coerce_mapping_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [_coerce_mapping(entry) for entry in value if isinstance(entry, Mapping) and entry]


def _normalize_operations(value: object) -> dict[str, dict[str, object]]:
    operations: dict[str, dict[str, object]] = {}
    if not isinstance(value, Mapping):
        return operations
    for op_key, meta in value.items():
        if not isinstance(meta, Mapping):
            continue
        operations[str(op_key)] = _coerce_mapping(meta)
    return operations


def _normalize_interface_meta(meta: Mapping[str, object]) -> dict[str, object]:
    normalized = _coerce_mapping(meta)
    operations = _normalize_operations(normalized.get("operations"))
    if operations:
        normalized["operations"] = operations
    tags = normalized.get("tags")
    if tags is not None:
        normalized["tags"] = _ensure_str_list(tags)
    problem_details = normalized.get("problem_details")
    if problem_details is not None:
        normalized["problem_details"] = _ensure_str_list(problem_details)
    return normalized


def _ensure_str_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return []


def _default_yaml_reader(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _raise_cli_error(
    *,
    detail: str,
    status: int,
    instance: str,
    extras: Mapping[str, object] | None = None,
) -> NoReturn:
    problem = build_problem_details(
        ProblemDetailsParams(
            type=_CLI_CONFIG_PROBLEM_TYPE,
            title=_CLI_CONFIG_PROBLEM_TITLE,
            status=status,
            detail=detail,
            instance=instance,
            extensions=extras,
        )
    )
    LOGGER.error(detail, extra={**({} if extras is None else dict(extras)), "status": "error"})
    raise CLIConfigError(problem)


@lru_cache(maxsize=16)
def _cached_augment(path_str: str) -> AugmentConfig:
    return _load_augment(Path(path_str), _default_yaml_reader)


@lru_cache(maxsize=16)
def _cached_registry(path_str: str) -> RegistryContext:
    return _load_registry(Path(path_str), _default_yaml_reader)


def _load_augment(resolved: Path, reader: Reader) -> AugmentConfig:
    try:
        raw_payload = reader(resolved)
    except FileNotFoundError:  # pragma: no cover - OS-level behaviour
        _raise_cli_error(
            detail=f"Augment file '{resolved}' does not exist.",
            status=404,
            instance=f"urn:cli:augment:{resolved.name}",
            extras={"path": str(resolved)},
        )
    except yaml.YAMLError as exc:
        _raise_cli_error(
            detail=f"Failed to parse augment file '{resolved}': {exc}",
            status=422,
            instance=f"urn:cli:augment:{resolved.name}",
            extras={"path": str(resolved)},
        )
    except OSError as exc:  # pragma: no cover - I/O failure
        _raise_cli_error(
            detail=f"Could not read augment file '{resolved}': {exc.__class__.__name__}",
            status=500,
            instance=f"urn:cli:augment:{resolved.name}",
            extras={"path": str(resolved)},
        )

    if raw_payload is None:
        raw_payload = {}
    if not isinstance(raw_payload, Mapping):
        _raise_cli_error(
            detail=f"Augment file '{resolved}' must decode to a mapping.",
            status=422,
            instance=f"urn:cli:augment:{resolved.name}",
            extras={"path": str(resolved)},
        )

    payload = _coerce_mapping(raw_payload)
    operations = _normalize_operations(payload.get("operations"))
    payload["operations"] = {key: dict(value) for key, value in operations.items()}

    tag_groups_list = _coerce_mapping_list(payload.get("x-tagGroups"))
    payload["x-tagGroups"] = [dict(entry) for entry in tag_groups_list]

    return AugmentConfig(
        path=resolved,
        payload=payload,
        operations=MappingProxyType(
            {key: MappingProxyType(dict(value)) for key, value in operations.items()}
        ),
        tag_groups=tuple(MappingProxyType(dict(entry)) for entry in tag_groups_list),
    )


def _load_registry(resolved: Path, reader: Reader) -> RegistryContext:
    try:
        raw_payload = reader(resolved)
    except FileNotFoundError:  # pragma: no cover - OS-level behaviour
        _raise_cli_error(
            detail=f"Registry file '{resolved}' does not exist.",
            status=404,
            instance=f"urn:cli:registry:{resolved.name}",
            extras={"path": str(resolved)},
        )
    except yaml.YAMLError as exc:
        _raise_cli_error(
            detail=f"Failed to parse registry file '{resolved}': {exc}",
            status=422,
            instance=f"urn:cli:registry:{resolved.name}",
            extras={"path": str(resolved)},
        )
    except OSError as exc:  # pragma: no cover - I/O failure
        _raise_cli_error(
            detail=f"Could not read registry file '{resolved}': {exc.__class__.__name__}",
            status=500,
            instance=f"urn:cli:registry:{resolved.name}",
            extras={"path": str(resolved)},
        )

    if raw_payload is None:
        raw_payload = {}
    if not isinstance(raw_payload, Mapping):
        _raise_cli_error(
            detail=f"Registry file '{resolved}' must decode to a mapping.",
            status=422,
            instance=f"urn:cli:registry:{resolved.name}",
            extras={"path": str(resolved)},
        )

    payload = _coerce_mapping(raw_payload)
    interfaces_raw = payload.get("interfaces")
    if not isinstance(interfaces_raw, Mapping):
        _raise_cli_error(
            detail=f"Registry file '{resolved}' must expose an 'interfaces' mapping.",
            status=422,
            instance=f"urn:cli:registry:{resolved.name}",
            extras={"path": str(resolved)},
        )

    interfaces: dict[str, JSONMapping] = {}
    for interface_id, meta in interfaces_raw.items():
        if not isinstance(meta, Mapping):
            LOGGER.warning(
                "Registry interface '%s' entry is not a mapping; skipping",
                interface_id,
                extra={"status": "invalid", "path": str(resolved)},
            )
            continue
        normalized = _normalize_interface_meta(meta)
        interfaces[str(interface_id)] = MappingProxyType(normalized)

    return RegistryContext(path=resolved, interfaces=MappingProxyType(interfaces))


__all__ = [
    "AugmentConfig",
    "AugmentPayload",
    "CLIConfigError",
    "CLIToolSettings",
    "CLIToolingContext",
    "JSONMapping",
    "build_cli_config",
    "load_augment_config",
    "load_cli_tooling_context",
    "load_registry_context",
]
