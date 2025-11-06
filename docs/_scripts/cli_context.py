"""Shared CLI context for documentation tooling commands.

This module mirrors the canonical CLI configuration helpers adopted across other
CLIs. It exposes cached accessors for ``CLIToolSettings`` and related metadata
so automation wrappers can load the same augment/registry contracts without
importing the heavier entry points.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, cast

from tools import (
    AugmentMetadataModel,
    CLIToolingContext,
    CLIToolSettings,
    OperationOverrideModel,
    RegistryInterfaceModel,
    RegistryMetadataModel,
    ToolingMetadataModel,
    load_cli_tooling_context,
)

if TYPE_CHECKING:
    from tools.typer_to_openapi_cli import CLIConfig, OperationContext


@dataclass(frozen=True)
class _CLIDefinition:
    """Static metadata describing an individual documentation CLI."""

    command: str
    title: str
    interface_id: str
    operation_ids: Mapping[str, str]


REPO_ROOT = Path(__file__).resolve().parents[2]
"""Repository root used to resolve augment and registry metadata paths."""


_CLI_DEFINITIONS: dict[str, _CLIDefinition] = {
    "docs-validate-artifacts": _CLIDefinition(
        command="docs-validate-artifacts",
        title="Documentation Artifact Validator",
        interface_id="docs-validate-cli",
        operation_ids={"validate": "docs.validate_artifacts"},
    ),
    "docs-build-symbol-index": _CLIDefinition(
        command="docs-build-symbol-index",
        title="Documentation Symbol Index Builder",
        interface_id="docs-symbol-index-cli",
        operation_ids={"build": "docs.symbol_index.build"},
    ),
    "docs-build-graphs": _CLIDefinition(
        command="docs-cli",
        title="Documentation Graph Builder",
        interface_id="docs-validate-cli",
        operation_ids={"build": "docs.build_graphs"},
    ),
}

_DEFAULT_COMMAND = "docs-validate-artifacts"


def _resolve_cli_version() -> str:
    """Return the version string advertised by the CLI metadata.

    Returns
    -------
    str
        Detected ``kgfoundry`` tooling version or ``"0.0.0"`` when unavailable.
    """
    for distribution in ("kgfoundry-tools", "kgfoundry"):
        try:
            return pkg_version(distribution)
        except PackageNotFoundError:  # pragma: no cover - editable installs fallback
            continue
    return "0.0.0"


def get_cli_definition(command: str) -> _CLIDefinition:
    """Return the CLI metadata definition for ``command``.

    Parameters
    ----------
    command : str
        CLI command identifier (for example ``"docs-validate-artifacts"``).

    Returns
    -------
    _CLIDefinition
        Structured metadata describing the requested CLI command.

    Raises
    ------
    KeyError
        Raised when ``command`` is not registered in ``_CLI_DEFINITIONS``.
    """
    try:
        return _CLI_DEFINITIONS[command]
    except KeyError as exc:  # pragma: no cover - defensive guard
        message = f"Unknown documentation CLI command: {command}"
        raise KeyError(message) from exc


@cache
def _settings_for(command: str) -> CLIToolSettings:
    definition = get_cli_definition(command)
    return CLIToolSettings(
        bin_name=definition.command,
        title=definition.title,
        version=_resolve_cli_version(),
        augment_path=REPO_ROOT / "openapi" / "_augment_cli.yaml",
        registry_path=REPO_ROOT / "tools" / "mkdocs_suite" / "api_registry.yaml",
        interface_id=definition.interface_id,
    )


def get_cli_settings(command: str = _DEFAULT_COMMAND) -> CLIToolSettings:
    """Return CLI settings for ``command`` (defaults to validation CLI).

    Parameters
    ----------
    command : str, optional
        CLI command name to resolve (defaults to ``"docs-validate-artifacts"``).

    Returns
    -------
    CLIToolSettings
        Materialised CLI settings including augment/registry paths and interface id.
    """
    return _settings_for(command)


@cache
def _context_for(command: str) -> CLIToolingContext:
    return load_cli_tooling_context(get_cli_settings(command))


def get_cli_context(command: str = _DEFAULT_COMMAND) -> CLIToolingContext:
    """Return the cached CLI tooling context for ``command``.

    Parameters
    ----------
    command : str, optional
        CLI command name to resolve.

    Returns
    -------
    CLIToolingContext
        Cached context with augment, registry, and typed configuration objects.
    """
    return _context_for(command)


def get_cli_config(command: str = _DEFAULT_COMMAND) -> CLIConfig:
    """Return the typed CLI configuration consumed by downstream tooling.

    Parameters
    ----------
    command : str, optional
        CLI command name to resolve.

    Returns
    -------
    CLIConfig
        Typed CLI configuration derived from augment + registry metadata.
    """
    context = get_cli_context(command)
    return cast("CLIConfig", context.cli_config)


def get_operation_context(command: str = _DEFAULT_COMMAND) -> OperationContext:
    """Return the helper used to construct OpenAPI operation payloads.

    Parameters
    ----------
    command : str, optional
        CLI command name to resolve.

    Returns
    -------
    OperationContext
        Helper object used for augment and OpenAPI operation lookups.
    """
    return get_cli_config(command).operation_context


def get_tooling_metadata(command: str = _DEFAULT_COMMAND) -> ToolingMetadataModel:
    """Return the composite augment/registry metadata bundle for ``command``.

    Parameters
    ----------
    command : str, optional
        CLI command name to resolve.

    Returns
    -------
    ToolingMetadataModel
        Immutable bundle combining augment and registry metadata models.
    """
    context = get_cli_context(command)
    return ToolingMetadataModel(augment=context.augment, registry=context.registry)


def get_augment_metadata(command: str = _DEFAULT_COMMAND) -> AugmentMetadataModel:
    """Return augment metadata describing operations for ``command``.

    Parameters
    ----------
    command : str, optional
        CLI command name to resolve.

    Returns
    -------
    AugmentMetadataModel
        Augment metadata scoped to the selected CLI command.
    """
    return get_cli_context(command).augment


def get_registry_metadata(command: str = _DEFAULT_COMMAND) -> RegistryMetadataModel:
    """Return registry metadata describing ``command``'s CLI interface.

    Parameters
    ----------
    command : str, optional
        CLI command name to resolve.

    Returns
    -------
    RegistryMetadataModel
        Registry metadata for the selected CLI command.
    """
    return get_cli_context(command).registry


def get_interface_metadata(command: str = _DEFAULT_COMMAND) -> RegistryInterfaceModel:
    """Return the registry interface metadata for ``command``.

    Parameters
    ----------
    command : str, optional
        CLI command name to resolve.

    Returns
    -------
    RegistryInterfaceModel
        Registry interface entry for the selected CLI command.

    Raises
    ------
    KeyError
        Raised when the registry metadata does not contain the expected interface.
    """
    interface_id = get_cli_definition(command).interface_id
    interface = get_registry_metadata(command).interface(interface_id)
    if interface is None:  # pragma: no cover - defensive guard for misconfiguration
        msg = f"Registry metadata missing interface '{interface_id}'."
        raise KeyError(msg)
    return interface


def get_operation_override(
    subcommand: str,
    *,
    command: str = _DEFAULT_COMMAND,
    tokens: tuple[str, ...] | None = None,
) -> OperationOverrideModel | None:
    """Return augment override metadata for ``subcommand`` when available.

    Parameters
    ----------
    subcommand : str
        Subcommand identifier whose override metadata should be fetched.
    command : str, optional
        CLI command name to resolve (defaults to validation CLI).
    tokens : tuple[str, ...] | None, optional
        Optional token sequence for nested command structures.

    Returns
    -------
    OperationOverrideModel | None
        Override metadata when defined; otherwise ``None``.
    """
    operation_id = get_cli_definition(command).operation_ids.get(subcommand)
    if operation_id is None:
        return None
    return get_augment_metadata(command).operation_override(operation_id, tokens=tokens)


CLI_COMMAND = get_cli_definition(_DEFAULT_COMMAND).command
CLI_TITLE = get_cli_definition(_DEFAULT_COMMAND).title
CLI_INTERFACE_ID = get_cli_definition(_DEFAULT_COMMAND).interface_id
CLI_OPERATION_IDS = dict(get_cli_definition(_DEFAULT_COMMAND).operation_ids)


__all__ = [
    "CLI_COMMAND",
    "CLI_INTERFACE_ID",
    "CLI_OPERATION_IDS",
    "CLI_TITLE",
    "REPO_ROOT",
    "get_augment_metadata",
    "get_cli_config",
    "get_cli_context",
    "get_cli_definition",
    "get_cli_settings",
    "get_interface_metadata",
    "get_operation_context",
    "get_operation_override",
    "get_registry_metadata",
    "get_tooling_metadata",
]
