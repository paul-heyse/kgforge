"""Shared CLI context for documentation tooling commands.

This module mirrors the canonical CLI configuration helpers adopted across other
CLIs. It exposes cached accessors for ``CLIToolSettings`` and related metadata
so automation wrappers can load the same augment/registry contracts without
importing the heavier entry points.
"""

from __future__ import annotations

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
)
from tools.cli_context_registry import (
    CLIContextDefinition,
    augment_for,
    context_for,
    default_version_resolver,
    definition_for,
    interface_for,
    operation_override_for,
    register_cli,
    registry_for,
    settings_for,
    tooling_metadata_for,
)

if TYPE_CHECKING:
    from tools.typer_to_openapi_cli import CLIConfig, OperationContext


REPO_ROOT = Path(__file__).resolve().parents[2]
"""Repository root used to resolve augment and registry metadata paths."""


_DOCS_DEFINITIONS: dict[str, CLIContextDefinition] = {
    "docs-validate-artifacts": CLIContextDefinition(
        command="docs-validate-artifacts",
        title="Documentation Artifact Validator",
        interface_id="docs-validate-cli",
        operation_ids={"validate": "docs.validate_artifacts"},
        bin_name="docs-validate-artifacts",
        version_resolver=default_version_resolver("kgfoundry-tools", "kgfoundry"),
    ),
    "docs-build-symbol-index": CLIContextDefinition(
        command="docs-build-symbol-index",
        title="Documentation Symbol Index Builder",
        interface_id="docs-symbol-index-cli",
        operation_ids={"build": "docs.symbol_index.build"},
        bin_name="docs-build-symbol-index",
        version_resolver=default_version_resolver("kgfoundry-tools", "kgfoundry"),
    ),
    "docs-build-graphs": CLIContextDefinition(
        command="docs-cli",
        title="Documentation Graph Builder",
        interface_id="docs-validate-cli",
        operation_ids={"build": "docs.build_graphs"},
        bin_name="docs-cli",
        version_resolver=default_version_resolver("kgfoundry-tools", "kgfoundry"),
    ),
}

for _key, _definition in _DOCS_DEFINITIONS.items():
    register_cli(_key, _definition)

_DEFAULT_COMMAND = "docs-validate-artifacts"


def get_cli_definition(command: str) -> CLIContextDefinition:
    """Return the CLI metadata definition for ``command``.

    Parameters
    ----------
    command : str
        CLI command identifier (for example ``"docs-validate-artifacts"``).

    Returns
    -------
    CLIContextDefinition
        Structured metadata describing the requested CLI command.
    """
    return definition_for(command)


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
    return settings_for(command)


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
    return context_for(command)


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
    return tooling_metadata_for(command)


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
    return augment_for(command)


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
    return registry_for(command)


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
    """
    return interface_for(command)


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
    return operation_override_for(command, subcommand=subcommand, tokens=tokens)


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
