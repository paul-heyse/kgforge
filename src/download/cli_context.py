"""Shared CLI context loaders for the download command suite.

The download CLI participates in repository-wide metadata contracts handled by the public
``tools`` facade. This module encapsulates the configuration needed to load the typed augment and
registry metadata, exposing cached helpers that other modules (for example ``download.cli``) can
import without duplicating path logic.
"""

from __future__ import annotations

from collections.abc import Sequence
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
"""Repository root used to resolve augment and registry paths."""

CLI_OPERATION_IDS: dict[str, str] = {
    "harvest": "cli.download.harvest",
}
"""Mapping of subcommand names to canonical operation identifiers."""


_CLI_KEY = "download"
_CLI_DEFINITION = CLIContextDefinition(
    command="download",
    title="KGFoundry Downloader",
    interface_id="download-cli",
    operation_ids=CLI_OPERATION_IDS,
    bin_name="kgf",
    version_resolver=default_version_resolver("kgfoundry"),
)

register_cli(_CLI_KEY, _CLI_DEFINITION)

CLI_COMMAND = _CLI_DEFINITION.command
"""Logical command name used for CLI envelopes and logging."""

CLI_TITLE = _CLI_DEFINITION.title
"""Human-readable title for CLI documentation and Typer help text."""

CLI_INTERFACE_ID = _CLI_DEFINITION.interface_id
"""Registry interface identifier backing the download CLI."""


def get_cli_settings() -> CLIToolSettings:
    """Return CLI settings describing augment and registry paths.

    Returns
    -------
    CLIToolSettings
        Cached settings referencing the shared augment and registry metadata.
    """
    return settings_for(_CLI_KEY)


def get_cli_context() -> CLIToolingContext:
    """Return the cached CLI tooling context for the download interface.

    Returns
    -------
    CLIToolingContext
        Tooling context bundling augment, registry, and CLI configuration models.
    """
    return context_for(_CLI_KEY)


def get_cli_config() -> CLIConfig:
    """Return the typed CLI configuration extracted from the tooling context.

    Returns
    -------
    CLIConfig
        CLI configuration providing access to the operation context helper.
    """
    context = get_cli_context()
    return cast("CLIConfig", context.cli_config)


def get_operation_context() -> OperationContext:
    """Return the operation context used for OpenAPI and diagram generation.

    Returns
    -------
    OperationContext
        Operation context derived from the CLI configuration.
    """
    return get_cli_config().operation_context


def get_tooling_metadata() -> ToolingMetadataModel:
    """Return the composite augment and registry metadata bundle.

    Returns
    -------
    ToolingMetadataModel
        Immutable metadata bundle combining augment and registry models.
    """
    return tooling_metadata_for(_CLI_KEY)


def get_augment_metadata() -> AugmentMetadataModel:
    """Return the augment metadata model for the download CLI.

    Returns
    -------
    AugmentMetadataModel
        Augment metadata describing operations, tag groups, and extras.
    """
    return augment_for(_CLI_KEY)


def get_registry_metadata() -> RegistryMetadataModel:
    """Return the registry metadata model backing the download CLI.

    Returns
    -------
    RegistryMetadataModel
        Registry metadata containing typed interface entries.
    """
    return registry_for(_CLI_KEY)


def get_interface_metadata() -> RegistryInterfaceModel:
    """Return interface metadata for the configured download CLI.

    Returns
    -------
    RegistryInterfaceModel
        Interface metadata defined in the registry for ``CLI_INTERFACE_ID``.
    """
    return interface_for(_CLI_KEY)


def get_operation_override(
    subcommand: str, *, tokens: Sequence[str] | None = None
) -> OperationOverrideModel | None:
    """Return augment override metadata for ``subcommand`` when available.

    Parameters
    ----------
    subcommand : str
        CLI subcommand name to resolve.
    tokens : Sequence[str] | None, optional
        Optional token sequence used for fallback matching when a direct operation ID
        mapping does not exist.

    Returns
    -------
    OperationOverrideModel | None
        Augment override model when defined; otherwise ``None``.
    """
    return operation_override_for(_CLI_KEY, subcommand=subcommand, tokens=tokens)


__all__ = [
    "CLI_COMMAND",
    "CLI_INTERFACE_ID",
    "CLI_OPERATION_IDS",
    "CLI_TITLE",
    "REPO_ROOT",
    "get_augment_metadata",
    "get_cli_config",
    "get_cli_context",
    "get_cli_settings",
    "get_interface_metadata",
    "get_operation_context",
    "get_operation_override",
    "get_registry_metadata",
    "get_tooling_metadata",
]
